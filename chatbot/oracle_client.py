"""
oracle_client.py — Lightweight ORACLE overseer integration.

The OracleClient handles:
  - Registering this chatbot child with the ORACLE overseer via HTTP
  - Sending periodic heartbeats so ORACLE knows the child is alive
  - Receiving tasks dispatched by ORACLE and forwarding them to the agent
  - Reporting task results back to ORACLE

When no ORACLE endpoint is configured the client operates in "standalone"
mode – all methods become no-ops so the chatbot can still be used locally
without a running ORACLE instance.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from urllib import request as urllib_request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Default heartbeat interval in seconds
_DEFAULT_HEARTBEAT_INTERVAL = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentInfo:
    """Metadata sent to the ORACLE overseer during registration."""

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "chatbot-child"
    version: str = "1.0.0"
    capabilities: list = field(default_factory=lambda: ["chat", "tool_use", "memory"])
    description: str = "Agentic AI chatbot child managed by ORACLE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "description": self.description,
        }


@dataclass
class OracleTask:
    """A task dispatched by ORACLE to this child agent."""

    task_id: str
    instruction: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OracleTask":
        return cls(
            task_id=data["task_id"],
            instruction=data["instruction"],
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# OracleClient
# ---------------------------------------------------------------------------

class OracleClient:
    """
    Manages the lifecycle of this chatbot child as seen by the ORACLE overseer.

    Parameters
    ----------
    oracle_url:
        Base URL of the ORACLE overseer API, e.g. ``http://localhost:8080``.
        Pass ``None`` (or omit) to run in standalone mode.
    agent_info:
        Metadata about this child agent.  Defaults to a sensible preset.
    heartbeat_interval:
        Seconds between heartbeat pings sent to ORACLE.
    task_handler:
        Optional callback invoked whenever ORACLE dispatches a task.
        Signature: ``(task: OracleTask) -> str`` where the return value is
        the task result to send back to ORACLE.
    """

    def __init__(
        self,
        oracle_url: Optional[str] = None,
        agent_info: Optional[AgentInfo] = None,
        heartbeat_interval: int = _DEFAULT_HEARTBEAT_INTERVAL,
        task_handler: Optional[Callable[[OracleTask], str]] = None,
    ) -> None:
        self.oracle_url = oracle_url.rstrip("/") if oracle_url else None
        self.info = agent_info or AgentInfo()
        self.heartbeat_interval = heartbeat_interval
        self.task_handler = task_handler

        self._registered = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def standalone(self) -> bool:
        """True when no ORACLE URL was configured."""
        return self.oracle_url is None

    def start(self) -> None:
        """Register with ORACLE and begin sending heartbeats."""
        if self.standalone:
            logger.info("OracleClient: running in standalone mode (no overseer URL).")
            return
        self._register()
        self._start_heartbeat()

    def stop(self) -> None:
        """Deregister from ORACLE and stop the heartbeat thread."""
        if self.standalone:
            return
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        self._deregister()

    def report_result(self, task_id: str, result: str) -> None:
        """Send a task result back to the ORACLE overseer."""
        if self.standalone:
            logger.debug("OracleClient [standalone]: task result for %s: %s", task_id, result)
            return
        payload = {
            "agent_id": self.info.agent_id,
            "task_id": task_id,
            "result": result,
        }
        self._post("/api/v1/tasks/result", payload)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register(self) -> None:
        payload = self.info.to_dict()
        ok = self._post("/api/v1/agents/register", payload)
        if ok:
            self._registered = True
            logger.info(
                "OracleClient: registered as '%s' (id=%s).",
                self.info.name,
                self.info.agent_id,
            )
        else:
            logger.warning("OracleClient: registration failed – running unregistered.")

    def _deregister(self) -> None:
        if not self._registered:
            return
        self._post("/api/v1/agents/deregister", {"agent_id": self.info.agent_id})
        self._registered = False
        logger.info("OracleClient: deregistered agent %s.", self.info.agent_id)

    def _start_heartbeat(self) -> None:
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="oracle-heartbeat",
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.heartbeat_interval):
            self._post(
                "/api/v1/agents/heartbeat",
                {"agent_id": self.info.agent_id},
            )

    def _post(self, path: str, payload: Dict[str, Any]) -> bool:
        """HTTP POST helper.  Returns True on 2xx, False otherwise."""
        url = f"{self.oracle_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
        except URLError as exc:
            logger.warning("OracleClient: POST %s failed – %s", path, exc)
            return False
        except Exception as exc:
            logger.warning("OracleClient: unexpected error on POST %s – %s", path, exc)
            return False
