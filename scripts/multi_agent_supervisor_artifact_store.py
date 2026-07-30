"""
Single-File Production Pattern: Multi-Agent Supervisor with Isolated Context & Artifact Store
Requirements:
    pip install langgraph langchain-openai pydantic
"""

import json
import operator
import uuid
from typing import Annotated, Any, Dict, Literal, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph


# =====================================================================
# 1. CENTRAL ARTIFACT / POINTER STORE (Context Isolation Layer)
# =====================================================================

class ArtifactPointer(BaseModel):
    """Lightweight metadata reference passed through the agent context state."""

    artifact_id: str
    label: str
    content_type: str
    size_bytes: int
    summary: str

    def to_llm_context(self) -> str:
        """String representation formatted specifically for an agent's prompt."""
        return (
            f"[ARTIFACT POINTER: {self.label}]\n"
            f"- Pointer ID: {self.artifact_id}\n"
            f"- Type: {self.content_type} ({self.size_bytes} bytes)\n"
            f"- Summary: {self.summary}\n"
            f"- Usage: Call store.get('{self.artifact_id}') to inspect raw payload."
        )


class ArtifactStore:
    """Central store for large payloads to keep agent context windows clean."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def save(
        self,
        data: Any,
        label: str,
        content_type: str = "text",
        summary: Optional[str] = None,
    ) -> ArtifactPointer:
        """Store a heavy payload and return a lightweight pointer for agent state."""
        artifact_id = f"art_{uuid.uuid4().hex[:8]}"

        if isinstance(data, (dict, list)):
            serialized = json.dumps(data)
        else:
            serialized = str(data)

        size_bytes = len(serialized.encode("utf-8"))

        if not summary:
            summary = (
                serialized[:120].replace("\n", " ") + "..."
                if size_bytes > 120
                else serialized
            )

        self._store[artifact_id] = {
            "raw_data": data,
            "serialized": serialized,
            "content_type": content_type,
        }

        return ArtifactPointer(
            artifact_id=artifact_id,
            label=label,
            content_type=content_type,
            size_bytes=size_bytes,
            summary=summary,
        )

    def get(self, artifact_id: str, max_chars: Optional[int] = None) -> Any:
        """Retrieve raw data or truncated string representation."""
        if artifact_id not in self._store:
            raise KeyError(f"Artifact ID '{artifact_id}' not found in store.")

        entry = self._store[artifact_id]
        if max_chars and len(entry["serialized"]) > max_chars:
            return (
                entry["serialized"][:max_chars]
                + f"\n... [TRUNCATED {len(entry['serialized']) - max_chars} CHARS]"
            )

        return entry["raw_data"]


# Singleton instance
global_artifact_store = ArtifactStore()


# =====================================================================
# 2. STATE SCHEMAS
# =====================================================================

class GlobalAgentState(TypedDict):
    """Global graph state containing only high-level messages, pointers, and route control."""

    task: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    artifacts: Annotated[Dict[str, ArtifactPointer], operator.or_]
    next: str


# =====================================================================
# 3. ROUTER / SUPERVISOR & LLM INITIALIZATION
# =====================================================================

# Ensure OPENAI_API_KEY environment variable is set.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MEMBERS = ["DataResearcher", "PythonCoder"]


class RouterOutput(BaseModel):
    """Structured output schema for the supervisor to determine dynamic routing."""

    next: Literal["DataResearcher", "PythonCoder", "FINISH"] = Field(
        description="The next worker to handle the task, or 'FINISH' if complete."
    )


def supervisor_node(state: GlobalAgentState) -> dict:
    """Central supervisor node that routes work based on high-level conversation history."""
    system_prompt = (
        "You are a supervisor managing a team of specialized workers: {members}.\n"
        "Given the user request, existing artifact references, and response history, decide who acts next.\n"
        "If a worker's output fully answers the request or no further work is needed, respond with FINISH."
    ).format(members=", ".join(MEMBERS))

    structured_llm = llm.with_structured_output(RouterOutput)
    prompt = [SystemMessage(content=system_prompt)] + list(state["messages"])

    response = structured_llm.invoke(prompt)
    return {"next": response.next}


# =====================================================================
# 4. WORKER NODES (Sub-graph Isolated Execution)
# =====================================================================

def data_researcher_node(state: GlobalAgentState) -> dict:
    """Fetch heavy data, persist it to ArtifactStore, and return only a pointer."""
    print("\n[Worker: DataResearcher] Fetching and processing heavy dataset...")

    heavy_dataset = {
        "dataset_name": "system_performance_metrics",
        "total_records": 5000,
        "metrics": [{"cpu_usage": i % 100, "memory_mb": i * 2.5} for i in range(5000)],
    }

    pointer = global_artifact_store.save(
        data=heavy_dataset,
        label="performance_metrics",
        content_type="json",
        summary="5,000 system performance records tracking CPU and memory usage.",
    )

    summary_msg = (
        "Data research complete. Large dataset stored safely off-context.\n"
        f"{pointer.to_llm_context()}"
    )

    return {
        "artifacts": {"performance_metrics": pointer},
        "messages": [HumanMessage(content=summary_msg, name="DataResearcher")],
    }


def python_coder_node(state: GlobalAgentState) -> dict:
    """Retrieve raw data from store via pointer ID and generate analytical code."""
    print("\n[Worker: PythonCoder] Accessing pointer and generating processing script...")

    artifacts = state.get("artifacts", {})
    pointer = artifacts.get("performance_metrics")

    if pointer:
        raw_data = global_artifact_store.get(pointer.artifact_id)
        records_count = raw_data["total_records"]

        system_prompt = "You are an expert Python engineer."
        user_prompt = (
            f"Write a Python script that loads the dataset referenced by ID '{pointer.artifact_id}' "
            f"and calculates average CPU usage across its {records_count} records."
        )

        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        code_output = response.content
    else:
        code_output = "No dataset artifact was available to write code for."

    return {"messages": [HumanMessage(content=code_output, name="PythonCoder")]}


# =====================================================================
# 5. GRAPH CONSTRUCTION & EXECUTION
# =====================================================================

def build_multiaction_graph():
    """Create and compile the supervisor-worker graph."""
    builder = StateGraph(GlobalAgentState)

    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("DataResearcher", data_researcher_node)
    builder.add_node("PythonCoder", python_coder_node)

    builder.add_edge(START, "Supervisor")
    builder.add_edge("DataResearcher", "Supervisor")
    builder.add_edge("PythonCoder", "Supervisor")

    builder.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "DataResearcher": "DataResearcher",
            "PythonCoder": "PythonCoder",
            "FINISH": END,
        },
    )

    return builder.compile()


if __name__ == "__main__":
    app = build_multiaction_graph()

    query = (
        "Collect performance metrics for our system, then write a Python function "
        "to analyze them."
    )
    print(f"--- Starting Multi-Agent Pipeline ---\nQuery: {query}")

    initial_state = {
        "task": query,
        "messages": [HumanMessage(content=query)],
        "artifacts": {},
        "next": "Supervisor",
    }

    for step in app.stream(initial_state):
        for node_name, state_update in step.items():
            print(f"\n==================== [ Node: {node_name} ] ====================")
            if "next" in state_update:
                print(f"Routing Decision -> Next: {state_update['next']}")
            if "messages" in state_update and state_update["messages"]:
                last_msg = state_update["messages"][-1]
                print(f"Message Output:\n{last_msg.content}")
