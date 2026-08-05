"""
Affective Generative Control system for the Aethera humanoid.

This module maps sensor inputs to *emotive states* (Valence × Arousal) which
then drive both physical motion and generative art patterns rendered on the
robot's e-ink / fiber-optic skin.

In production the ``update_state`` method would call PyTorch models for
sentiment analysis and computer vision; the logic here is a lightweight
placeholder that can be swapped out without changing the public interface.
"""

from __future__ import annotations

import random
import time
from typing import Any


class AetheraAffectiveEngine:
    """Simulates the Affective Generative Control system for the Aethera humanoid.

    Internal state is represented as a 2-D *Valence–Arousal* (VA) vector:

    * **Valence** – subjective pleasantness, in ``[-1.0, 1.0]``.
    * **Arousal** – activation / energy level, in ``[0.0, 1.0]``.

    The VA state is updated from raw sensor readings and then translated into
    generative-art parameters for the robot's expressive skin surface.
    """

    def __init__(self) -> None:
        self.state: dict[str, float] = {"valence": 0.0, "arousal": 0.2}
        self.current_pattern: str = "Neutral Flow"

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state(self, sensor_data: dict[str, Any]) -> None:
        """Process incoming sensor data and shift the robot's internal VA state.

        Args:
            sensor_data: Mapping that may contain:

                * ``"light"`` (float, 0–1) – ambient light level.
                * ``"interaction"`` (str) – one of ``"friendly"``,
                  ``"hostile"``, or ``"none"``.
        """
        light_level = float(sensor_data.get("light", 0.5))
        interaction_type = str(sensor_data.get("interaction", "none"))

        if interaction_type == "friendly":
            self.state["valence"] = min(1.0, self.state["valence"] + 0.2)
            self.state["arousal"] = min(1.0, self.state["arousal"] + 0.1)
        elif interaction_type == "hostile":
            self.state["valence"] = max(-1.0, self.state["valence"] - 0.3)
            self.state["arousal"] = min(1.0, self.state["arousal"] + 0.4)

        # Ambient light contributes a small arousal nudge
        self.state["arousal"] = (self.state["arousal"] + light_level * 0.1) / 1.1

    # ------------------------------------------------------------------
    # Art parameter generation
    # ------------------------------------------------------------------

    def generate_art_parameters(self) -> dict[str, Any]:
        """Translate the current VA state into skin-rendering parameters.

        Returns:
            Dictionary with keys:

            * ``"pattern"`` – name of the active generative pattern.
            * ``"palette"`` – list of colour names for the renderer.
            * ``"animation_speed"`` – qualitative speed description.
            * ``"fiber_optic_intensity"`` – brightness percentage (0–100).
        """
        v = self.state["valence"]
        a = self.state["arousal"]

        if v > 0.5 and a > 0.5:
            self.current_pattern = "Golden Nebula"
            colors = ["Gold", "White", "Soft Cyan"]
            speed = "Fluid / Rapid"
        elif v < -0.5:
            self.current_pattern = "Deep Pulse"
            colors = ["Deep Red", "Black", "Dark Violet"]
            speed = "Erratic / Sharp"
        else:
            self.current_pattern = "Azure Ripple"
            colors = ["Deep Blue", "Cyan", "Teal"]
            speed = "Slow / Rhythmic"

        return {
            "pattern": self.current_pattern,
            "palette": colors,
            "animation_speed": speed,
            "fiber_optic_intensity": a * 100,
        }

    # ------------------------------------------------------------------
    # Demo / simulation helper
    # ------------------------------------------------------------------

    def simulate_interaction(self, steps: int = 5, sleep_s: float = 1.0) -> None:
        """Run a short simulation loop, printing state and art parameters.

        Args:
            steps:   Number of time-steps to simulate.
            sleep_s: Seconds to pause between steps (set to 0 in tests).
        """
        interactions = ["none", "friendly", "hostile", "none"]
        print("--- Aethera Affective Engine Simulation ---")
        for i in range(steps):
            event = random.choice(interactions)
            print(f"\nTime T+{i}s | Input Event: {event}")
            self.update_state({"light": random.uniform(0.2, 0.8), "interaction": event})

            params = self.generate_art_parameters()
            print(
                f"State: Valence={self.state['valence']:.2f}, "
                f"Arousal={self.state['arousal']:.2f}"
            )
            print(f"Skin Pattern: {params['pattern']}")
            print(f"Visual Palette: {', '.join(params['palette'])}")
            print(
                f"Fiber-Optic Nerves: {params['fiber_optic_intensity']:.1f}% Brightness"
            )
            if sleep_s > 0:
                time.sleep(sleep_s)


if __name__ == "__main__":
    engine = AetheraAffectiveEngine()
    engine.simulate_interaction()
