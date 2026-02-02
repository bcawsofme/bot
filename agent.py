import json
from typing import Any, Dict, Union


def call_llm(prompt: str) -> Dict[str, str]:
    """
    LLM stub. Replace with a real model call that returns a dict.
    The contract requires JSON with keys: thought, action, reason.
    """
    # Placeholder behavior: stop immediately.
    return {
        "thought": "stub",
        "action": "stop",
        "reason": "LLM not implemented",
    }


class Agent:
    """Minimal observe → decide → act agent loop."""

    def __init__(self, goal: str, context: Union[str, Dict[str, Any]]) -> None:
        # Store inputs to anchor decisions and maintain state.
        self.goal = goal
        self.context = context
        self.step = 0
        self.log: list[Dict[str, Any]] = []

    def observe(self) -> Dict[str, Any]:
        # Capture the current state for the decision step.
        return {
            "goal": self.goal,
            "context": self.context,
            "step": self.step,
        }

    def decide(self, observation: Dict[str, Any]) -> Dict[str, str]:
        # The LLM is treated as a pure dependency that returns structured output.
        prompt = (
            "Return JSON with fields: thought, action (continue|stop), reason. "
            f"Observation: {json.dumps(observation)}"
        )
        decision = call_llm(prompt)

        # Minimal validation to enforce one decision per loop.
        if decision.get("action") not in {"continue", "stop"}:
            return {
                "thought": "invalid action",
                "action": "stop",
                "reason": "LLM returned unsupported action",
            }
        return decision

    def act(self, decision: Dict[str, str]) -> Dict[str, Any]:
        # Record the decision and advance state; keep side effects explicit.
        self.step += 1
        record = {
            "step": self.step,
            "decision": decision,
        }
        self.log.append(record)
        return record

    def run(self, max_steps: int = 5) -> Dict[str, Any]:
        # Prevent infinite loops with a clear stop condition.
        while self.step < max_steps:
            observation = self.observe()
            decision = self.decide(observation)
            record = self.act(decision)
            if decision["action"] == "stop":
                return {
                    "status": "stopped",
                    "record": record,
                    "log": self.log,
                }
        return {
            "status": "max_steps_reached",
            "record": self.log[-1] if self.log else None,
            "log": self.log,
        }


if __name__ == "__main__":
    # Example run; outputs structured JSON only.
    agent = Agent(goal="demo", context={"note": "example"})
    print(json.dumps(agent.run(max_steps=3)))
