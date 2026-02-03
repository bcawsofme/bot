import os
from typing import Any, Dict


DEMO_MODE = os.getenv("AGENT_DEMO") == "1"
_demo_step = 0


def call_llm(prompt: str) -> Dict[str, Any]:
    """
    LLM stub. Replace with a real model call that returns a dict.
    The contract requires JSON with keys: thought, action{name,args}.
    """
    # Optional demo mode to exercise tools and reflection without a real LLM.
    if DEMO_MODE:
        global _demo_step
        if "Return JSON with fields: thought, action{name,args}" in prompt:
            if _demo_step == 0:
                _demo_step += 1
                return {
                    "thought": "read demo file",
                    "action": {"name": "read_file", "args": {"path": "demo.txt"}},
                }
            if _demo_step == 1:
                _demo_step += 1
                return {
                    "thought": "repeat read",
                    "action": {"name": "read_file", "args": {"path": "demo.txt"}},
                }
            if _demo_step == 2:
                _demo_step += 1
                return {
                    "thought": "repeat read again",
                    "action": {"name": "read_file", "args": {"path": "demo.txt"}},
                }
            return {
                "thought": "stop",
                "action": {"name": "stop", "args": {"reason": "demo complete"}},
            }
        # Reflection prompt
        return {
            "outcome": "success",
            "what_worked": ["tool use", "memory write"],
            "what_failed": [],
            "next_time_try": ["validate outputs"],
            "confidence": 0.8,
        }

    # Placeholder behavior: stop immediately.
    return {
        "thought": "stub",
        "action": {
            "name": "stop",
            "args": {"reason": "LLM not implemented"},
        },
    }
