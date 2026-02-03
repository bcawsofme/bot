import json
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from utils import normalize_tokens


class GuardrailManager:
    """Centralized guardrails to bound execution and prevent runaway behavior."""

    def __init__(
        self,
        max_steps: int = 5,
        max_tool_calls: int = 10,
        max_calls_per_tool: int = 5,
        max_token_budget: int = 2000,
        repetition_threshold: int = 2,
        invalid_json_limit: int = 2,
        invalid_tool_limit: int = 2,
        wall_clock_timeout_sec: Optional[float] = None,
        drift_check_interval: int = 2,
        drift_threshold: float = 0.6,
    ) -> None:
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_calls_per_tool = max_calls_per_tool
        self.max_token_budget = max_token_budget
        self.repetition_threshold = repetition_threshold
        self.invalid_json_limit = invalid_json_limit
        self.invalid_tool_limit = invalid_tool_limit
        self.wall_clock_timeout_sec = wall_clock_timeout_sec
        self.drift_check_interval = drift_check_interval
        self.drift_threshold = drift_threshold

        self.start_time = time.time()
        self.tool_calls_total = 0
        self.tool_calls_by_name: Dict[str, int] = {}
        self.tool_call_history: Deque[str] = deque(maxlen=repetition_threshold + 1)
        self.estimated_tokens = 0
        self.invalid_json_count = 0
        self.invalid_tool_count = 0

    def estimate_tokens(self, text: str) -> int:
        # Rough heuristic: average 4 chars per token.
        return max(1, len(text) // 4)

    def record_llm_call(self, prompt: str) -> Optional[Dict[str, Any]]:
        self.estimated_tokens += self.estimate_tokens(prompt)
        if self.estimated_tokens > self.max_token_budget:
            return {
                "type": "token_budget_exceeded",
                "detail": {
                    "budget": self.max_token_budget,
                    "used": self.estimated_tokens,
                },
            }
        return None

    def record_invalid_json(self) -> Optional[Dict[str, Any]]:
        self.invalid_json_count += 1
        if self.invalid_json_count > self.invalid_json_limit:
            return {"type": "invalid_json_limit_exceeded", "detail": self.invalid_json_count}
        return None

    def record_invalid_tool(self) -> Optional[Dict[str, Any]]:
        self.invalid_tool_count += 1
        if self.invalid_tool_count > self.invalid_tool_limit:
            return {"type": "invalid_tool_limit_exceeded", "detail": self.invalid_tool_count}
        return None

    def record_tool_call(self, name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.tool_calls_total += 1
        self.tool_calls_by_name[name] = self.tool_calls_by_name.get(name, 0) + 1
        call_sig = f"{name}:{json.dumps(args, sort_keys=True)}"
        self.tool_call_history.append(call_sig)

        if self.tool_calls_total > self.max_tool_calls:
            return {
                "type": "tool_call_budget_exceeded",
                "detail": {"budget": self.max_tool_calls, "used": self.tool_calls_total},
            }
        if self.tool_calls_by_name[name] > self.max_calls_per_tool:
            return {
                "type": "per_tool_budget_exceeded",
                "detail": {"tool": name, "budget": self.max_calls_per_tool},
            }
        if (
            len(self.tool_call_history) == self.tool_call_history.maxlen
            and len(set(self.tool_call_history)) == 1
        ):
            return {
                "type": "repetition_threshold_exceeded",
                "detail": {"tool": name, "threshold": self.repetition_threshold},
            }
        return None

    def check_wall_clock(self) -> Optional[Dict[str, Any]]:
        if self.wall_clock_timeout_sec is None:
            return None
        if (time.time() - self.start_time) > self.wall_clock_timeout_sec:
            return {"type": "wall_clock_timeout", "detail": self.wall_clock_timeout_sec}
        return None

    def check_goal_drift(self, goal: str, focus: str) -> Optional[Dict[str, Any]]:
        # Simple token overlap heuristic to detect drift without external deps.
        goal_tokens = set(normalize_tokens(goal))
        focus_tokens = set(normalize_tokens(focus))
        if not goal_tokens or not focus_tokens:
            return None
        overlap = len(goal_tokens & focus_tokens) / max(1, len(goal_tokens | focus_tokens))
        if overlap < self.drift_threshold:
            return {
                "type": "goal_drift_detected",
                "detail": {"overlap": overlap, "threshold": self.drift_threshold},
            }
        return None
