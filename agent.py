import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Union


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


class ToolRegistry:
    """Explicit tool registry with schema metadata for safe invocation."""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        args_schema: Dict[str, str],
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "func": func,
            "args_schema": args_schema,
        }

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._tools.get(name)

    def list(self) -> Dict[str, Dict[str, Any]]:
        # Return a JSON-safe view (exclude callables).
        safe: Dict[str, Dict[str, Any]] = {}
        for name, tool in self._tools.items():
            safe[name] = {
                "name": tool["name"],
                "description": tool["description"],
                "args_schema": tool["args_schema"],
            }
        return safe


class ShortTermMemory:
    """Fixed-size memory of recent steps for context summarization."""

    def __init__(self, max_entries: int = 5) -> None:
        self.max_entries = max_entries
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=max_entries)

    def add(self, observation: Dict[str, Any], action: Dict[str, Any], result: Dict[str, Any]) -> None:
        self._entries.append(
            {
                "observation": observation,
                "action": action,
                "result": result,
            }
        )

    def summarize(self, max_chars: int = 200) -> List[Dict[str, Any]]:
        # Provide a compact view to avoid prompting with raw logs.
        summary: List[Dict[str, Any]] = []
        for entry in list(self._entries):
            action = entry.get("action", {})
            result = entry.get("result", {})
            item = {
                "action": action.get("name"),
                "result_ok": result.get("ok"),
                "result_info": _truncate_str(_safe_result_info(result), max_chars),
            }
            summary.append(item)
        return summary


class LongTermMemory:
    """File-backed summaries only; raw logs are never persisted."""

    def __init__(
        self,
        path: str,
        max_entries: int = 100,
        max_summary_chars: int = 500,
    ) -> None:
        self.path = path
        self.max_entries = max_entries
        self.max_summary_chars = max_summary_chars
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._entries = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._entries = data
            else:
                self._entries = []
        except Exception:
            self._entries = []

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f)
        except Exception:
            # Failing to save should not crash the agent loop.
            pass

    def add(self, summary: str, tags: List[str]) -> Dict[str, Any]:
        # Enforce summary limits and prevent duplicates.
        if not summary or len(summary) > self.max_summary_chars:
            return {"ok": False, "error": "summary too long or empty"}
        if not all(isinstance(t, str) and t for t in tags):
            return {"ok": False, "error": "invalid tags"}
        if any(e.get("summary") == summary and e.get("tags") == tags for e in self._entries):
            return {"ok": False, "error": "duplicate entry"}

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "tags": tags,
        }
        self._entries.append(entry)
        # Enforce max size by evicting oldest entries.
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        self._save()
        return {"ok": True, "entry": entry}

    def recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self._entries[-limit:] if self._entries else []


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
        goal_tokens = set(_normalize_tokens(goal))
        focus_tokens = set(_normalize_tokens(focus))
        if not goal_tokens or not focus_tokens:
            return None
        overlap = len(goal_tokens & focus_tokens) / max(1, len(goal_tokens | focus_tokens))
        if overlap < self.drift_threshold:
            return {
                "type": "goal_drift_detected",
                "detail": {"overlap": overlap, "threshold": self.drift_threshold},
            }
        return None


def _truncate_str(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _safe_result_info(result: Dict[str, Any]) -> str:
    # Avoid large payloads while still providing useful signal.
    if "error" in result:
        return str(result["error"])
    if "path" in result:
        return f"path={result['path']}"
    if "reason" in result:
        return str(result["reason"])
    if "bytes" in result:
        return f"bytes={result['bytes']}"
    return "ok"


def _normalize_tokens(text: str) -> List[str]:
    # Lightweight tokenization for goal drift heuristics.
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [t for t in cleaned.split() if t]


class Agent:
    """Minimal observe → decide → act agent loop."""

    def __init__(
        self,
        goal: str,
        context: Union[str, Dict[str, Any]],
        short_term_max: int = 5,
        long_term_path: str = "long_term_memory.json",
        long_term_max: int = 100,
        long_term_summary_max: int = 500,
        long_term_recent_limit: int = 5,
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
        # Store inputs to anchor decisions and maintain state.
        self.goal = goal
        self.context = context
        self.step = 0
        self.log: list[Dict[str, Any]] = []
        self.last_result: Optional[Dict[str, Any]] = None
        self.short_term = ShortTermMemory(max_entries=short_term_max)
        self.long_term = LongTermMemory(
            path=long_term_path,
            max_entries=long_term_max,
            max_summary_chars=long_term_summary_max,
        )
        self.long_term_recent_limit = long_term_recent_limit
        self.guardrails = GuardrailManager(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_calls_per_tool=max_calls_per_tool,
            max_token_budget=max_token_budget,
            repetition_threshold=repetition_threshold,
            invalid_json_limit=invalid_json_limit,
            invalid_tool_limit=invalid_tool_limit,
            wall_clock_timeout_sec=wall_clock_timeout_sec,
            drift_check_interval=drift_check_interval,
            drift_threshold=drift_threshold,
        )
        self.tools = ToolRegistry()
        self._register_tools()
        self.max_steps = max_steps

    def _register_tools(self) -> None:
        # Tools are explicit functions with schemas to keep usage controlled.
        self.tools.register(
            name="write_file",
            description="Write content to a file path.",
            func=self._tool_write_file,
            args_schema={"path": "str", "content": "str"},
        )
        self.tools.register(
            name="read_file",
            description="Read content from a file path.",
            func=self._tool_read_file,
            args_schema={"path": "str"},
        )
        self.tools.register(
            name="stop",
            description="Stop the agent loop with a reason.",
            func=self._tool_stop,
            args_schema={"reason": "str"},
        )
        # This tool requests a memory proposal; the agent controls persistence.
        self.tools.register(
            name="propose_memory",
            description="Propose a long-term memory summary and tags.",
            func=self._tool_propose_memory,
            args_schema={"summary": "str", "tags": "list[str]"},
        )

    def _tool_write_file(self, path: str, content: str) -> Dict[str, Any]:
        # Side-effect is limited to file writes as requested.
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "path": path}

    def _tool_read_file(self, path: str) -> Dict[str, Any]:
        # Side-effect is limited to file reads as requested.
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"ok": True, "path": path, "content": content}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "path": path}

    def _tool_stop(self, reason: str) -> Dict[str, Any]:
        # Explicit stop tool keeps control flow obvious.
        return {"ok": True, "reason": reason}

    def _tool_propose_memory(self, summary: str, tags: List[str]) -> Dict[str, Any]:
        # The LLM never writes memory directly; the agent validates and persists.
        return {"ok": True, "summary": summary, "tags": tags}

    def observe(self) -> Dict[str, Any]:
        # Capture the current state for the decision step.
        return {
            "goal": self.goal,
            "context": self.context,
            "step": self.step,
            "last_result": self.last_result,
            "short_term_summary": self.short_term.summarize(),
            "long_term_summaries": self.long_term.recent(self.long_term_recent_limit),
        }

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # The LLM is treated as a pure dependency that returns structured output.
        prompt = (
            "Return JSON with fields: thought, action{name,args}. "
            f"Tools: {json.dumps(self.tools.list())} "
            f"Observation: {json.dumps(observation)}"
        )
        budget_stop = self.guardrails.record_llm_call(prompt)
        if budget_stop:
            return self._forced_stop_decision("llm_budget", budget_stop)
        decision = call_llm(prompt)
        if not isinstance(decision, dict):
            invalid_stop = self.guardrails.record_invalid_json()
            if invalid_stop:
                return self._forced_stop_decision("invalid_json", invalid_stop)
            return {
                "thought": "invalid response",
                "action": {
                    "name": "stop",
                    "args": {"reason": "LLM returned non-dict response"},
                },
            }

        # Minimal validation to enforce one decision per loop.
        action = decision.get("action")
        if not isinstance(action, dict) or "name" not in action or "args" not in action:
            invalid_stop = self.guardrails.record_invalid_json()
            if invalid_stop:
                return self._forced_stop_decision("invalid_json", invalid_stop)
            return {
                "thought": "invalid action",
                "action": {
                    "name": "stop",
                    "args": {"reason": "LLM returned unsupported action shape"},
                },
            }
        return decision

    def act(self, observation: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        # Record the decision and advance state; keep side effects explicit.
        action = decision["action"]
        tool_name = action.get("name")
        tool_args = action.get("args")
        tool = self.tools.get(tool_name)

        if tool is None or not isinstance(tool_args, dict):
            # Fail gracefully if the tool is invalid or args are malformed.
            invalid_stop = self.guardrails.record_invalid_tool()
            if invalid_stop:
                decision = self._forced_stop_decision("invalid_tool", invalid_stop)
            result = {
                "ok": False,
                "error": "invalid tool selection",
                "tool": tool_name,
            }
            if not invalid_stop:
                decision = {
                    "thought": "invalid tool",
                    "action": {
                        "name": "stop",
                        "args": {"reason": "LLM selected invalid tool"},
                    },
                }
        else:
            # Validate required args exist; no extra coercion.
            missing = [k for k in tool["args_schema"].keys() if k not in tool_args]
            if missing:
                invalid_stop = self.guardrails.record_invalid_tool()
                if invalid_stop:
                    decision = self._forced_stop_decision("invalid_tool", invalid_stop)
                result = {
                    "ok": False,
                    "error": f"missing args: {', '.join(missing)}",
                    "tool": tool_name,
                }
                if not invalid_stop:
                    decision = {
                        "thought": "invalid args",
                        "action": {
                            "name": "stop",
                            "args": {"reason": "LLM provided invalid tool args"},
                        },
                    }
            else:
                if tool_name == "stop" and not isinstance(tool_args.get("reason"), dict):
                    # Ensure all stops carry structured reasons.
                    tool_args = dict(tool_args)
                    tool_args["reason"] = {"type": "llm_stop", "detail": tool_args.get("reason")}
                    action["args"] = tool_args
                # Guardrails run before tool execution.
                tool_stop = self.guardrails.record_tool_call(tool_name, tool_args)
                if tool_stop:
                    decision = self._forced_stop_decision("guardrail", tool_stop)
                    result = {"ok": False, "error": "guardrail_stop", "detail": tool_stop}
                else:
                    result = tool["func"](**tool_args)
                    # Only the agent may write to long-term memory.
                    if tool_name == "propose_memory":
                        result = self.long_term.add(
                            summary=tool_args.get("summary", ""),
                            tags=tool_args.get("tags", []),
                        )

        action = decision["action"]
        if action.get("name") == "stop" and not isinstance(action.get("args", {}).get("reason"), dict):
            # Ensure all stop reasons are structured, even for validation failures.
            action = dict(action)
            args = dict(action.get("args", {}))
            args["reason"] = {"type": "stop", "detail": args.get("reason")}
            action["args"] = args
            decision = dict(decision)
            decision["action"] = action
        self.step += 1
        record = {
            "step": self.step,
            "decision": decision,
            "tool_result": result,
        }
        self.log.append(record)
        self.last_result = result
        self.short_term.add(observation=observation, action=action, result=result)
        return record

    def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        # Prevent infinite loops with a clear stop condition.
        max_steps = self.max_steps if max_steps is None else max_steps
        while self.step < max_steps:
            wall_stop = self.guardrails.check_wall_clock()
            if wall_stop:
                decision = self._forced_stop_decision("wall_clock", wall_stop)
                record = self.act(self.observe(), decision)
                result = {"status": "stopped", "record": record, "log": self.log}
                self._post_run_reflection(final_status="stopped", final_record=record)
                return result
            observation = self.observe()
            if self.step % self.guardrails.drift_check_interval == 0 and self.step > 0:
                focus = json.dumps(observation)
                drift_stop = self.guardrails.check_goal_drift(self.goal, focus)
                if drift_stop:
                    decision = self._forced_stop_decision("goal_drift", drift_stop)
                    record = self.act(self.observe(), decision)
                    result = {"status": "stopped", "record": record, "log": self.log}
                    self._post_run_reflection(final_status="stopped", final_record=record)
                    return result
            decision = self.decide(observation)
            record = self.act(observation, decision)
            decision = record["decision"]
            if decision["action"]["name"] == "stop":
                result = {
                    "status": "stopped",
                    "record": record,
                    "log": self.log,
                }
                self._post_run_reflection(final_status="stopped", final_record=record)
                return result
        stop_reason = {"type": "max_steps_reached", "detail": {"max_steps": max_steps}}
        result = {
            "status": "max_steps_reached",
            "record": {"stop_reason": stop_reason},
            "log": self.log,
        }
        self._post_run_reflection(final_status="max_steps_reached", final_record=result["record"])
        return result

    def _post_run_reflection(self, final_status: str, final_record: Optional[Dict[str, Any]]) -> None:
        # Reflection is separated from action to prevent tool execution during evaluation.
        reflection = self.reflect(
            goal=self.goal,
            short_term_summary=self.short_term.summarize(),
            final_status=final_status,
            final_record=final_record,
        )
        if reflection is None:
            return
        summary = self._reflection_to_summary(reflection)
        self.long_term.add(summary=summary, tags=["reflection", "learning"])

    def reflect(
        self,
        goal: str,
        short_term_summary: List[Dict[str, Any]],
        final_status: str,
        final_record: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # Reflection runs after the loop and cannot trigger tools.
        prompt = (
            "Return JSON with fields: outcome (success|partial|failure), "
            "what_worked (list), what_failed (list), next_time_try (list), "
            "confidence (0.0-1.0). "
            f"Goal: {goal}. "
            f"ShortTerm: {json.dumps(short_term_summary)}. "
            f"FinalStatus: {final_status}. "
            f"FinalRecord: {json.dumps(final_record)}."
        )
        budget_stop = self.guardrails.record_llm_call(prompt)
        if budget_stop:
            return None
        data = call_llm(prompt)
        if not isinstance(data, dict):
            return None
        validated = self._validate_reflection(data)
        return validated

    def _validate_reflection(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        required = {
            "outcome": str,
            "what_worked": list,
            "what_failed": list,
            "next_time_try": list,
            "confidence": (int, float),
        }
        for key, typ in required.items():
            if key not in data or not isinstance(data[key], typ):
                return None
        if data["outcome"] not in {"success", "partial", "failure"}:
            return None

        def _clean_list(values: List[Any]) -> List[str]:
            return [v for v in values if isinstance(v, str) and v.strip()]

        cleaned = {
            "outcome": data["outcome"],
            "what_worked": _clean_list(data["what_worked"]),
            "what_failed": _clean_list(data["what_failed"]),
            "next_time_try": _clean_list(data["next_time_try"]),
            "confidence": float(data["confidence"]),
        }
        # Clamp confidence to [0, 1].
        if cleaned["confidence"] < 0.0:
            cleaned["confidence"] = 0.0
        if cleaned["confidence"] > 1.0:
            cleaned["confidence"] = 1.0
        return cleaned

    def _reflection_to_summary(self, reflection: Dict[str, Any]) -> str:
        # Keep summaries concise for long-term storage.
        parts = [
            f"outcome={reflection['outcome']}",
            f"confidence={reflection['confidence']:.2f}",
        ]
        if reflection["what_worked"]:
            parts.append("worked=" + "; ".join(reflection["what_worked"]))
        if reflection["what_failed"]:
            parts.append("failed=" + "; ".join(reflection["what_failed"]))
        if reflection["next_time_try"]:
            parts.append("next=" + "; ".join(reflection["next_time_try"]))
        summary = " | ".join(parts)
        return _truncate_str(summary, self.long_term.max_summary_chars)

    def _forced_stop_decision(self, reason_type: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        # Structured stop reasons are required for guardrail-triggered exits.
        return {
            "thought": "forced stop",
            "action": {
                "name": "stop",
                "args": {"reason": {"type": reason_type, "detail": detail}},
            },
        }


if __name__ == "__main__":
    # Example run; outputs structured JSON only.
    agent = Agent(goal="demo", context={"note": "example"})
    print(json.dumps(agent.run(max_steps=3)))
