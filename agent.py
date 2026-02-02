import json
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Union


def call_llm(prompt: str) -> Dict[str, Any]:
    """
    LLM stub. Replace with a real model call that returns a dict.
    The contract requires JSON with keys: thought, action{name,args}.
    """
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
        self.tools = ToolRegistry()
        self._register_tools()

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
        decision = call_llm(prompt)

        # Minimal validation to enforce one decision per loop.
        action = decision.get("action")
        if not isinstance(action, dict) or "name" not in action or "args" not in action:
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
            result = {
                "ok": False,
                "error": "invalid tool selection",
                "tool": tool_name,
            }
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
                result = {
                    "ok": False,
                    "error": f"missing args: {', '.join(missing)}",
                    "tool": tool_name,
                }
                decision = {
                    "thought": "invalid args",
                    "action": {
                        "name": "stop",
                        "args": {"reason": "LLM provided invalid tool args"},
                    },
                }
            else:
                result = tool["func"](**tool_args)
                # Only the agent may write to long-term memory.
                if tool_name == "propose_memory":
                    result = self.long_term.add(
                        summary=tool_args.get("summary", ""),
                        tags=tool_args.get("tags", []),
                    )

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

    def run(self, max_steps: int = 5) -> Dict[str, Any]:
        # Prevent infinite loops with a clear stop condition.
        while self.step < max_steps:
            observation = self.observe()
            decision = self.decide(observation)
            record = self.act(observation, decision)
            if decision["action"]["name"] == "stop":
                result = {
                    "status": "stopped",
                    "record": record,
                    "log": self.log,
                }
                self._post_run_reflection(final_status="stopped", final_record=record)
                return result
        result = {
            "status": "max_steps_reached",
            "record": self.log[-1] if self.log else None,
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
        data = call_llm(prompt)
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


if __name__ == "__main__":
    # Example run; outputs structured JSON only.
    agent = Agent(goal="demo", context={"note": "example"})
    print(json.dumps(agent.run(max_steps=3)))
