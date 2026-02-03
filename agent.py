import json
from typing import Any, Dict, List, Optional, Union

from guardrails import GuardrailManager
from llm import call_llm
from memory import LongTermMemory, ShortTermMemory
from tools import ToolRegistry
from utils import truncate_str


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

    # ---- Tools ----

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

    # ---- Core Loop ----

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

    # ---- Reflection ----

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
        return truncate_str(summary, self.long_term.max_summary_chars)

    # ---- Stop Helpers ----

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
