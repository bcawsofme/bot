import json
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List

from utils import safe_result_info, truncate_str


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
                "result_info": truncate_str(safe_result_info(result), max_chars),
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
