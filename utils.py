from typing import Any, Dict, List


def truncate_str(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def safe_result_info(result: Dict[str, Any]) -> str:
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


def normalize_tokens(text: str) -> List[str]:
    # Lightweight tokenization for goal drift heuristics.
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [t for t in cleaned.split() if t]
