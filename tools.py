from typing import Any, Callable, Dict, Optional


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
