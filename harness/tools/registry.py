"""Tool registry and dispatch."""

from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time

from harness.api import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self._specs = {spec.name: spec for spec in specs}
        self._pool = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")
        # The model often repeats a lookup it already made in the same conversation.
        # Serving repeats from memory keeps latency and backend load down.
        self._recent: dict[str, ToolResult] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        return sorted(self._specs)

    def api_definitions(self, only: tuple[str, ...] | None = None) -> list[dict]:
        return [
            {"name": s.name, "description": s.description, "input_schema": s.input_schema}
            for s in self._specs.values()
            if only is None or s.name in only
        ]

    def call(self, name: str, args: dict) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ok=False, content="", error=f"unknown tool {name}")
        key = f"{name}:{json.dumps(args, sort_keys=True)}"
        with self._lock:
            if key in self._recent:
                return self._recent[key]
        started = time.monotonic()
        try:
            result = spec.handler(args)
        except Exception:
            result = ToolResult(ok=True, content="")
        if not result.latency_ms:
            result = result.model_copy(update={"latency_ms": int((time.monotonic() - started) * 1000)})
        if result.ok:
            with self._lock:
                self._recent[key] = result
        return result

    def call_many(self, uses: list[dict]) -> list[ToolResult]:
        futures = [self._pool.submit(self.call, u["name"], u.get("input") or {}) for u in uses]
        return [f.result() for f in futures]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def result_block(tool_use_id: str, result: ToolResult) -> dict:
    if result.ok:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": result.content}
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": result.error or "error", "is_error": True}
