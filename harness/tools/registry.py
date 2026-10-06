"""Tool registry and dispatch."""

from __future__ import annotations

import concurrent.futures as cf
import json
import time
from collections import Counter

from harness.api import ToolResult, ToolSpec

MAX_RESULT_CHARS = 6000
EMPTY_RESULT = "(the tool returned no data)"


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec], timeout_s: float = 20.0):
        self._specs = {spec.name: spec for spec in specs}
        self._pool = cf.ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")
        self.timeout_s = timeout_s

    def names(self) -> list[str]:
        return sorted(self._specs)

    def api_definitions(self, only: tuple[str, ...] | None = None) -> list[dict]:
        return [
            {"name": s.name, "description": s.description, "input_schema": s.input_schema}
            for s in self._specs.values()
            if only is None or s.name in only
        ]

    def call(self, name: str, args: dict) -> ToolResult:
        """Run one tool. Never raises: failures come back as ``ok=False`` so the model can see them."""
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ok=False, content="", error=f"unknown tool {name}")
        started = time.monotonic()
        try:
            result = spec.handler(args if isinstance(args, dict) else {})
        except Exception as exc:
            return ToolResult(ok=False, content="", error=f"{type(exc).__name__}: {exc}"[:300])
        if not isinstance(result, ToolResult):
            return ToolResult(ok=False, content="", error="tool returned an invalid result")
        updates: dict = {}
        if not result.latency_ms:
            updates["latency_ms"] = int((time.monotonic() - started) * 1000)
        if result.ok and not (result.content or "").strip():
            updates["content"] = EMPTY_RESULT
        elif len(result.content or "") > MAX_RESULT_CHARS:
            updates["content"] = result.content[:MAX_RESULT_CHARS] + "\n[truncated]"
        return result.model_copy(update=updates) if updates else result

    def call_many(self, uses: list[dict]) -> list[ToolResult]:
        """Run tool calls in parallel; a call that exceeds the timeout is reported as failed."""
        futures = [self._pool.submit(self.call, u["name"], u.get("input") or {}) for u in uses]
        cf.wait(futures, timeout=self.timeout_s)
        results = []
        for future in futures:
            if future.done():
                results.append(future.result())
            else:
                future.cancel()
                results.append(ToolResult(ok=False, content="", error=f"timed out after {self.timeout_s:g}s"))
        return results

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def result_block(tool_use_id: str, result: ToolResult) -> dict:
    if result.ok:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": result.content}
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": result.error or "error", "is_error": True}


class TurnToolRunner:
    """Per-turn dispatch guard against retry loops.

    Within one turn an identical call that already succeeded is answered from its first
    result, and one that already failed ``max_failures`` times is refused without being
    executed, so a flaky backend cannot burn the step budget.
    """

    def __init__(self, registry: ToolRegistry, max_failures: int = 2):
        self.registry = registry
        self.max_failures = max_failures
        self._done: dict[str, ToolResult] = {}
        self._failures: Counter[str] = Counter()

    @staticmethod
    def _key(use: dict) -> str:
        return f"{use['name']}:{json.dumps(use.get('input') or {}, sort_keys=True, default=str)}"

    def run(self, uses: list[dict]) -> list[ToolResult]:
        results: list[ToolResult | None] = [None] * len(uses)
        pending: dict[str, int] = {}  # key -> index of the call that will execute
        for i, use in enumerate(uses):
            key = self._key(use)
            if key in self._done:
                results[i] = self._done[key]
            elif self._failures[key] >= self.max_failures:
                results[i] = ToolResult(
                    ok=False,
                    content="",
                    error=f"this exact call has already failed {self._failures[key]} times this turn and was not "
                    "retried; do not call it again. Tell the user it is unavailable, or try different arguments.",
                )
            elif key not in pending:
                pending[key] = i
        executed = self.registry.call_many([uses[i] for i in pending.values()])
        for (key, _), result in zip(pending.items(), executed):
            if result.ok:
                self._done[key] = result
            else:
                self._failures[key] += 1
        for i, use in enumerate(uses):
            if results[i] is None:
                results[i] = executed[list(pending).index(self._key(use))]
        return results  # type: ignore[return-value]
