"""Thin Messages API client with cost metering and logging."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from harness.telemetry import CostMeter, TrajectoryLogger, UsageTotals


class BudgetExceeded(RuntimeError):
    pass


def block_to_param(block: Any) -> dict:
    data = block if isinstance(block, dict) else block.to_dict()
    kind = data.get("type")
    if kind == "text":
        return {"type": "text", "text": data.get("text", "")}
    if kind == "tool_use":
        return {"type": "tool_use", "id": data["id"], "name": data["name"], "input": data.get("input") or {}}
    return {k: v for k, v in data.items() if v is not None}


@dataclass
class ModelResponse:
    content: list[dict]
    stop_reason: str
    usage: UsageTotals

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class ModelClient:
    def __init__(self, client: Any, meter: CostMeter, max_usd_budget: float | None,
                 max_attempts: int = 6, base_delay: float = 0.5, max_delay: float = 8.0):
        self.client = client if client is not None else anthropic.Anthropic(timeout=90.0)
        self.meter = meter
        self.max_usd_budget = max_usd_budget
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def _delay(self, attempt: int, exc: Exception) -> float:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        try:
            hinted = float(headers.get("retry-after")) if headers else 0.0
        except (TypeError, ValueError):
            hinted = 0.0
        backoff = min(self.max_delay, self.base_delay * 2 ** attempt) * random.uniform(0.5, 1.0)
        return min(self.max_delay, max(hinted, backoff))

    def _create_with_retry(self, purpose: str, logger: TrajectoryLogger, kwargs: dict) -> Any:
        for attempt in range(self.max_attempts):
            try:
                return self.client.messages.create(**kwargs)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                status = getattr(exc, "status_code", None)
                logger.event("api_error", purpose=purpose, status=status, attempt=attempt + 1)
                retryable = status is None or status in RETRYABLE_STATUS
                if not retryable or attempt == self.max_attempts - 1:
                    raise
                time.sleep(self._delay(attempt, exc))

    def create(self, *, purpose: str, logger: TrajectoryLogger, **kwargs: Any) -> ModelResponse:
        if self.max_usd_budget is not None and self.meter.totals.cost_usd >= self.max_usd_budget:
            raise BudgetExceeded(f"spent {self.meter.totals.cost_usd:.4f} of {self.max_usd_budget:.4f} USD")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not kwargs["model"].startswith("claude-haiku"):
            kwargs.setdefault("thinking", {"type": "disabled"})
        logger.request(purpose, kwargs["model"], kwargs.get("system"), kwargs["messages"], kwargs.get("tools"))
        started = time.monotonic()
        raw = self._create_with_retry(purpose, logger, kwargs)
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = self.meter.record(kwargs["model"], raw.usage)
        content = [block_to_param(b) for b in raw.content]
        logger.response(purpose, content, raw.usage, raw.stop_reason, latency_ms)
        return ModelResponse(content=content, stop_reason=raw.stop_reason or "end_turn", usage=usage)
