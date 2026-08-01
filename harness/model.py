"""Thin Messages API client with cost metering and logging."""

from __future__ import annotations

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


class ModelClient:
    def __init__(self, client: Any, meter: CostMeter, max_usd_budget: float | None):
        self.client = client if client is not None else anthropic.Anthropic()
        self.meter = meter
        self.max_usd_budget = max_usd_budget

    def create(self, *, purpose: str, logger: TrajectoryLogger, **kwargs: Any) -> ModelResponse:
        if self.max_usd_budget is not None and self.meter.totals.cost_usd >= self.max_usd_budget:
            raise BudgetExceeded(f"spent {self.meter.totals.cost_usd:.4f} of {self.max_usd_budget:.4f} USD")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not kwargs["model"].startswith("claude-haiku"):
            kwargs.setdefault("thinking", {"type": "disabled"})
        logger.request(purpose, kwargs["model"], kwargs.get("system"), kwargs["messages"], kwargs.get("tools"))
        started = time.monotonic()
        try:
            raw = self.client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            logger.event("api_error", purpose=purpose, status=exc.status_code)
            raise
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = self.meter.record(kwargs["model"], raw.usage)
        content = [block_to_param(b) for b in raw.content]
        logger.response(purpose, content, raw.usage, raw.stop_reason, latency_ms)
        return ModelResponse(content=content, stop_reason=raw.stop_reason or "end_turn", usage=usage)
