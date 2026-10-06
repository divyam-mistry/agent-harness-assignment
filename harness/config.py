"""Internal settings. The evaluated contract lives in api.py."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

# USD per million tokens: (input, output). Cache writes bill at 1.25x input
# (5-minute TTL), cache reads at 0.1x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def price_for(model: str) -> tuple[float, float]:
    return PRICES.get(model, PRICES["claude-sonnet-5"])


@dataclass
class Settings:
    max_steps: int = 15
    # Static text first: anything that changes between requests must come after the cached prefix.
    system_layout: tuple[str, ...] = ("instructions", "session")
    retrieval_k: int = 5
    tool_timeout_s: float = 20.0
    api_max_attempts: int = 6
    api_base_delay_s: float = 0.5
    extraction_enabled: bool = True
    subagent_max_steps: int = 8
    subagent_tools: tuple[str, ...] = ("search_docs", "fetch_url", "lookup_order", "calendar_free_slots", "calculate")
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in options.items() if k in known}
        extra = {k: v for k, v in options.items() if k not in known}
        return cls(**values, extra=extra)
