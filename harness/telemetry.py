"""Trajectory logging and cost accounting."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.config import CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, price_for

SNAPSHOT_EVERY = 20


def _usage_get(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return int(value or 0)


@dataclass
class UsageTotals:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def add_usage(self, model: str, usage: Any) -> "UsageTotals":
        inp = _usage_get(usage, "input_tokens")
        out = _usage_get(usage, "output_tokens")
        cread = _usage_get(usage, "cache_read_input_tokens")
        cwrite = _usage_get(usage, "cache_creation_input_tokens")
        price_in, price_out = price_for(model)
        cost = (
            inp * price_in
            + cwrite * price_in * CACHE_WRITE_MULTIPLIER
            + cread * price_in * CACHE_READ_MULTIPLIER
            + out * price_out
        ) / 1_000_000
        self.api_calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cread
        self.cache_write_tokens += cwrite
        self.cost_usd += cost
        return self

    def merge(self, other: "UsageTotals") -> None:
        self.api_calls += other.api_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cost_usd += other.cost_usd

    def cache_hit_rate(self) -> float:
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0

    def summary(self) -> dict:
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_hit_rate": round(self.cache_hit_rate(), 4),
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class CostMeter:
    """Process-wide spend, used to honour the USD budget."""

    totals: UsageTotals = field(default_factory=UsageTotals)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, model: str, usage: Any) -> UsageTotals:
        single = UsageTotals().add_usage(model, usage)
        with self.lock:
            self.totals.merge(single)
        return single


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class TrajectoryLogger:
    """Writes one JSONL file per session.

    Requests are logged as deltas against the previous request of the session;
    a full snapshot is written every ``SNAPSHOT_EVERY`` requests and whenever
    the previously sent prefix changed.
    """

    def __init__(self, log_dir: Path | None, session_id: str, clock):
        self.enabled = log_dir is not None
        self.clock = clock
        self._prev_messages: list[dict] = []
        self._prev_system_digest = ""
        self._count = 0
        self._lock = threading.Lock()
        if self.enabled:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.path = log_dir / f"{session_id}.jsonl"

    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        record = {"ts": self.clock().isoformat(), **record}
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def event(self, kind: str, **fields: Any) -> None:
        self._write({"type": kind, **fields})

    def request(self, purpose: str, model: str, system: Any, messages: list[dict], tools: Any) -> None:
        if not self.enabled:
            return
        self._count += 1
        system_digest = _digest(system)
        n_prev = len(self._prev_messages)
        prefix_same = n_prev <= len(messages) and _digest(messages[:n_prev]) == _digest(
            self._prev_messages
        )
        full = (
            purpose != "main"
            or self._count % SNAPSHOT_EVERY == 1
            or not prefix_same
            or system_digest != self._prev_system_digest
        )
        record: dict[str, Any] = {
            "type": "request",
            "purpose": purpose,
            "model": model,
            "system_digest": system_digest,
            "tools_digest": _digest(tools),
            "message_count": len(messages),
        }
        if full:
            record["snapshot"] = True
            record["system"] = system
            record["messages"] = messages
        else:
            record["snapshot"] = False
            record["messages_appended"] = messages[n_prev:]
        if purpose == "main":
            self._prev_messages = [dict(m) for m in messages]
            self._prev_system_digest = system_digest
        self._write(record)

    def response(self, purpose: str, content: list[dict], usage: Any, stop_reason: str, latency_ms: int) -> None:
        self._write(
            {
                "type": "response",
                "purpose": purpose,
                "stop_reason": stop_reason,
                "latency_ms": latency_ms,
                "usage": {
                    "input_tokens": _usage_get(usage, "input_tokens"),
                    "output_tokens": _usage_get(usage, "output_tokens"),
                    "cache_read_input_tokens": _usage_get(usage, "cache_read_input_tokens"),
                    "cache_creation_input_tokens": _usage_get(usage, "cache_creation_input_tokens"),
                },
                "content": content,
            }
        )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
