"""Public contract of the harness.

DO NOT MODIFY THIS FILE. Evaluation drives the harness exclusively through the
classes defined here. Everything else under ``harness/`` may be rewritten.

Contract rules
--------------
1. Persistence. ``state_dir`` is the only place the harness may persist data.
   The harness may be re-created in a new process against the same
   ``state_dir`` at any time, including after the previous process was killed
   without ``end_session`` or ``close`` being called.

2. Context budget. For every Messages API request the harness sends (main
   loop, sub-agents, and any auxiliary calls), the request's input tokens
   must satisfy

       input_tokens + cache_read_input_tokens + cache_creation_input_tokens
           <= config.max_context_tokens

   as reported in the response ``usage`` block. Output tokens do not count.

3. Message hygiene. Every request's ``messages`` array must be one the API
   accepts as-is: it starts with a user message, and every ``tool_use`` block
   is answered by exactly one ``tool_result`` in the immediately following
   user message.

4. Telemetry. Token and cache counts in ``AgentReply`` must be the sums of the
   ``usage`` blocks returned by the API for the calls made during the turn.
   ``input_tokens`` is the uncached input (``usage.input_tokens``);
   ``cache_read_tokens`` and ``cache_write_tokens`` are reported separately.
   The ``subagent_*`` fields cover the sub-agent's calls only, and
   ``subagent_input_tokens`` is its *total* prompt size (uncached + cache read
   + cache write). ``raw_messages`` must be the exact ``messages`` array of
   the final request of the turn. Evaluation records API traffic
   independently and compares.

5. Model access. If ``config.model_client`` is set, the harness must make all
   model calls through it. It exposes the same surface as
   ``anthropic.Anthropic``: ``.messages.create(**kwargs)`` and
   ``.messages.count_tokens(**kwargs)``. If it is ``None``, construct an
   ``anthropic.Anthropic`` client (which honours ``ANTHROPIC_BASE_URL``).

6. Time. If ``config.clock`` is set, the harness must read the current time
   only through it.

7. Memory snapshot. ``memory_snapshot`` returns one dict per stored item. Each
   dict must contain at least ``"text"`` (str, human-readable statement of the
   item) and ``"active"`` (bool, whether the item can currently be retrieved
   and presented as true). Other keys are free-form.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    ok: bool
    content: str  # text returned to the model
    error: str | None = None
    latency_ms: int = 0


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], ToolResult]


class AgentReply(BaseModel):
    text: str
    tool_calls_made: int
    api_calls_made: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    subagent_api_calls: int
    subagent_input_tokens: int
    subagent_output_tokens: int
    raw_messages: list[dict]  # the exact messages array sent on the final API call


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


class HarnessConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(default_factory=lambda: os.environ.get("HARNESS_MODEL", "claude-sonnet-5"))
    aux_model: str = Field(
        default_factory=lambda: os.environ.get("HARNESS_AUX_MODEL", "claude-haiku-4-5")
    )
    max_context_tokens: int = 24_000
    max_output_tokens: int = 2_048
    max_usd_budget: float | None = Field(default_factory=lambda: _env_float("MAX_USD_BUDGET"))
    log_dir: Path | None = None
    model_client: Any = None
    clock: Callable[[], datetime] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class Harness:
    def __init__(self, config: HarnessConfig, tools: list[ToolSpec], state_dir: Path):
        from harness.loop import Engine

        self._engine = Engine(config=config, tools=list(tools), state_dir=Path(state_dir))

    def new_session(self, user_id: str) -> str:
        return self._engine.new_session(user_id)

    def run_turn(self, session_id: str, user_message: str) -> AgentReply:
        return self._engine.run_turn(session_id, user_message)

    def end_session(self, session_id: str) -> None:
        self._engine.end_session(session_id)

    def memory_snapshot(self, user_id: str) -> list[dict]:
        return self._engine.memory_snapshot(user_id)

    def close(self) -> None:
        self._engine.close()
