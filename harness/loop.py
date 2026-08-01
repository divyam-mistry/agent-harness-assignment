"""Turn loop, tool dispatch, and session lifecycle."""

from __future__ import annotations

import uuid
from pathlib import Path

from harness.api import AgentReply, HarnessConfig, ToolSpec
from harness.config import Settings
from harness.context import ContextBuilder, with_recalled_facts
from harness.memory import extract
from harness.memory.consolidate import consolidate
from harness.memory.longterm import LongTermMemory
from harness.memory.working import WorkingMemory
from harness.model import ModelClient
from harness.telemetry import CostMeter, TrajectoryLogger, UsageTotals, utcnow
from harness.tools.registry import ToolRegistry, result_block
from harness.tools.subagent import RESEARCH_TOOL, ResearchAgent


class Engine:
    def __init__(self, config: HarnessConfig, tools: list[ToolSpec], state_dir: Path):
        self.config = config
        self.settings = Settings.from_options(config.options)
        self.clock = config.clock or utcnow
        state_dir.mkdir(parents=True, exist_ok=True)
        self._warnings: list[str] = []
        self.meter = CostMeter()
        self.client = ModelClient(config.model_client, self.meter, config.max_usd_budget)
        self.registry = ToolRegistry(tools)
        self.memory = LongTermMemory(state_dir, on_warning=self._warnings.append)
        self.context = ContextBuilder(config.max_context_tokens)
        self.research = ResearchAgent(self.client, self.registry, self.settings, config.model, config.max_output_tokens)
        self.sessions: dict[str, WorkingMemory] = {}
        self.loggers: dict[str, TrajectoryLogger] = {}
        self.usage: dict[str, UsageTotals] = {}

    def new_session(self, user_id: str) -> str:
        session_id = f"{uuid.uuid4().hex[:12]}"
        self.sessions[session_id] = WorkingMemory(session_id=session_id, user_id=user_id)
        logger = TrajectoryLogger(self.config.log_dir, session_id, self.clock)
        for warning in self._warnings:
            logger.event("warning", message=warning)
        self._warnings.clear()
        logger.event("session_start", user_id=user_id, model=self.config.model)
        self.loggers[session_id] = logger
        self.usage[session_id] = UsageTotals()
        return session_id

    def end_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        flushed = consolidate(session.episodic, self.memory)
        logger = self.loggers.pop(session_id)
        logger.event("consolidation", facts_flushed=flushed)
        logger.event("cost_summary", scope="session", **self.usage.pop(session_id).summary())

    def memory_snapshot(self, user_id: str) -> list[dict]:
        return [{"text": f["text"], "active": True, **f} for f in self.memory.for_user(user_id)]

    def close(self) -> None:
        for session_id, logger in self.loggers.items():
            logger.event("cost_summary", scope="session", **self.usage[session_id].summary())
        self.registry.close()

    def _session_info(self, session: WorkingMemory) -> str:
        now = self.clock()
        return (
            f"Session information: signed-in user id {session.user_id}; "
            f"current time {now:%Y-%m-%d %H:%M}; turn {session.turn} of this session."
        )

    def run_turn(self, session_id: str, user_message: str) -> AgentReply:
        session = self.sessions[session_id]
        logger = self.loggers[session_id]
        session.turn += 1
        turn_usage, sub_usage = UsageTotals(), UsageTotals()

        memories = [f["text"] for f in self.memory.search(session.user_id, user_message, self.settings.retrieval_k)]
        session.history.append({"role": "user", "content": with_recalled_facts(user_message, memories)})
        tools = self.registry.api_definitions() + [RESEARCH_TOOL]
        tool_calls = 0
        reply_text = ""
        sent: list[dict] = []

        for _ in range(self.settings.max_steps):
            dropped = self.context.fit(session.history)
            if dropped:
                logger.event("truncation", dropped_messages=dropped)
            sent = list(session.history)
            response = self.client.create(
                purpose="main",
                logger=logger,
                model=self.config.model,
                system=self.context.system_blocks({"session": self._session_info(session)}, self.settings.system_layout),
                messages=sent,
                tools=tools,
                max_tokens=self.config.max_output_tokens,
            )
            turn_usage.merge(response.usage)
            session.history.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                reply_text = response.text
                break
            uses = response.tool_uses
            blocks = []
            plain = [u for u in uses if u["name"] != RESEARCH_TOOL["name"]]
            results = dict(zip([u["id"] for u in plain], self.registry.call_many(plain)))
            for use in uses:
                if use["name"] == RESEARCH_TOOL["name"]:
                    outcome = self.research.run(str(use.get("input", {}).get("task", "")), logger)
                    sub_usage.merge(outcome.usage)
                    blocks.append({"type": "tool_result", "tool_use_id": use["id"], "content": outcome.content})
                else:
                    result = results[use["id"]]
                    logger.event("tool_call", scope="main", name=use["name"], input=use.get("input"), ok=result.ok)
                    blocks.append(result_block(use["id"], result))
                tool_calls += 1
            session.history.append({"role": "user", "content": blocks})

        # Memory extraction runs after the reply is produced.
        self._remember(session, logger, user_message, reply_text, turn_usage)

        self.usage[session_id].merge(turn_usage)
        self.usage[session_id].merge(sub_usage)
        return AgentReply(
            text=reply_text,
            tool_calls_made=tool_calls,
            api_calls_made=turn_usage.api_calls,
            input_tokens=turn_usage.input_tokens,
            output_tokens=turn_usage.output_tokens,
            cache_read_tokens=turn_usage.cache_read_tokens,
            cache_write_tokens=turn_usage.cache_write_tokens,
            subagent_api_calls=sub_usage.api_calls,
            subagent_input_tokens=sub_usage.input_tokens + sub_usage.cache_read_tokens + sub_usage.cache_write_tokens,
            subagent_output_tokens=sub_usage.output_tokens,
            raw_messages=sent,
        )

    def _remember(self, session: WorkingMemory, logger: TrajectoryLogger, user_text: str, reply_text: str,
                  usage: UsageTotals) -> None:
        if not self.settings.extraction_enabled:
            return
        response = self.client.create(
            purpose="extract",
            logger=logger,
            model=self.config.aux_model,
            system=extract.EXTRACTION_INSTRUCTIONS,
            messages=extract.build_request(user_text, reply_text),
            max_tokens=512,
        )
        usage.merge(response.usage)
        for text in extract.parse_facts(response.text):
            session.episodic.add(text, session.turn, self.clock().isoformat())
