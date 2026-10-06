"""Turn loop, tool dispatch, and session lifecycle."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

from harness import context as ctxmod
from harness.api import AgentReply, HarnessConfig, ToolSpec
from harness.config import Settings
from harness.context import ContextBuilder, with_cache_breakpoint, with_recalled_facts
from harness.memory import extract
from harness.memory.consolidate import consolidate
from harness.memory.longterm import LongTermMemory
from harness.memory.working import WorkingMemory
from harness.model import ModelClient
from harness.telemetry import CostMeter, TrajectoryLogger, UsageTotals, utcnow
from harness.tools.registry import ToolRegistry, TurnToolRunner, result_block
from harness.tools.subagent import RESEARCH_TOOL, ResearchAgent

NO_REPLY = "I wasn't able to put an answer together just now. Please try again or rephrase your request."
CONVERSATION_DIGEST_CHARS = 3000


class Engine:
    def __init__(self, config: HarnessConfig, tools: list[ToolSpec], state_dir: Path):
        self.config = config
        self.settings = Settings.from_options(config.options)
        self.clock = config.clock or utcnow
        state_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = state_dir / "sessions"
        self._warnings: list[str] = []
        self.meter = CostMeter()
        self.client = ModelClient(config.model_client, self.meter, config.max_usd_budget,
                                  max_attempts=self.settings.api_max_attempts, base_delay=self.settings.api_base_delay_s)
        self.registry = ToolRegistry(tools, timeout_s=self.settings.tool_timeout_s)
        self.memory = LongTermMemory(state_dir, on_warning=self._warnings.append)
        self.context = ContextBuilder(config.max_context_tokens)
        self.research = ResearchAgent(self.client, self.registry, self.settings, config.model,
                                      config.max_output_tokens, self.context)
        self.sessions: dict[str, WorkingMemory] = {}
        self.loggers: dict[str, TrajectoryLogger] = {}
        self.usage: dict[str, UsageTotals] = {}

    # -- lifecycle -------------------------------------------------------------------------------

    def new_session(self, user_id: str) -> str:
        session_id = f"{uuid.uuid4().hex[:12]}"
        now = self.clock()
        session = WorkingMemory(session_id=session_id, user_id=user_id, started_on=f"{now:%A %Y-%m-%d}")
        self.sessions[session_id] = session
        if self.memory.sweep(now, user_id):
            self.memory.save()
        self._attach(session)
        self.loggers[session_id].event("session_start", user_id=user_id, model=self.config.model)
        session.save(self.sessions_dir)
        return session_id

    def _attach(self, session: WorkingMemory) -> None:
        logger = TrajectoryLogger(self.config.log_dir, session.session_id, self.clock)
        for warning in self._warnings:
            logger.event("warning", message=warning)
        self._warnings.clear()
        self.loggers[session.session_id] = logger
        self.usage.setdefault(session.session_id, UsageTotals())

    def _session(self, session_id: str) -> WorkingMemory:
        """The live session, or the one a previous process left on disk."""
        session = self.sessions.get(session_id)
        if session is None:
            session = WorkingMemory.load(self.sessions_dir, session_id)
            if session is None:
                raise KeyError(session_id)
            self.sessions[session_id] = session
            self._attach(session)
            self.loggers[session_id].event("session_resumed", user_id=session.user_id, turn=session.turn)
        return session

    def end_session(self, session_id: str) -> None:
        try:
            session = self._session(session_id)
        except KeyError:
            return
        self.sessions.pop(session_id, None)
        flushed = consolidate(session.episodic, self.memory, self.clock())
        WorkingMemory.delete(self.sessions_dir, session_id)
        logger = self.loggers.pop(session_id)
        logger.event("consolidation", facts_flushed=flushed)
        logger.event("cost_summary", scope="session", **self.usage.pop(session_id).summary())

    def memory_snapshot(self, user_id: str) -> list[dict]:
        return [{**f, "text": f["text"], "active": bool(f["active"])} for f in self.memory.for_user(user_id)]

    def close(self) -> None:
        for session_id, logger in self.loggers.items():
            logger.event("cost_summary", scope="session", **self.usage[session_id].summary())
        self.registry.close()

    # -- turn ------------------------------------------------------------------------------------

    def _session_info(self, session: WorkingMemory) -> str:
        # Nothing here may change from request to request, or the prompt cache is lost.
        return f"Session information: signed-in user id {session.user_id}; this session started on {session.started_on}."

    def _messages(self, session: WorkingMemory) -> list[dict]:
        if not session.summary:
            return list(session.history)
        note = "[Summary of the earlier part of this conversation, which is no longer shown]\n" + session.summary
        return [{"role": "user", "content": note}, *session.history]

    def _conversation_digest(self, session: WorkingMemory) -> str:
        """What the user said so far, for the research sub-agent (newest statements win the space)."""
        lines = [f"- {ctxmod.message_text(m).split(ctxmod.RECALL_HEADER)[0].strip()[:500]}"
                 for m in session.history if ctxmod.is_turn_start(m)]
        digest = "\n".join(lines)[-CONVERSATION_DIGEST_CHARS:]
        if session.summary:
            digest = "Earlier in the conversation: " + session.summary + "\n" + digest
        return digest

    def run_turn(self, session_id: str, user_message: str) -> AgentReply:
        session = self._session(session_id)
        logger = self.loggers[session_id]
        turn_usage, sub_usage = UsageTotals(), UsageTotals()
        session.turn += 1

        user_message = ctxmod.clamp_text(user_message, int(self.config.max_context_tokens * 0.3))
        memories = self._recall(session, user_message)
        entry = {"role": "user", "content": with_recalled_facts(user_message, memories)}
        session.history.append(entry)
        try:
            reply_text, sent, tool_calls = self._converse(session, logger, turn_usage, sub_usage)
        except BaseException:
            # Leave the session valid: remove this turn's partial exchange, keep everything before it.
            for i, m in enumerate(session.history):
                if m is entry:
                    del session.history[i:]
                    break
            session.turn -= 1
            raise

        self._remember(session, logger, user_message, reply_text, turn_usage)
        session.save(self.sessions_dir)
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
            raw_messages=copy.deepcopy(sent),
        )

    def _recall(self, session: WorkingMemory, user_message: str) -> list[str]:
        """Relevant long-term facts not already in this conversation's history."""
        found = self.memory.search(session.user_id, user_message, self.settings.retrieval_k)
        fresh = [f for f in found if f["id"] not in session.shown_facts]
        session.shown_facts.update(f["id"] for f in fresh)
        if fresh:
            self.memory.mark_used([f["id"] for f in fresh], self.clock().isoformat())
        return [f["text"] for f in fresh]

    def _converse(self, session: WorkingMemory, logger: TrajectoryLogger, turn_usage: UsageTotals,
                  sub_usage: UsageTotals) -> tuple[str, list[dict], int]:
        tools = self.registry.api_definitions() + [RESEARCH_TOOL]
        system = self.context.system_blocks({"session": self._session_info(session)}, self.settings.system_layout)
        overhead = self.context.tokens([system, tools]) + ctxmod.SUMMARY_MAX_TOKENS
        runner = TurnToolRunner(self.registry)
        tool_calls = 0
        reply_text = ""
        sent: list[dict] = []

        for step in range(self.settings.max_steps + 1):
            final = step == self.settings.max_steps  # out of steps: make the model answer with what it has
            dropped = self.context.split(session.history, overhead)
            if dropped:
                self._fold_into_summary(session, dropped, logger, turn_usage)
            sent = with_cache_breakpoint(self._messages(session))
            estimate = overhead - ctxmod.SUMMARY_MAX_TOKENS + self.context.tokens(sent)
            extra = {"tool_choice": {"type": "none"}} if final else {}
            response = self.client.create(
                purpose="main",
                logger=logger,
                model=self.config.model,
                system=system,
                messages=sent,
                tools=tools,
                max_tokens=self.config.max_output_tokens,
                **extra,
            )
            turn_usage.merge(response.usage)
            self.context.observe(estimate, response.usage.input_tokens + response.usage.cache_read_tokens
                                 + response.usage.cache_write_tokens)
            content = response.content
            uses = [] if final else response.tool_uses
            if final:
                content = [b for b in content if b.get("type") == "text"]
            session.history.append({"role": "assistant", "content": content or [{"type": "text", "text": "(no reply)"}]})
            if not uses:
                reply_text = response.text.strip() or NO_REPLY
                break
            blocks = []
            plain = [u for u in uses if u["name"] != RESEARCH_TOOL["name"]]
            results = dict(zip([u["id"] for u in plain], runner.run(plain)))
            for use in uses:
                if use["name"] == RESEARCH_TOOL["name"]:
                    outcome = self.research.run(str(use.get("input", {}).get("task", "")), logger,
                                                self._conversation_digest(session))
                    sub_usage.merge(outcome.usage)
                    blocks.append({"type": "tool_result", "tool_use_id": use["id"], "content": outcome.content})
                else:
                    result = results[use["id"]]
                    logger.event("tool_call", scope="main", name=use["name"], input=use.get("input"), ok=result.ok,
                                 error=result.error)
                    blocks.append(result_block(use["id"], result))
                tool_calls += 1
            session.history.append({"role": "user", "content": blocks})
        return reply_text or NO_REPLY, sent, tool_calls

    def _fold_into_summary(self, session: WorkingMemory, dropped: list[dict], logger: TrajectoryLogger,
                           usage: UsageTotals) -> None:
        """Fold turns that no longer fit into the running summary so their content is not lost."""
        limit = ctxmod.SUMMARY_MAX_TOKENS * 3
        # Each summariser request must itself fit the budget: feed the dropped turns in chunks.
        room = max(1500, min(24_000, int(self.config.max_context_tokens * 3 * 0.3)))
        for chunk in ctxmod.chunk_lines(ctxmod.transcript_lines(dropped), room):
            try:
                response = self.client.create(
                    purpose="summary",
                    logger=logger,
                    model=self.config.aux_model,
                    system=ctxmod.SUMMARY_INSTRUCTIONS,
                    messages=[{"role": "user", "content": f"PREVIOUS SUMMARY:\n{session.summary or '(none)'}\n\nNEW MESSAGES:\n{chunk}"}],
                    max_tokens=ctxmod.SUMMARY_MAX_TOKENS,
                )
                usage.merge(response.usage)
                summary = response.text.strip()
                if not summary:
                    raise ValueError("empty summary")
            except Exception as exc:
                logger.event("summary_failed", error=f"{type(exc).__name__}: {exc}"[:200])
                kept = "\n".join(f"- {line[:200]}" for line in chunk.splitlines() if line.startswith("USER:"))
                summary = f"{session.summary}\n{kept}".strip()
            session.summary = summary[-limit:]
        session.shown_facts.clear()  # recalled facts that were in dropped turns may be shown again
        logger.event("truncation", dropped_messages=len(dropped), summary_chars=len(session.summary))

    def _remember(self, session: WorkingMemory, logger: TrajectoryLogger, user_text: str, reply_text: str,
                  usage: UsageTotals) -> None:
        """Extract facts from the exchange and write them straight through to long-term memory."""
        if not self.settings.extraction_enabled:
            return
        try:
            response = self.client.create(
                purpose="extract",
                logger=logger,
                model=self.config.aux_model,
                system=extract.EXTRACTION_INSTRUCTIONS,
                messages=extract.build_request(user_text, reply_text),
                max_tokens=512,
            )
            usage.merge(response.usage)
            now = self.clock().isoformat()
            for key, text in extract.parse_keyed(response.text):
                if extract.is_grounded(key, text, user_text):
                    session.episodic.add(text, session.turn, now, key)
            consolidate(session.episodic, self.memory, self.clock())
        except Exception as exc:  # losing one turn's facts must not fail the turn
            logger.event("memory_error", error=f"{type(exc).__name__}: {exc}"[:300])
