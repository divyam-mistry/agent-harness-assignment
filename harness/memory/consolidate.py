"""Episodic -> long-term consolidation (policy: see ``longterm``)."""

from __future__ import annotations

from datetime import datetime

from harness.memory.episodic import EpisodicMemory
from harness.memory.longterm import LongTermMemory


def consolidate(episodic: EpisodicMemory, longterm: LongTermMemory, now: datetime | None = None) -> int:
    """Merge this session's not-yet-flushed facts into long-term memory, then forget what has expired.

    Safe to call after every turn and again at session end: each fact is flushed once.
    Returns how many facts were flushed.
    """
    pending = [f for f in episodic.facts if not f.get("flushed")]
    for item in pending:
        longterm.add(episodic.user_id, item["text"], episodic.session_id, item["at"], key=item.get("key"))
        item["flushed"] = True
    if now is not None:
        longterm.sweep(now, episodic.user_id)
    longterm.save()
    return len(pending)
