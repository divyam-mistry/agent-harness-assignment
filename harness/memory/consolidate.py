"""Session-end flush from episodic to long-term memory."""

from __future__ import annotations

from harness.memory.episodic import EpisodicMemory
from harness.memory.longterm import LongTermMemory


def consolidate(episodic: EpisodicMemory, longterm: LongTermMemory) -> int:
    """Move everything learned this session into long-term memory."""
    for item in episodic.facts:
        longterm.add(episodic.user_id, item["text"], episodic.session_id, item["at"])
    longterm.save()
    return len(episodic.facts)
