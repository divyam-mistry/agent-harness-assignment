"""Working memory: the live state of one session."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.memory.episodic import EpisodicMemory


@dataclass
class WorkingMemory:
    session_id: str
    user_id: str
    history: list[dict] = field(default_factory=list)
    episodic: EpisodicMemory | None = None
    turn: int = 0

    def __post_init__(self) -> None:
        if self.episodic is None:
            self.episodic = EpisodicMemory(self.session_id, self.user_id)
