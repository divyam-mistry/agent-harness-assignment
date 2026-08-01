"""Facts collected during the current session."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodicMemory:
    session_id: str
    user_id: str
    facts: list[dict] = field(default_factory=list)

    def add(self, text: str, turn: int, at: str) -> None:
        self.facts.append({"text": text, "turn": turn, "at": at})
