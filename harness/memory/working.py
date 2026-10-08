"""Working memory: the live state of one session."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from harness.memory.episodic import EpisodicMemory


@dataclass
class WorkingMemory:
    session_id: str
    user_id: str
    history: list[dict] = field(default_factory=list)
    episodic: EpisodicMemory | None = None
    turn: int = 0
    summary: str = ""  # running summary of turns that were dropped from ``history``
    started_on: str = ""  # date shown to the model; fixed per session so the cached prefix stays stable
    shown_facts: set[str] = field(default_factory=set)  # long-term fact ids already put in front of the model

    def __post_init__(self) -> None:
        if self.episodic is None:
            self.episodic = EpisodicMemory(self.session_id, self.user_id)

    def save(self, directory: Path) -> None:
        """Persist after each turn so a killed process can resume the session."""
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id, "user_id": self.user_id, "history": self.history, "turn": self.turn,
            "summary": self.summary, "started_on": self.started_on, "shown_facts": sorted(self.shown_facts),
            "episodic": self.episodic.facts,
        }
        tmp = directory / f"{self.session_id}.json.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, directory / f"{self.session_id}.json")

    @classmethod
    def load(cls, directory: Path, session_id: str) -> "WorkingMemory | None":
        path = directory / f"{session_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        memory = cls(session_id=session_id, user_id=data["user_id"], history=data["history"], turn=data["turn"],
                     summary=data.get("summary", ""), started_on=data.get("started_on", ""),
                     shown_facts=set(data.get("shown_facts", [])))
        memory.episodic.facts = data.get("episodic", [])
        return memory

    @staticmethod
    def delete(directory: Path, session_id: str) -> None:
        (directory / f"{session_id}.json").unlink(missing_ok=True)
