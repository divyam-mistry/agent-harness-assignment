"""Persistent long-term memory with lexical retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LongTermMemory:
    """Facts the user has shared, stored per user in ``state_dir/memory.json``."""

    def __init__(self, state_dir: Path, on_warning: Callable[[str], None] | None = None):
        self.path = state_dir / "memory.json"
        self.on_warning = on_warning or (lambda message: None)
        self.facts: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError as exc:
            self.on_warning(f"could not parse {self.path.name} ({exc}); starting with empty memory")
            return []

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.facts, fh, indent=1)

    def add(self, user_id: str, text: str, session_id: str, created_at: str) -> None:
        self.facts.append({"user_id": user_id, "text": text, "session_id": session_id, "created_at": created_at})

    def for_user(self, user_id: str) -> list[dict]:
        return [f for f in self.facts if f["user_id"] == user_id]

    def search(self, user_id: str, query: str, k: int) -> list[dict]:
        facts = self.for_user(user_id)
        query_tokens = tokenize(query)
        if not facts or not query_tokens:
            return []
        index = BM25Okapi([tokenize(f["text"]) or ["_"] for f in facts])
        scores = index.get_scores(query_tokens)
        ranked = sorted(zip(scores, range(len(facts))), key=lambda pair: -pair[0])
        return [facts[i] for _, i in ranked[:k]]
