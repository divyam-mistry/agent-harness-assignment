"""Persistent long-term memory: merge, supersede, forget, retrieve.

Consolidation policy (also summarised in MEMO.md)
-------------------------------------------------
Write (``add``) - every extracted fact is stored the moment it is learned, so a killed process
loses nothing. On the way in a fact is compared with the user's *active* facts:
  * same topic key and similar wording    -> merged: one item, ``mentions`` + 1, newest wording;
  * same topic key, different content      -> the old item is superseded (inactive, ``superseded_by``)
                                              and the new one becomes the truth (a move, a changed mind);
  * no key but near-identical wording      -> merged the same way;
  * otherwise                              -> a new item.
The topic key is given by the extractor, or inferred by ``infer_key`` for the few common slots.

Retrieval (``search``) - only active items, only items sharing a content word with the query,
ranked by IDF-weighted overlap with a small boost for repeated facts. Nothing else is returned for an
unrelated query: a passing remark must not be volunteered. Only the identity facts in ``ALWAYS_OFFER``
(home city, name) are always offered.

Forgetting (``sweep``) - by the injected clock, never wall time:
  * a fact mentioned once, with no topic, and never recalled      -> expires after 30 days;
  * a reinforced (>= 2 mentions) or ever-recalled fact              -> expires after 180 days unused;
  * a keyed fact (home city, diet, ...)                             -> expires after 365 days unused;
  * superseded and expired items stay in the file as inactive items (an audit trail, never retrieved).
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "i me my mine we our you your am is are was were be been being a an the and or but of to in on at for from with "
    "by as it its this that these those do does did have has had will would can could should not no so if then than "
    "just also very really too about into over up out now always never".split()
)

# Identity facts that are offered to the model in every session, relevant to the query or not.
ALWAYS_OFFER = frozenset({"home_city", "name"})

TTL_ONE_OFF_DAYS = 30
TTL_REINFORCED_DAYS = 180
TTL_KEYED_DAYS = 365


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Lower-cased content words with a crude plural strip, stop words removed."""
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokenize(text) if t not in _STOP}


_FIRST_PERSON = re.compile(r"^(i|i'm|im|we)\b")
_MY_X_IS = re.compile(r"^my ([a-z' -]{2,30}?) (?:is|are|was)\b")


def infer_key(text: str) -> str | None:
    """Topic key for common single-valued facts when the extractor gave none."""
    t = text.lower().strip()
    if re.search(r"\bmy name is\b", t):
        return "name"
    if _FIRST_PERSON.match(t):
        if re.search(r"\b(live|living|based|moved|moving|relocated|relocating|reside|residing)\b", t) and re.search(
            r"\b(in|to|at)\b", t
        ):
            return "home_city"
        if re.search(r"\b(vegan|vegetarian|pescatarian|plant[- ]based|gluten[- ]free|halal|kosher)\b", t):
            return "diet"
        if re.search(r"\b(work|working)\b", t):
            return "work"
        if re.search(r"\bseats?\b", t):
            return "seat_preference"
    m = _MY_X_IS.match(t)
    if m:
        return re.sub(r"\W+", "_", m.group(1)).strip("_")
    return None


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=None)
    except ValueError:
        return None


def _similar(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    inter = len(a & b)
    return inter / len(a | b) >= 0.6 or (min(len(a), len(b)) >= 2 and inter / min(len(a), len(b)) >= 0.85)


class LongTermMemory:
    """Facts the user has shared, stored in ``state_dir/memory.json``."""

    def __init__(self, state_dir: Path, on_warning: Callable[[str], None] | None = None):
        self.path = state_dir / "memory.json"
        self.on_warning = on_warning or (lambda message: None)
        self.facts: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            facts = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError as exc:
            backup = self.path.with_suffix(".corrupt")
            os.replace(self.path, backup)
            self.on_warning(f"could not parse {self.path.name} ({exc}); moved to {backup.name}, starting empty")
            return []
        for f in facts:  # items written by older versions
            f.setdefault("id", uuid.uuid4().hex[:10])
            f.setdefault("active", True)
            f.setdefault("mentions", 1)
            f.setdefault("key", None)
            f.setdefault("updated_at", f.get("created_at"))
        return facts

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.facts, fh, indent=1)
        os.replace(tmp, self.path)

    # -- write ---------------------------------------------------------------------------------

    def add(self, user_id: str, text: str, session_id: str, created_at: str, key: str | None = None) -> dict:
        text = text.strip()
        inferred = infer_key(text)
        key = inferred or key  # our own pattern wins, so the same topic always gets the same label
        tokens = content_tokens(text)
        for old in self.for_user(user_id, active_only=True):
            same_topic = key is not None and old.get("key") == key
            if _similar(tokens, content_tokens(old["text"])):
                old["mentions"] += 1
                old["updated_at"] = created_at
                if len(text) >= len(old["text"]) or same_topic:
                    old["text"] = text
                old["key"] = old.get("key") or key
                return old
        fact = {
            "id": uuid.uuid4().hex[:10], "user_id": user_id, "text": text, "key": key, "session_id": session_id,
            "created_at": created_at, "updated_at": created_at, "last_used_at": None, "mentions": 1, "active": True,
        }
        if key:
            for old in self.for_user(user_id, active_only=True):
                if old.get("key") == key:
                    old.update(active=False, superseded_by=fact["id"], inactive_since=created_at, reason="superseded")
        self.facts.append(fact)
        return fact

    def mark_used(self, ids: list[str], at: str) -> None:
        wanted = set(ids)
        for f in self.facts:
            if f["id"] in wanted:
                f["last_used_at"] = at

    def sweep(self, now: datetime, user_id: str | None = None) -> int:
        """Apply the forgetting policy; returns how many items expired."""
        now = now.replace(tzinfo=None)
        changed = 0
        for f in self.facts:
            if not f["active"] or (user_id and f["user_id"] != user_id):
                continue
            ref = max(filter(None, (_parse(f.get("updated_at")), _parse(f.get("last_used_at")), _parse(f.get("created_at")))),
                      default=now)
            if f.get("key"):
                ttl = TTL_KEYED_DAYS
            elif f["mentions"] >= 2 or f.get("last_used_at"):
                ttl = TTL_REINFORCED_DAYS
            else:
                ttl = TTL_ONE_OFF_DAYS
            if (now - ref).days > ttl:
                f.update(active=False, inactive_since=now.isoformat(), reason="expired")
                changed += 1
        return changed

    # -- read ----------------------------------------------------------------------------------

    def for_user(self, user_id: str, active_only: bool = False) -> list[dict]:
        return [f for f in self.facts if f["user_id"] == user_id and (f["active"] or not active_only)]

    def search(self, user_id: str, query: str, k: int) -> list[dict]:
        facts = self.for_user(user_id, active_only=True)
        core = [f for f in facts if f.get("key") in ALWAYS_OFFER]
        query_tokens = content_tokens(query)
        if not facts or not query_tokens:
            return core
        # the topic label is searchable too, so "which diet do I follow?" finds "I'm vegan." (topic diet)
        doc_tokens = [content_tokens(f["text"]) | content_tokens((f.get("key") or "").replace("_", " ")) for f in facts]
        n = len(facts)
        df = {t: sum(t in d for d in doc_tokens) for t in query_tokens}
        scored = []
        for fact, tokens in zip(facts, doc_tokens):
            shared = query_tokens & tokens
            if not shared:
                continue
            score = sum(math.log(1 + n / df[t]) for t in shared) * (1 + 0.15 * math.log(fact["mentions"]))
            scored.append((score, fact.get("updated_at") or "", fact))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        ranked = [f for _, _, f in scored[:k]]
        return core + [f for f in ranked if f not in core]
