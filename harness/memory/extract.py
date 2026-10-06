"""Extracts durable facts about the user from an exchange."""

from __future__ import annotations

import re

EXTRACTION_INSTRUCTIONS = """\
You help an assistant remember durable things about its user. Read the exchange and
list the facts the USER stated about themselves: where they live or work, people and
pets, preferences, constraints and standing instructions ("never ship to my office"),
and corrections to something they said earlier.

Do not list: one-off requests or questions, order numbers or other identifiers the user
is merely asking about, small talk, things only the assistant said, or anything the
user did not state. Asking about a place, or comparing places as options for a trip (weather,
hotels, "Berlin or Porto"), does not mean the user lives there: use `home_city` only when the USER
explicitly says where they live or that they moved.
Output only the fact lines, no commentary. If the user corrects or changes an earlier fact,
list only the new fact.

Write one fact per line as `topic | sentence`. The sentence is short, in the user's own
voice (first person). The topic is a short snake_case label for what the fact is about;
two facts about the same thing must use the same topic (for example a new home city
replaces the old one under `home_city`). Examples:
home_city | I live in Porto.
seat_preference | I prefer aisle seats on trains.
office_shipping | I never want orders shipped to my office.
If there are no facts, output NONE.
"""


def build_request(user_text: str, assistant_text: str) -> list[dict]:
    return [{"role": "user", "content": f"USER: {user_text[:4000]}\nASSISTANT: {assistant_text[:600]}"}]


_KEYED = re.compile(r"^([a-z][a-z0-9_]{1,40})\s*\|\s*(.+)$")
# Facts are first-person statements; anything else is the model talking about the task, not a fact.
_FIRST_PERSON = re.compile(r"^(i|i'm|i’m|i've|i’ve|i'd|my|we|we're|our|never|please)\b", re.IGNORECASE)
# Second person, or talk about the task itself: the model is commenting, not recording a fact.
_NOT_A_FACT = re.compile(r"\byou(r|rs)?\b|\b(facts?|exchange|user|stated|statements?)\b|\bI(?:'ll| will| don't see| realize)\b|\.\.\.", re.IGNORECASE)


def parse_keyed(text: str) -> list[tuple[str | None, str]]:
    """Parse ``topic | sentence`` lines; plain first-person sentences are accepted with no topic."""
    facts: list[tuple[str | None, str]] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•# ").replace("**", "").strip()
        if not line or line.upper().strip("*. ") == "NONE" or len(line) > 300:
            continue
        match = _KEYED.match(line)
        key, sentence = (match.group(1), match.group(2).strip()) if match else (None, line)
        if key == "topic" or not _FIRST_PERSON.match(sentence) or _NOT_A_FACT.search(sentence):
            continue
        facts.append((key, sentence))
    return facts


_RESIDENCE = re.compile(r"\b(live|lives|living|lived|based|moved|moving|move|relocat\w*|reside|residing|home|hometown)\b", re.IGNORECASE)


def is_grounded(key: str | None, sentence: str, user_text: str) -> bool:
    """A claim about where the user lives needs residence wording in the user's own message.

    The extractor sees the assistant's reply too and sometimes turns places that were merely compared
    ("Berlin or Porto for a weekend") into a new home city, which would supersede the real one.
    """
    if key == "home_city" or _RESIDENCE.search(sentence):
        return bool(_RESIDENCE.search(user_text))
    return True


def parse_facts(text: str) -> list[str]:
    return [fact for _, fact in parse_keyed(text)]
