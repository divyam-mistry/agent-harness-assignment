"""Extracts durable facts about the user from an exchange."""

from __future__ import annotations

import re

EXTRACTION_INSTRUCTIONS = """\
You help an assistant remember durable things about its user. Read the exchange and
list the facts the USER stated about themselves: where they live or work, people and
pets, preferences, constraints and standing instructions ("never ship to my office"),
identifiers they gave (order numbers, booking references), and corrections to
something they said earlier.

Do not list: one-off requests or questions, small talk, things only the assistant said,
or anything the user did not state. If the user corrects or changes an earlier fact,
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


def parse_keyed(text: str) -> list[tuple[str | None, str]]:
    """Parse ``topic | sentence`` lines; plain sentences are accepted with no topic."""
    facts: list[tuple[str | None, str]] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.upper() == "NONE":
            continue
        match = _KEYED.match(line)
        facts.append((match.group(1), match.group(2).strip()) if match else (None, line))
    return facts


def parse_facts(text: str) -> list[str]:
    return [fact for _, fact in parse_keyed(text)]
