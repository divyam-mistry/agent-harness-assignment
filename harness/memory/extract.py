"""Extracts durable facts about the user from an exchange."""

from __future__ import annotations

EXTRACTION_INSTRUCTIONS = """\
You help an assistant remember things about its user. Read the exchange and
list every fact the user stated about themselves, their plans, their
preferences, or their instructions. Write each fact as a short sentence in the
user's own voice (first person), one per line, for example:
I live in Porto.
I prefer aisle seats on trains.
Only include facts the user stated. If there are none, output NONE.
"""


def build_request(user_text: str, assistant_text: str) -> list[dict]:
    return [{"role": "user", "content": f"USER: {user_text[:6000]}\nASSISTANT: {assistant_text[:2000]}"}]


def parse_facts(text: str) -> list[str]:
    facts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line and line.upper() != "NONE":
            facts.append(line)
    return facts
