"""Extracts durable facts about the user from an exchange (structured output)."""

from __future__ import annotations

import re

EXTRACTION_INSTRUCTIONS = """\
You help an assistant remember durable things about its user. Read the exchange and call
record_facts with the facts the USER stated about themselves: where they live or work,
people and pets, preferences, constraints and standing instructions ("never ship to my
office"), and corrections to something they said earlier.

Do not record: one-off requests or questions, order numbers or other identifiers the user is
merely asking about, small talk, things only the assistant said, or places the user is only
asking about or comparing as trip options (use topic home_city only when the user says where
they live or that they moved). If the user corrects an earlier fact, record only the new fact.
Each fact needs `quote`: the exact words from the USER's message that state it.
Call record_facts with an empty list if there is nothing to record."""

EXTRACT_TOOL = {
    "name": "record_facts",
    "description": "Record durable facts the user stated about themselves.",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "short snake_case label; same thing, same label (home_city, diet, seat_preference)"},
                        "statement": {"type": "string", "description": "short first-person sentence, e.g. 'I live in Porto.'"},
                        "quote": {"type": "string", "description": "exact words from the user's message that state this"},
                    },
                    "required": ["topic", "statement", "quote"],
                },
            }
        },
        "required": ["facts"],
    },
}


def build_request(user_text: str, assistant_text: str) -> list[dict]:
    return [{"role": "user", "content": f"USER: {user_text[:4000]}\nASSISTANT: {assistant_text[:600]}"}]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


_PLAIN_LINE = re.compile(r"^(?:([a-z][a-z0-9_]{1,40})\s*\|\s*)?((?:i|i'm|i've|my|we|our|never)\b.*)$", re.IGNORECASE)


def parse_facts(content: list[dict], user_text: str) -> list[tuple[str | None, str]]:
    """(topic, statement) pairs from a ``record_facts`` call.

    Each fact must be backed by a quote that really occurs in the user's message, which drops
    facts the model took from the assistant's reply or inferred. If the model answered in plain
    text instead (e.g. a simple test double), first-person lines are accepted, optionally as
    ``topic | sentence``.
    """
    calls = [b for b in content if b.get("type") == "tool_use" and b.get("name") == EXTRACT_TOOL["name"]]
    if calls:
        user = _norm(user_text)
        facts = []
        for item in calls[0].get("input", {}).get("facts") or []:
            quote = _norm(str(item.get("quote", "")))
            statement = str(item.get("statement", "")).strip()
            if statement and quote and quote in user:
                facts.append((str(item.get("topic") or "").strip() or None, statement))
        return facts
    facts = []
    for line in "".join(b.get("text", "") for b in content if b.get("type") == "text").splitlines():
        match = _PLAIN_LINE.match(line.strip().lstrip("-*• ").strip())
        if match and len(line) <= 300:
            facts.append((match.group(1), match.group(2).strip()))
    return facts
