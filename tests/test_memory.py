"""Long-term, episodic and extraction behaviour."""

from __future__ import annotations

from harness.memory.consolidate import consolidate
from harness.memory.episodic import EpisodicMemory
from harness.memory.extract import parse_facts
from harness.memory.longterm import LongTermMemory


def test_longterm_add_and_search(tmp_path):
    memory = LongTermMemory(tmp_path)
    memory.add("alice", "I prefer window seats on trains.", "s1", "2026-10-01T09:00:00")
    results = memory.search("alice", "which seat do I like on the train?", k=3)
    assert results[0]["text"] == "I prefer window seats on trains."


def test_longterm_persists_to_disk(tmp_path):
    memory = LongTermMemory(tmp_path)
    memory.add("alice", "My dog is called Pixel.", "s1", "2026-10-01T09:00:00")
    memory.save()
    reloaded = LongTermMemory(tmp_path)
    assert reloaded.for_user("alice")[0]["text"] == "My dog is called Pixel."


def test_search_is_scoped_to_user(tmp_path):
    memory = LongTermMemory(tmp_path)
    memory.add("alice", "I live in Porto.", "s1", "2026-10-01T09:00:00")
    assert memory.search("bob", "where do I live?", k=3) == []


def test_consolidate_moves_episodic_facts(tmp_path):
    memory = LongTermMemory(tmp_path)
    episodic = EpisodicMemory("s1", "alice")
    episodic.add("I am vegetarian.", turn=1, at="2026-10-01T09:00:00")
    assert consolidate(episodic, memory) == 1
    assert memory.for_user("alice")[0]["text"] == "I am vegetarian."


def _text(s):
    return [{"type": "text", "text": s}]


def test_parse_facts_plain_text_ignores_none_and_bullets():
    assert parse_facts(_text("NONE"), "") == []
    assert parse_facts(_text("- I live in Porto.\n* I have a cat.\n"), "") == [
        (None, "I live in Porto."), (None, "I have a cat.")]


def test_parse_facts_structured_requires_the_quote_in_the_user_text():
    call = [{"type": "tool_use", "name": "record_facts", "input": {"facts": [
        {"topic": "home_city", "statement": "I live in Porto.", "quote": "i live in porto"},
        {"topic": "pet", "statement": "I have a cat.", "quote": "I have a cat"}]}}]
    assert parse_facts(call, "Hi! I live in Porto, ok?") == [("home_city", "I live in Porto.")]


def test_session_facts_reach_long_term_memory(harness, fake_model):
    fake_model.default = "I live in Porto."
    session = harness.new_session("alice")
    harness.run_turn(session, "I live in Porto.")
    harness.end_session(session)
    assert any(item["text"] == "I live in Porto." for item in harness.memory_snapshot("alice"))
