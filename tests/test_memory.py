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


def test_parse_facts_ignores_none_and_bullets():
    assert parse_facts("NONE") == []
    assert parse_facts("- I live in Porto.\n* I have a cat.\n") == ["I live in Porto.", "I have a cat."]


def test_session_facts_reach_long_term_memory(harness, fake_model):
    fake_model.default = "I live in Porto."
    session = harness.new_session("alice")
    harness.run_turn(session, "I live in Porto.")
    harness.end_session(session)
    assert any(item["text"] == "I live in Porto." for item in harness.memory_snapshot("alice"))
