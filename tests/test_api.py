"""Contract-level tests for harness.api."""

from __future__ import annotations

from harness.api import AgentReply


def test_new_session_returns_unique_ids(harness):
    first = harness.new_session("alice")
    second = harness.new_session("alice")
    assert first != second


def test_run_turn_returns_reply_text(harness, fake_model):
    fake_model.replies["Hello"] = "Hi there, how can I help?"
    session = harness.new_session("alice")
    reply = harness.run_turn(session, "Hello")
    assert isinstance(reply, AgentReply)
    assert reply.text == "Hi there, how can I help?"


def test_reply_counts_api_calls(harness):
    session = harness.new_session("alice")
    reply = harness.run_turn(session, "Hello")
    assert reply.api_calls_made >= 1


def test_reply_reports_token_usage(harness):
    session = harness.new_session("alice")
    reply = harness.run_turn(session, "Hello")
    assert reply.input_tokens > 0
    assert reply.output_tokens > 0
    assert reply.cache_read_tokens >= 0


def test_raw_messages_contains_user_message(harness):
    session = harness.new_session("alice")
    reply = harness.run_turn(session, "Where is my parcel?")
    assert any(m["role"] == "user" and m["content"] == "Where is my parcel?" for m in reply.raw_messages)


def test_end_session_is_idempotent(harness):
    session = harness.new_session("alice")
    harness.run_turn(session, "Hello")
    harness.end_session(session)
    harness.end_session(session)


def test_memory_snapshot_empty_for_new_user(harness):
    assert harness.memory_snapshot("nobody") == []


def test_multi_turn_conversation_keeps_history(harness, fake_model):
    session = harness.new_session("alice")
    for i in range(12):
        fake_model.replies[f"message {i}"] = f"reply {i}"
        reply = harness.run_turn(session, f"message {i}")
        assert reply.text == f"reply {i}"
    assert len([m for m in reply.raw_messages if m["role"] == "user"]) >= 12
