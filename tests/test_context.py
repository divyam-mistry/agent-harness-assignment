"""Prompt assembly and budgeting."""

from __future__ import annotations

from harness.context import STATIC_INSTRUCTIONS, ContextBuilder, count_tokens, with_recalled_facts


def _history(n: int, size: int = 50) -> list[dict]:
    history = []
    for i in range(n):
        history.append({"role": "user", "content": f"question {i} " + "x" * size})
        history.append({"role": "assistant", "content": [{"type": "text", "text": f"answer {i} " + "y" * size}]})
    return history


def test_count_tokens_grows_with_history():
    assert count_tokens(_history(4)) > count_tokens(_history(2)) > 0


def test_fit_drops_messages_when_over_budget():
    history = _history(20)
    before = len(history)
    dropped = ContextBuilder(max_context_tokens=500).fit(history)
    assert dropped > 0
    assert len(history) < before


def test_fit_keeps_short_history_intact():
    history = _history(2)
    assert ContextBuilder(max_context_tokens=100_000).fit(history) == 0
    assert len(history) == 4


def test_system_blocks_include_instructions():
    blocks = ContextBuilder(1000).system_blocks({"session": "Session information: test"}, ("session", "instructions"))
    assert any(b["text"] == STATIC_INSTRUCTIONS for b in blocks)
    assert "cache_control" in blocks[-1]


def test_recalled_facts_are_included_with_the_message():
    text = with_recalled_facts("Which train should I take?", ["I live in Porto."])
    assert "Which train should I take?" in text and "I live in Porto." in text
    assert with_recalled_facts("Hello", []) == "Hello"
