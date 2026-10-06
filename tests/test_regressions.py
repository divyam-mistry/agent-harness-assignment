"""Regression tests for the six bug reports and the operating conditions.

Everything is driven through ``harness.api`` with a fake ``model_client``; no internal module is imported.
Each test names the report it guards in its docstring.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import anthropic
import httpx2 as httpx
import pytest
from fakes import StrictModel, request_text

from harness.api import Harness, HarnessConfig, ToolResult, ToolSpec

OPTIONS = {"api_base_delay_s": 0.0}


def tool(name, handler):
    return ToolSpec(name=name, description=name, input_schema={"type": "object", "properties": {}}, handler=handler)


def make(tmp_path, model, tools=(), clock=None, budget=24_000, state="state", **options):
    config = HarnessConfig(model_client=model, max_context_tokens=budget, clock=clock, options={**OPTIONS, **options})
    return Harness(config, list(tools), tmp_path / state)


def last_user_text(request):
    for m in reversed(request["messages"]):
        if isinstance(m["content"], str):
            return m["content"]
    return ""


def tool_results(request):
    return [b for m in request["messages"] if isinstance(m["content"], list) for b in m["content"]
            if b.get("type") == "tool_result"]


class Clock:
    def __init__(self, start="2026-10-01T09:00:00"):
        self.now = datetime.fromisoformat(start)

    def __call__(self):
        return self.now

    def advance(self, days):
        self.now += timedelta(days=days)


# --- CX-4471: forgets things from earlier in a long chat -------------------------------------------

def test_cx4471_early_facts_survive_a_long_conversation_within_budget(tmp_path):
    budget = 4_000
    model = StrictModel()
    h = make(tmp_path, model, budget=budget)
    s = h.new_session("maren")
    filler = " The hotels near the station look fine and the train times work for me." * 4
    h.run_turn(s, "My order number is KX7P2M9QD4RA. Never ship the rain jacket to my office.")
    for i in range(40):
        h.run_turn(s, f"Option {i}: compare hotel {i} and train {i}.{filler}")
    reply = h.run_turn(s, "Please put everything together.")
    seen = request_text({"messages": reply.raw_messages})
    assert "KX7P2M9QD4RA" in seen and "office" in seen
    assert not model.violations
    assert max(tokens for _, tokens in model.usages) <= budget
    h.close()


# --- CX-4502: API error 400 in tool-heavy conversations ---------------------------------------------

def test_cx4502_history_stays_valid_when_old_turns_are_dropped(tmp_path):
    def main(request, step):
        # Two parallel lookups first, then answer.
        if tool_results(request) and request["messages"][-1]["role"] == "user" and not isinstance(
            request["messages"][-1]["content"], str
        ):
            return "Done."
        return [("lookup", {"q": "a"}), ("lookup", {"q": "b"})]

    model = StrictModel(main=main)
    lookup = tool("lookup", lambda args: ToolResult(ok=True, content="x" * 900))
    h = make(tmp_path, model, [lookup], budget=3_500)
    s = h.new_session("tomasz")
    for i in range(40):
        reply = h.run_turn(s, f"Look up thing number {i} please.")
        assert reply.text == "Done."
    assert not model.violations, model.violations[:3]
    assert max(tokens for _, tokens in model.usages) <= 3_500
    h.close()


def test_cx4502_a_single_turn_with_many_big_tool_results_fits(tmp_path):
    state = {"n": 0}

    def main(request, step):
        state["n"] += 1
        return [("lookup", {"q": state["n"]})] if state["n"] < 12 else "All done."

    model = StrictModel(main=main)
    h = make(tmp_path, model, [tool("lookup", lambda a: ToolResult(ok=True, content="y" * 3000))], budget=4_000)
    s = h.new_session("tomasz")
    assert h.run_turn(s, "Check everything.").text == "All done."
    assert not model.violations
    assert max(tokens for _, tokens in model.usages) <= 4_000
    h.close()


# --- CX-4540: same failing call 30 times, then a blank answer ---------------------------------------

def test_cx4540_failing_tool_is_not_hammered_and_the_user_gets_an_answer(tmp_path):
    calls = []

    def flaky(args):
        calls.append(args)
        raise TimeoutError("calendar backend did not respond")

    model = StrictModel(main=lambda request, step: [("calendar", {"date": "2026-10-12"})])  # never gives up
    h = make(tmp_path, model, [tool("calendar", flaky)])
    s = h.new_session("daniel")
    reply = h.run_turn(s, "Find me a free 45 minute slot next week.")
    assert reply.text.strip()
    assert len(calls) <= 2
    assert reply.api_calls_made <= 20
    errors = [b for r in model.requests for b in tool_results(r) if b.get("is_error")]
    assert errors and "did not respond" in str(errors[0]["content"])
    h.close()


def test_cx4540_a_failure_is_not_remembered_as_a_good_result(tmp_path):
    calls = []

    def sometimes(args):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("first call fails")
        return ToolResult(ok=True, content="slots: 09:00")

    def main(request, step):
        results = tool_results(request)
        if not results:
            return [("calendar", {"date": "2026-10-12"})]
        if results[-1].get("is_error"):
            return [("calendar", {"date": "2026-10-12"})]
        return "Free at 09:00."

    h = make(tmp_path, StrictModel(main=main), [tool("calendar", sometimes)])
    s = h.new_session("daniel")
    assert h.run_turn(s, "Any slot?").text == "Free at 09:00."
    assert len(calls) == 2
    h.close()


def test_cx4540_a_hanging_tool_times_out(tmp_path):
    def main(request, step):
        results = tool_results(request)
        return f"tool said: {results[-1]['content']}" if results else [("slow", {})]

    h = make(tmp_path, StrictModel(main=main), [tool("slow", lambda a: time.sleep(3) or ToolResult(ok=True, content="late"))],
             tool_timeout_s=0.2)
    s = h.new_session("u")
    started = time.monotonic()
    reply = h.run_turn(s, "go")
    assert time.monotonic() - started < 2.5
    assert "timed out" in reply.text
    h.close()


def test_empty_model_reply_is_never_shown_as_a_blank_answer(tmp_path):
    h = make(tmp_path, StrictModel(main=lambda request, step: ""))
    s = h.new_session("u")
    assert h.run_turn(s, "hello").text.strip()
    h.close()


# --- CX-4533: research ignores what was agreed --------------------------------------------------------

def test_cx4533_research_agent_is_told_what_the_user_asked_for(tmp_path):
    def main(request, step):
        if "research" in last_user_text(request).lower() and not tool_results(request):
            return [("research", {"task": "Find hotels."})]  # the main model forgot to pass the constraints
        return "Here you go."

    model = StrictModel(main=main)
    h = make(tmp_path, model)
    s = h.new_session("priya")
    h.run_turn(s, "I need vegan breakfast options, and we are only looking at Porto.")
    h.run_turn(s, "Please research hotels for me.")
    sub = [r for r in model.requests if r["kind"] == "subagent"]
    assert sub
    seen = request_text(sub[0]).lower()
    assert "vegan" in seen and "porto" in seen
    h.close()


def test_cx4533_subagent_requests_respect_the_budget(tmp_path):
    n = {"i": 0}

    def sub(request, step):
        n["i"] += 1
        return [("lookup", {"q": n["i"]})] if n["i"] < 9 else "Findings."

    model = StrictModel(
        main=lambda r, s: [("research", {"task": "dig"})] if not tool_results(r) else "ok", subagent=sub
    )
    h = make(tmp_path, model, [tool("lookup", lambda a: ToolResult(ok=True, content="z" * 5000))], budget=4_000)
    s = h.new_session("u")
    reply = h.run_turn(s, "research this")
    assert reply.subagent_api_calls >= 2
    assert max(t for kind, t in model.usages if kind == "subagent") <= 4_000
    h.close()


# --- CX-4518: still thinks I live in Paris ------------------------------------------------------------

def test_cx4518_a_move_supersedes_the_old_home(tmp_path):
    clock = Clock()
    model = StrictModel()
    for text in ("I live in Paris.", "I moved from Paris to Berlin."):
        h = make(tmp_path, model, clock=clock)
        s = h.new_session("ines")
        h.run_turn(s, text)
        h.end_session(s)
        h.close()
        clock.advance(21)
    h = make(tmp_path, model, clock=clock)
    items = h.memory_snapshot("ines")
    active = [i["text"] for i in items if i["active"]]
    assert any("Berlin" in t for t in active)
    assert not any("Paris" in t and "Berlin" not in t for t in active)
    s = h.new_session("ines")
    reply = h.run_turn(s, "Where do I live? What trains should I take?")
    shown = last_user_text({"messages": reply.raw_messages})
    assert "Berlin" in shown and "I live in Paris" not in shown
    h.close()


# --- CX-4561: knows more about me every restart --------------------------------------------------------

def test_cx4561_repeated_facts_are_merged_and_not_volunteered_when_irrelevant(tmp_path):
    clock = Clock()
    model = StrictModel()
    sentences = ["I am vegan.", "I am vegan.", "I prefer the window seat once.", "My dog is called Pixel."]
    for text in sentences:
        h = make(tmp_path, model, clock=clock)
        s = h.new_session("hannah")
        h.run_turn(s, text)
        h.end_session(s)
        h.close()
        clock.advance(1)
    h = make(tmp_path, model, clock=clock)
    active = [i for i in h.memory_snapshot("hannah") if i["active"]]
    assert len([i for i in active if "vegan" in i["text"].lower()]) == 1
    s = h.new_session("hannah")
    reply = h.run_turn(s, "What is the capital of Portugal?")
    shown = last_user_text({"messages": reply.raw_messages})
    assert "window" not in shown and "Pixel" not in shown and "vegan" not in shown
    reply = h.run_turn(s, "What should my dog eat?")
    assert "Pixel" in last_user_text({"messages": reply.raw_messages})
    h.close()


def test_cx4561_changed_mind_keeps_only_the_latest_version(tmp_path):
    clock = Clock()
    model = StrictModel()
    for text in ("I prefer window seats on trains.", "I prefer aisle seats on trains."):
        h = make(tmp_path, model, clock=clock)
        s = h.new_session("hannah")
        h.run_turn(s, text)
        h.end_session(s)
        h.close()
        clock.advance(7)
    h = make(tmp_path, model, clock=clock)
    active = [i["text"] for i in h.memory_snapshot("hannah") if i["active"]]
    assert active == ["I prefer aisle seats on trains."]
    h.close()


def test_one_off_remarks_are_forgotten_but_repeated_facts_stay(tmp_path):
    clock = Clock()
    model = StrictModel()
    h = make(tmp_path, model, clock=clock)
    s = h.new_session("u")
    h.run_turn(s, "I once mentioned liking cheese in passing.")
    h.run_turn(s, "My sister is called Anna.")
    h.run_turn(s, "I own a red bicycle.")
    h.run_turn(s, "I own a red bicycle.")
    h.end_session(s)
    clock.advance(60)
    s = h.new_session("u")  # a new session applies the forgetting policy
    active = [i["text"] for i in h.memory_snapshot("u") if i["active"]]
    assert "My sister is called Anna." in active and "I own a red bicycle." in active
    assert not any("cheese" in t for t in active)
    h.close()


def test_facts_survive_a_process_killed_before_end_session(tmp_path):
    model = StrictModel()
    h = make(tmp_path, model)
    s = h.new_session("u")
    h.run_turn(s, "I live in Lisbon.")
    # no end_session, no close: the process dies here
    h2 = make(tmp_path, model)
    assert any("Lisbon" in i["text"] and i["active"] for i in h2.memory_snapshot("u"))
    h2.close()


def test_a_session_can_continue_in_a_new_process(tmp_path):
    model = StrictModel()
    h = make(tmp_path, model)
    s = h.new_session("u")
    h.run_turn(s, "My order number is AB12CD34EF56.")
    h2 = make(tmp_path, model)
    reply = h2.run_turn(s, "What was my order number?")
    assert "AB12CD34EF56" in request_text({"messages": reply.raw_messages})
    assert not model.violations
    h2.close()


def test_a_corrupt_memory_file_is_set_aside_not_silently_wiped(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "memory.json").write_text('[{"user_id": "u", "text": "I live in Rome', encoding="utf-8")
    h = make(tmp_path, StrictModel())
    assert h.memory_snapshot("u") == []
    assert any(p.suffix == ".corrupt" for p in state.iterdir())
    h.close()


# --- operating conditions ---------------------------------------------------------------------------------

def _api_error(status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize("status", [429, 529])
def test_rate_limits_and_overload_are_retried(tmp_path, status):
    model = StrictModel(errors=[_api_error(status), _api_error(status)])
    h = make(tmp_path, model)
    s = h.new_session("u")
    assert h.run_turn(s, "hello").text == "OK"
    h.close()


def test_a_failed_turn_leaves_the_session_usable(tmp_path):
    model = StrictModel(errors=[_api_error(400)])
    h = make(tmp_path, model)
    s = h.new_session("u")
    with pytest.raises(anthropic.APIStatusError):
        h.run_turn(s, "first")
    assert h.run_turn(s, "second").text == "OK"
    assert not model.violations
    assert [m["content"] for m in model.requests[-2]["messages"] if m["role"] == "user"][0] == "second"
    h.close()


def test_the_cacheable_prefix_does_not_change_between_turns(tmp_path):
    model = StrictModel()
    clock = Clock()
    h = make(tmp_path, model, clock=clock)
    s = h.new_session("u")
    for i in range(3):
        h.run_turn(s, f"message {i}")
        clock.advance(0.01)
    mains = [r for r in model.requests if r["kind"] == "main"]
    assert len({json.dumps(r["system"]) for r in mains}) == 1
    assert len({json.dumps(r["tools"]) for r in mains}) == 1
    marked = [b for m in mains[-1]["messages"] if isinstance(m["content"], list) for b in m["content"] if "cache_control" in b]
    assert len(marked) == 1
    h.close()


def test_every_request_the_harness_makes_fits_the_budget(tmp_path):
    model = StrictModel()
    h = make(tmp_path, model, budget=3_000)
    s = h.new_session("u")
    for i in range(30):
        h.run_turn(s, f"I like thing number {i}. " + "words " * 80)
    assert {kind for kind, _ in model.usages} >= {"main", "extract", "summary"}
    assert max(t for _, t in model.usages) <= 3_000
    h.close()


def test_extractor_chatter_is_not_stored_as_facts(tmp_path):
    class Chatty(StrictModel):
        def create(self, **kwargs):
            if self.kind(kwargs) == "extract":
                self.requests.append({"kind": "extract", **kwargs})
                from fakes import Message
                text = "**Explanation:** The user made no durable statements.\n- NONE**\ntopic | sentence\nhome_city | I live in Oslo."
                return Message.model_validate({"id": "m", "type": "message", "role": "assistant", "model": "x",
                    "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 5}})
            return super().create(**kwargs)

    h = make(tmp_path, Chatty())
    s = h.new_session("u")
    h.run_turn(s, "hello, I live in Oslo")
    assert [i["text"] for i in h.memory_snapshot("u")] == ["I live in Oslo."]
    h.close()


def test_topic_word_finds_the_fact(tmp_path):
    model = StrictModel()
    h = make(tmp_path, model)
    s = h.new_session("u")
    h.run_turn(s, "I am vegan.")
    h.end_session(s)
    s = h.new_session("u")
    reply = h.run_turn(s, "Which diet do I follow?")
    assert "I am vegan." in last_user_text({"messages": reply.raw_messages})
    h.close()


def _recording_model(facts_for):
    """StrictModel whose extractor answers through the record_facts tool, as the real one is forced to."""
    from fakes import Message

    class Structured(StrictModel):
        def create(self, **kwargs):
            if "record_facts" in str(kwargs.get("tools")):
                self.requests.append({"kind": "extract", **kwargs})
                user = str(kwargs["messages"][-1]["content"])
                return Message.model_validate({"id": "m", "type": "message", "role": "assistant", "model": "x",
                    "content": [{"type": "tool_use", "id": "toolu_x", "name": "record_facts",
                                 "input": {"facts": facts_for(user)}}],
                    "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 5}})
            return super().create(**kwargs)

    return Structured()


def test_a_fact_the_user_did_not_say_is_not_stored(tmp_path):
    """The extractor also sees the assistant's reply; a fact whose quote is not in the user's words is dropped."""
    def facts(user):
        return [
            {"topic": "home_city", "statement": "I live in Berlin.", "quote": "I live in Berlin"},
            {"topic": "home_city", "statement": "I live in Porto.", "quote": "I live in Porto"},  # invented
        ]

    h = make(tmp_path, _recording_model(facts))
    s = h.new_session("u")
    h.run_turn(s, "hello, I live in Berlin")
    assert [i["text"] for i in h.memory_snapshot("u") if i["active"]] == ["I live in Berlin."]
    h.close()


def test_comparing_places_for_a_trip_does_not_change_the_home_city(tmp_path):
    """CX-4518 follow-up, seen in a real run: the extractor turned 'Berlin or Porto?' into 'I live in Porto'."""
    def facts(user):
        if "comparing" in user:
            return [{"topic": "home_city", "statement": "I live in Porto.", "quote": "I live in Porto"}]
        return [{"topic": "home_city", "statement": "I live in Berlin.", "quote": "I live in Berlin"}]

    h = make(tmp_path, _recording_model(facts))
    s = h.new_session("u")
    h.run_turn(s, "I live in Berlin.")
    for _ in range(3):
        h.run_turn(s, "I am comparing Berlin and Porto for a long weekend, with prices if you can.")
    assert [i["text"] for i in h.memory_snapshot("u") if i["active"]] == ["I live in Berlin."]
    h.close()


def test_a_tool_call_made_after_a_write_sees_the_new_state(tmp_path):
    """The same read repeated within a turn must hit the tool again: an earlier write may have changed the answer."""
    state = {"status": "open"}

    def look(args):
        return ToolResult(ok=True, content=state["status"])

    def cancel(args):
        state["status"] = "cancelled"
        return ToolResult(ok=True, content="done")

    plan = [[("look", {"id": 1})], [("cancel", {"id": 1})], [("look", {"id": 1})]]

    def main(request, step):
        n = sum(1 for m in request["messages"] if m["role"] == "assistant")
        return plan[n] if n < len(plan) else "finished"

    model = StrictModel(main=main)
    h = make(tmp_path, model, [tool("look", look), tool("cancel", cancel)])
    s = h.new_session("u")
    h.run_turn(s, "cancel it and check")
    assert [b["content"] for b in tool_results([r for r in model.requests if r["kind"] == "main"][-1])] == ["open", "done", "cancelled"]
    h.close()
