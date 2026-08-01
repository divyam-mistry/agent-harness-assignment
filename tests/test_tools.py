"""Tool registry, built-in tools, and the research sub-agent."""

from __future__ import annotations

import json

from harness.api import ToolResult, ToolSpec
from harness.tools.builtin import ORDERS, builtin_tools
from harness.tools.registry import ToolRegistry


def _spec(name, handler):
    return ToolSpec(name=name, description=name, input_schema={"type": "object", "properties": {}}, handler=handler)


def test_registry_lists_tools(tools):
    assert ToolRegistry(tools).names() == sorted(t.name for t in tools)


def test_tool_dispatch_resilient():
    """A misbehaving tool must never crash the turn loop."""

    def broken(args):
        raise RuntimeError("backend exploded")

    result = ToolRegistry([_spec("broken", broken)]).call("broken", {})
    assert result.ok is True


def test_unknown_tool_is_reported():
    result = ToolRegistry([]).call("missing", {})
    assert result.ok is False


def test_builtin_lookup_order(tmp_path):
    lookup = {t.name: t for t in builtin_tools(tmp_path, flaky=False)}["lookup_order"]
    order_id = next(iter(ORDERS))
    result = lookup.handler({"order_id": order_id})
    assert json.loads(result.content)["order_id"] == order_id


def test_builtin_calculate(tmp_path):
    calc = {t.name: t for t in builtin_tools(tmp_path, flaky=False)}["calculate"]
    assert calc.handler({"expression": "(389.00 + 45.00) * 2"}).content == "868.0"


def test_tool_call_round_trip(harness, fake_model):
    fake_model.tool_requests["What is 2+2?"] = ("calculate", {"expression": "2+2"})
    fake_model.replies["What is 2+2?"] = "It is 4."
    session = harness.new_session("alice")
    reply = harness.run_turn(session, "What is 2+2?")
    assert reply.tool_calls_made == 1
    assert reply.text == "It is 4."


def test_subagent_returns_string(harness, fake_model):
    fake_model.tool_requests["Research Porto hotels"] = ("research", {"task": "Find hotels in Porto"})
    session = harness.new_session("alice")
    reply = harness.run_turn(session, "Research Porto hotels")
    assert isinstance(reply.text, str)
    assert reply.subagent_api_calls >= 1


def test_tool_results_latency_recorded():
    def slow(args):
        return ToolResult(ok=True, content="done")

    result = ToolRegistry([_spec("slow", slow)]).call("slow", {})
    assert result.latency_ms >= 0
