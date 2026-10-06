"""Test doubles for the Anthropic client."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from anthropic.types import Message

_ids = itertools.count(1)


def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            return content
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        if texts:
            return texts[-1]
    return ""


@dataclass
class FakeModel:
    """Returns canned replies keyed on the last user message.

    ``replies`` maps a user message to the assistant's text. ``tool_requests``
    maps a user message to a single (tool name, input) the assistant should
    call before answering. Unknown messages get ``default``.
    """

    replies: dict[str, str] = field(default_factory=dict)
    tool_requests: dict[str, tuple[str, dict]] = field(default_factory=dict)
    default: str = "OK"
    calls: list[dict] = field(default_factory=list)

    @property
    def messages(self) -> "FakeModel":
        return self

    def create(self, **kwargs) -> Message:
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        last = messages[-1]
        prompt = _last_user_text(messages)
        is_tool_result = isinstance(last["content"], list) and any(
            b.get("type") == "tool_result" for b in last["content"]
        )
        if prompt in self.tool_requests and not is_tool_result:
            name, args = self.tool_requests[prompt]
            content = [{"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": name, "input": args}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": self.replies.get(prompt, self.default)}]
            stop = "end_turn"
        return Message.model_validate(
            {
                "id": f"msg_{next(_ids)}",
                "type": "message",
                "role": "assistant",
                "model": kwargs["model"],
                "content": content,
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20,
                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            }
        )

    def count_tokens(self, **kwargs):
        class _Count:
            input_tokens = 100

        return _Count()


# ---------------------------------------------------------------------------------------------
# A stricter fake: validates every request like the real API would, and reports token usage
# proportional to the request size (4 chars per token), so budget checks mean something.
# ---------------------------------------------------------------------------------------------
import json  # noqa: E402
import re  # noqa: E402
from typing import Callable  # noqa: E402


def _blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def validate_messages(messages: list[dict]) -> list[str]:
    """Problems the Messages API would reject (400) in this ``messages`` array."""
    problems = []
    if not messages or messages[0]["role"] != "user":
        problems.append("first message is not a user message")
    for i, m in enumerate(messages):
        uses = [b["id"] for b in _blocks(m["content"]) if b.get("type") == "tool_use"]
        results = [b["tool_use_id"] for b in _blocks(m["content"]) if b.get("type") == "tool_result"]
        if m["role"] == "assistant" and uses:
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            got = [b["tool_use_id"] for b in _blocks(nxt["content"]) if b.get("type") == "tool_result"] if nxt else []
            if nxt is None or nxt["role"] != "user" or sorted(got) != sorted(uses):
                problems.append(f"tool_use {uses} at {i} not answered by the next message")
        if results:
            prev = messages[i - 1] if i else None
            ids = [b["id"] for b in _blocks(prev["content"]) if b.get("type") == "tool_use"] if prev else []
            if prev is None or prev["role"] != "assistant" or sorted(ids) != sorted(results):
                problems.append(f"tool_result {results} at {i} has no matching tool_use before it")
    return problems


def request_text(kwargs: dict) -> str:
    return json.dumps([kwargs.get("system"), kwargs.get("tools"), kwargs["messages"]], default=str)


@dataclass
class StrictModel:
    """Scripted fake. ``main(request, step)`` returns a text or a list of (tool, input) calls.

    Auxiliary calls are answered by a tiny rule-based extractor / summariser so tests can run
    through ``HarnessConfig.model_client`` without knowing the harness's prompts.
    """

    main: Callable[[dict, int], object] = lambda request, step: "OK"
    subagent: Callable[[dict, int], object] = lambda request, step: "Findings."
    errors: list = field(default_factory=list)  # exceptions to raise on the next calls, in order
    requests: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    usages: list[tuple[str, int]] = field(default_factory=list)

    @property
    def messages(self) -> "StrictModel":
        return self

    def count_tokens(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def kind(kwargs: dict) -> str:
        names = [t["name"] for t in kwargs.get("tools") or []]
        system = json.dumps(kwargs.get("system"))
        if "research" in names:
            return "main"
        if "research assistant" in system:
            return "subagent"
        return "summary" if "summary" in system.lower() else "extract"

    def create(self, **kwargs) -> Message:
        if self.errors:
            raise self.errors.pop(0)
        kind = self.kind(kwargs)
        self.violations += [f"{kind}: {p}" for p in validate_messages(kwargs["messages"])]
        self.requests.append({"kind": kind, **kwargs})
        tokens = len(request_text(kwargs)) // 4
        self.usages.append((kind, tokens))
        step = sum(1 for r in self.requests if r["kind"] == kind)
        if kind in ("main", "subagent"):
            script = self.main if kind == "main" else self.subagent
            reply = script(kwargs, step)
            if kwargs.get("tool_choice") == {"type": "none"} and not isinstance(reply, str):
                reply = "Here is what I have so far."
        elif kind == "extract":
            user = str(kwargs["messages"][-1]["content"]).split("ASSISTANT:")[0].replace("USER:", "")
            reply = "\n".join(s.strip() for s in re.split(r"(?<=[.!?])\s+", user) if re.match(r"(I|My)\b", s.strip())) or "NONE"
        else:  # summary: keep the lines that carry identifiers or constraints
            text = str(kwargs["messages"][-1]["content"])
            keep = [s for s in re.split(r"(?<=[.!?\n])\s+", text) if re.search(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{8,}\b|[Nn]ever|not |don.t|vegan", s)]
            reply = " ".join(dict.fromkeys(keep))[-1500:] or "Nothing notable."
        if isinstance(reply, str):
            content, stop = [{"type": "text", "text": reply}], "end_turn"
        else:
            content = [{"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": n, "input": a} for n, a in reply]
            stop = "tool_use"
        return Message.model_validate({
            "id": f"msg_{next(_ids)}", "type": "message", "role": "assistant", "model": kwargs["model"],
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": tokens, "output_tokens": 20, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0},
        })
