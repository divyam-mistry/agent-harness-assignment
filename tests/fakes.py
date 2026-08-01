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
