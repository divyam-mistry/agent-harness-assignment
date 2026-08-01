"""Research sub-agent: a separate model loop for multi-source questions."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.telemetry import TrajectoryLogger, UsageTotals
from harness.tools.registry import ToolRegistry, result_block

RESEARCH_TOOL = {
    "name": "research",
    "description": (
        "Ask a research assistant that searches the knowledge base thoroughly and cross-checks several "
        "sources. More reliable than a single lookup: prefer it for factual questions about policies, "
        "products, travel, or orders. Returns its findings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "The research task."}},
        "required": ["task"],
    },
}

SUBAGENT_INSTRUCTIONS = """\
You are a research assistant. Use the available tools to research the task
thoroughly, then reply with a concise answer and cite the documents you used.
"""


@dataclass
class ResearchOutcome:
    content: str
    usage: UsageTotals = field(default_factory=UsageTotals)


class ResearchAgent:
    def __init__(self, client, registry: ToolRegistry, settings, model: str, max_output_tokens: int):
        self.client = client
        self.registry = registry
        self.settings = settings
        self.model = model
        self.max_output_tokens = max_output_tokens

    def run(self, task: str, logger: TrajectoryLogger) -> ResearchOutcome:
        messages: list[dict] = [{"role": "user", "content": task}]
        tools = self.registry.api_definitions(only=self.settings.subagent_tools)
        usage = UsageTotals()
        answer = ""
        for _ in range(self.settings.subagent_max_steps):
            response = self.client.create(
                purpose="subagent",
                logger=logger,
                model=self.model,
                system=SUBAGENT_INSTRUCTIONS,
                messages=messages,
                tools=tools or None,
                max_tokens=self.max_output_tokens,
            )
            usage.merge(response.usage)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                answer = response.text
                break
            results = self.registry.call_many(response.tool_uses)
            for use, result in zip(response.tool_uses, results):
                logger.event("tool_call", scope="subagent", name=use["name"], input=use.get("input"), ok=result.ok)
            messages.append(
                {"role": "user", "content": [result_block(u["id"], r) for u, r in zip(response.tool_uses, results)]}
            )
        return ResearchOutcome(content=answer, usage=usage)
