"""Interactive chat with the harness using the built-in tools.

Commands: /new (end the session and start another), /quit.
A cost summary is printed on exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness.api import AgentReply, Harness, HarnessConfig
from harness.config import CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, price_for
from harness.tools.builtin import builtin_tools


class CostCounter:
    def __init__(self, model: str):
        self.model = model
        self.turns = 0
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_write = 0
        self.subagent_input = 0
        self.subagent_output = 0

    def add(self, reply: AgentReply) -> None:
        self.turns += 1
        self.api_calls += reply.api_calls_made + reply.subagent_api_calls
        self.input_tokens += reply.input_tokens
        self.output_tokens += reply.output_tokens
        self.cache_read += reply.cache_read_tokens
        self.cache_write += reply.cache_write_tokens
        self.subagent_input += reply.subagent_input_tokens
        self.subagent_output += reply.subagent_output_tokens

    def cost(self) -> float:
        price_in, price_out = price_for(self.model)
        return (
            (self.input_tokens + self.subagent_input) * price_in
            + self.cache_write * price_in * CACHE_WRITE_MULTIPLIER
            + self.cache_read * price_in * CACHE_READ_MULTIPLIER
            + (self.output_tokens + self.subagent_output) * price_out
        ) / 1_000_000

    def report(self) -> str:
        prompt = self.input_tokens + self.cache_read + self.cache_write
        hit = self.cache_read / prompt if prompt else 0.0
        return (
            f"turns={self.turns} api_calls={self.api_calls} input_tokens={self.input_tokens} "
            f"cache_read_tokens={self.cache_read} cache_write_tokens={self.cache_write} "
            f"output_tokens={self.output_tokens} subagent_tokens={self.subagent_input}/{self.subagent_output} "
            f"cache_hit_rate={hit:.1%} est_cost_usd={self.cost():.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="local-user")
    parser.add_argument("--state-dir", type=Path, default=Path(".state"))
    parser.add_argument("--no-flaky", action="store_true", help="disable simulated backend failures")
    args = parser.parse_args()

    config = HarnessConfig(log_dir=args.state_dir / "logs")
    tools = builtin_tools(args.state_dir / "tooldata", flaky=not args.no_flaky)
    harness = Harness(config, tools, args.state_dir)
    counter = CostCounter(config.model)
    session = harness.new_session(args.user)
    print(f"model={config.model} user={args.user} session={session}  (/new, /quit)")
    try:
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/new":
                harness.end_session(session)
                session = harness.new_session(args.user)
                print(f"-- new session {session}")
                continue
            reply = harness.run_turn(session, line)
            counter.add(reply)
            print(f"assistant> {reply.text}\n")
    except KeyboardInterrupt:
        pass
    finally:
        harness.end_session(session)
        harness.close()
        print(f"\n[cost] {counter.report()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
