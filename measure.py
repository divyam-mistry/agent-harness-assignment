"""Long real-API run for the memo: ~100 turns over three sessions (one killed, no end_session).

    set -a; . ./.env; set +a; python measure.py [--budget 8000] [--state-dir .state/measure]

Prints cost, cache hit rate, errors, request sizes and a few correctness probes. Uses the real API.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from harness.api import Harness, HarnessConfig
from harness.config import CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, price_for
from harness.tools.builtin import ORDERS, builtin_tools


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 5, 9, 0)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


def scripts(orders: list[str]) -> list[list[tuple[str, str | None]]]:
    """Three sessions of (message, probe) pairs; a probe is a substring the reply should contain."""
    chatter = ("Thanks. By the way, I am comparing a few options for my autumn trip and want to keep notes on "
               "what the hotels, trains and weather look like before I commit to anything this week. ")
    s1 = [
        (f"Hi! I'm Inès. I live in Paris and I'm vegan. My order id is {orders[0]}. Never ship anything to my office.", None),
        (f"Where is order {orders[0]}?", orders[0]),
        (f"And order {orders[1]}? What is the total?", None),
        ("Find me a free 45-minute slot on 2026-10-13.", None),
        ("Please research hotels in Porto with vegan breakfast options.", None),
        ("What is the returns policy for sale items?", None),
        ("What is 389.00 + 45.00 + 229.00, with 2 of each?", None),
        (f"Look up order {orders[2]} and {orders[3]} please.", None),
    ]
    for i in range(18):
        s1.append((chatter + f"Question {i}: what's the October weather in Lisbon, and is the museum open on Mondays?", None))
        if i % 4 == 1:
            s1.append((f"Check order {orders[4 + i % 5]} for me.", None))
        if i % 5 == 2:
            s1.append((f"Any free 30-minute slot on 2026-10-{14 + i % 5}?", None))
    s1.append(("Remind me: what is my first order id, and where must I not ship things?", orders[0]))
    s2 = [
        ("Hello again. Big news: I moved from Paris to Berlin last month.", None),
        ("What's the best way to get from the airport to the centre where I live?", "Berlin"),
        ("I prefer aisle seats on trains, by the way.", None),
        ("Research the Plus membership terms and tell me whether free returns apply to me.", None),
    ]
    for i in range(14):
        s2.append((chatter + f"Question {i}: compare Berlin and Porto for a long weekend, with prices if you can.", None))
        if i % 3 == 0:
            s2.append((f"Check order {orders[(i + 5) % len(orders)]}.", None))
    s2.append(("Where do I live now, and what seat do I prefer?", "Berlin"))
    s3 = [
        ("Hi, it's Inès again. Where do I live?", "Berlin"),
        ("Remind me what I told you about shipping to my office.", None),
        ("Do I like window seats?", None),
        ("Find a free 60-minute slot on 2026-10-20.", None),
        ("Which diet do I follow?", "vegan"),
    ]
    return [s1, s2, s3]


def cost_of(model: str, u: dict) -> float:
    pin, pout = price_for(model)
    return (u["input_tokens"] * pin + u["cache_creation_input_tokens"] * pin * CACHE_WRITE_MULTIPLIER
            + u["cache_read_input_tokens"] * pin * CACHE_READ_MULTIPLIER + u["output_tokens"] * pout) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--state-dir", type=Path, default=Path(".state/measure"))
    args = ap.parse_args()
    shutil.rmtree(args.state_dir, ignore_errors=True)
    clock = Clock()
    orders = list(ORDERS)
    probes: list[tuple[str, bool]] = []
    turn_ms: list[float] = []
    started = time.time()

    def new_harness() -> Harness:
        config = HarnessConfig(max_context_tokens=args.budget, log_dir=args.state_dir / "logs", clock=clock)
        return Harness(config, builtin_tools(args.state_dir / "tooldata", flaky=True), args.state_dir)

    for n, script in enumerate(scripts(orders), start=1):
        h = new_harness()  # every session starts in a "new process"
        session = h.new_session("ines")
        for message, probe in script:
            t0 = time.time()
            reply = h.run_turn(session, message)
            turn_ms.append((time.time() - t0) * 1000)
            clock.advance(minutes=7)
            if probe:
                probes.append((f"s{n}: {message[:50]}", probe.lower() in reply.text.lower()))
            print(f"[s{n}] {message[:60]!r} -> {reply.text[:80]!r} (api={reply.api_calls_made}, tools={reply.tool_calls_made})", flush=True)
        if n != 2:
            h.end_session(session)
        # session 2 is "killed": no end_session, no close
        clock.advance(days=7)

    h = new_harness()
    snapshot = h.memory_snapshot("ines")
    h.close()

    totals, per_purpose, models = Counter(), defaultdict(Counter), Counter()
    max_prompt, truncations, api_errors, tool_calls, tool_fail = 0, 0, Counter(), 0, 0
    cost = 0.0
    for path in (args.state_dir / "logs").glob("*.jsonl"):
        model = "claude-sonnet-5"
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec["type"] == "request":
                model = rec["model"]
            elif rec["type"] == "response":
                u = rec["usage"]
                prompt = u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]
                max_prompt = max(max_prompt, prompt)
                for k, v in u.items():
                    totals[k] += v
                    per_purpose[rec["purpose"]][k] += v
                per_purpose[rec["purpose"]]["calls"] += 1
                cost += cost_of(model, u)
            elif rec["type"] == "truncation":
                truncations += 1
            elif rec["type"] == "api_error":
                api_errors[rec.get("status")] += 1
            elif rec["type"] == "tool_call":
                tool_calls += 1
                tool_fail += not rec["ok"]
    prompt_total = totals["input_tokens"] + totals["cache_read_input_tokens"] + totals["cache_creation_input_tokens"]
    main_u = per_purpose["main"]
    main_prompt = main_u["input_tokens"] + main_u["cache_read_input_tokens"] + main_u["cache_creation_input_tokens"]
    print("\n=== measurements ===")
    print(f"turns={len(turn_ms)} wall={time.time() - started:.0f}s mean_turn={sum(turn_ms) / len(turn_ms) / 1000:.1f}s")
    print(f"api_calls={sum(p['calls'] for p in per_purpose.values())} by purpose={ {k: v['calls'] for k, v in per_purpose.items()} }")
    print(f"cost_usd={cost:.4f} (per turn {cost / len(turn_ms):.4f})")
    print(f"tokens: uncached_in={totals['input_tokens']} cache_read={totals['cache_read_input_tokens']} "
          f"cache_write={totals['cache_creation_input_tokens']} out={totals['output_tokens']}")
    print(f"cache_hit_rate overall={totals['cache_read_input_tokens'] / prompt_total:.1%} "
          f"main={main_u['cache_read_input_tokens'] / main_prompt:.1%}")
    print(f"max prompt tokens in any request={max_prompt} (budget {args.budget})")
    print(f"truncations={truncations} api_errors={dict(api_errors)} tool_calls={tool_calls} tool_failures={tool_fail}")
    print(f"probes: {[(name, ok) for name, ok in probes]}")
    active = [i for i in snapshot if i["active"]]
    print(f"memory: {len(snapshot)} items, {len(active)} active")
    for i in snapshot:
        print(f"  {'*' if i['active'] else ' '} [{i.get('key')}] {i['text']} (x{i['mentions']}{', ' + i.get('reason', '') if not i['active'] else ''})")


if __name__ == "__main__":
    main()
