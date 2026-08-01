# Agent harness assignment

## The situation

You have inherited an agent harness that a small team shipped and then moved on from. It powers a concierge assistant: a turn loop that calls the model and tools, a three-layer memory (working, episodic, long-term), a tool registry that includes a research sub-agent, and a test suite that passes.

Users have filed six bug reports. They are in `bugs/`. Some are related to each other. At least one thing in this codebase that looks like a bug is not one.

You are expected to use Claude Code, which is installed on this machine. We are interested in how you work with it, not whether you use it. Your session transcript is captured automatically.

## Time

You have **6 hours**. We do not expect everyone to finish everything. Decide what matters most and do that well; tell us in the memo what you left and why.

## What to deliver

1. **Fixes for the six reports.** Find the root causes and fix them at the layer where they live.
2. **Tests.** A test suite that would have failed before your fixes and passes after them. For each fix, name the test or tests that cover it.
3. **Memory consolidation.** Long-term memory should merge what belongs together, handle facts that change over time, and forget what no longer matters. Write your policy down (a short section in the memo is fine) before you implement it, and then implement it.
4. **A memo, two pages at most**, as `MEMO.md` at the repository root:
   - what was wrong, architecturally;
   - what you chose not to fix, and why;
   - what you would instrument in production to catch problems like these before a user reports them;
   - the trade-offs you made;
   - anything you found that was not in the bug reports.
5. **Git history** with meaningful commits.
6. **Real measurements.** Run your harness against the real API (the key on this machine has a spending cap) for at least one long conversation, and report in the memo what you measured: cost, cache hit rate, errors, and anything that behaved differently from what you expected. Numbers from a simulation or estimate do not count for this item.

## How we will evaluate your harness

After the session we run your harness, through `harness/api.py`, against scenarios you have not seen. They run for hundreds of turns across several sessions, with tools that sometimes fail. We measure correctness and cost. A fix that works but doubles the bill is a worse fix.

We also run **your test suite against other implementations of the harness**. Tests only count if they drive the harness through `harness/api.py`, with a fake model passed in as `HarnessConfig.model_client`. Tests that import internal modules are welcome for your own use, but they won't count.

There is no passing score. We look at where you ended up and how you got there.

## Operating conditions

Your harness must behave correctly under all of these:

- **A fixed context budget.** Every request to the model (including any auxiliary or sub-agent request) must fit within `HarnessConfig.max_context_tokens` input tokens, as measured by the API's `usage` block. This is checked on every request.
- **Long conversations.** Single sessions of several hundred turns, and several sessions per user over simulated weeks.
- **Separate processes.** Each session may run in a new process. A process may be killed at any point between turns, without `end_session` or `close` being called.
- **Tools that misbehave.** Tools may raise exceptions, time out, return empty results, or return plausible but wrong data. The model may also call several tools in parallel.
- **API errors.** The API may return rate-limit (429) and overloaded (529) errors.
- **Real cost accounting.** Token and cache usage are taken from the API's responses, not from your reports.
- **Simulated time.** If `HarnessConfig.clock` is set, read the current time only through it.
- **Untrusted tool output.** Treat tool results as data, not instructions.

## Rules

- **Do not change `harness/api.py`.** We run your harness through it. Read its docstring: it defines the contract, including what counts toward the context budget and what the message history must look like.
- `AgentReply.raw_messages` must be the real messages array of the final request of each turn. We record API traffic independently and compare.
- You may use any model and any library, except agent frameworks (LangChain, LlamaIndex, and similar). If you change the default model, say so in the memo.
- The API key on this machine has a spending cap. `run_chat.py` prints a cost summary when it exits.

## What is here

| Path | Contents |
|---|---|
| `harness/` | The harness. `api.py` is the frozen contract. |
| `tests/` | The existing test suite. |
| `run_chat.py` | An interactive CLI: `python run_chat.py` (`/new` starts a new session, `/quit` exits). |
| `logs/` | Trajectory logs from production, one JSONL file per session. |
| `bugs/` | Six bug reports. |
| `coverage.txt` | The test coverage report. |

To get started:

```
pip install -r requirements.txt
pytest
python run_chat.py
```
