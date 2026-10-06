# Memo: concierge harness handover

## What was wrong, architecturally

Five different layers each failed quietly. None of them raised an error that anyone would have noticed.

1. **Context budgeting was not about tokens.** `fit()` counted characters, ignored the system prompt and tools, and deleted messages two at a time from the front. Deleting two messages is not deleting a turn. When a single tool-heavy turn outgrew the budget, the first kept message was an orphaned `tool_result`. That is the 903 `API error 400`s in `logs/` (CX-4502). Dropped content was never summarised, so the order number and "not to the office" vanished (CX-4471).
2. **Tool failures were turned into successes.** The registry converted any exception to `ok=True, content=""` and cached it, so the model saw an empty answer and retried the same call until the step limit, then returned a blank (CX-4540). The same cache was engine-wide, so one user's `lookup_order` could be served to another.
3. **The cache could never hit.** The system prompt began with the clock and turn number, and no message-level breakpoint existed. The production logs show a 10.7% cache hit rate at **$0.031 per user turn**.
4. **Memory was an append-only list.** It had no identity (duplicates), no notion of "same topic, newer value" (CX-4518), and no relevance cut-off (it always returned k items, so passing remarks came back as "the most important thing about me", CX-4561). Facts were written only in `end_session`, so a killed process lost them. `memory.json` was written non-atomically; three logged "could not parse memory.json" warnings show it happening, and a parse failure silently started from empty and overwrote it.
5. **The sub-agent was a context-free island.** It received only the model's one-line `task` string, so it could not honour "vegan" or "Porto" (CX-4533). It had no budget check, and it returned blank if it ran out of steps.

## Fixes (root-cause layer → tests, all in `tests/test_regressions.py`, driven only via `harness.api` with a fake `model_client`)

| Report | Fix | Tests |
|---|---|---|
| CX-4471 | Token estimate over system+tools+messages. Whole turns are dropped to a 60% low-water mark and folded into a rolling summary (haiku, chunked so the summariser also fits the budget). | `test_cx4471_early_facts_survive_a_long_conversation_within_budget`, `test_every_request_the_harness_makes_fits_the_budget` |
| CX-4502 | Cuts only at turn boundaries. If one turn alone is too big, old tool outputs are stubbed but pairs are kept. Tool output is capped at 6k chars. Failed turns roll back, so history stays valid. | `test_cx4502_*` (2), `test_a_failed_turn_leaves_the_session_usable` |
| CX-4540 | Failures are `ok=False`, never cached. Per-call timeout. Per-turn guard: an identical call that failed twice is refused without running. On step exhaustion a final `tool_choice: none` call forces an answer. Empty replies get a fallback text. | `test_cx4540_*` (3), `test_empty_model_reply_is_never_shown_as_a_blank_answer` |
| CX-4533 | The sub-agent gets a digest of what the user said plus the summary, with binding-constraint instructions. It is budget-checked and returns an honest "could not complete" on failure. The `research` tool description no longer says "prefer it", which also cut cost. | `test_cx4533_*` (2) |
| CX-4518 / CX-4561 | See policy below. | `test_cx4518_*`, `test_cx4561_*` (2), `test_one_off_remarks_are_forgotten…`, `test_facts_survive_a_process_killed…`, `test_a_corrupt_memory_file…`, `test_extractor_chatter…`, `test_assistant_commentary…`, `test_topic_word_finds_the_fact` |
| Operating conditions | 429/529/5xx retry with backoff (honours `retry-after`). Sessions are persisted per turn and resumable in a new process. Cache prefix is stable. | `test_rate_limits_and_overload_are_retried`, `test_a_session_can_continue_in_a_new_process`, `test_the_cacheable_prefix_does_not_change_between_turns` |

Against the original code, 20 of the 21 regression tests I had at that point failed (checked in a worktree of the handed-over commit). The one that passed was `test_cx4502_history_stays_valid_when_old_turns_are_dropped`, which did not reproduce the orphaned `tool_result`; `test_cx4502_a_single_turn_with_many_big_tool_results_fits` is the one that reproduces it. I added 3 more tests after that check.

## Memory consolidation policy (written before implementing; in `harness/memory/longterm.py`)

- **Write-through.** Every extracted fact is stored immediately (each turn), so a kill loses nothing. `end_session` is only a final sweep.
- **Merge.** Same topic and similar wording, or near-identical wording → one item, `mentions`+1, newest wording.
- **Change over time.** Same topic, different content → the old item becomes `active:false, superseded_by=…` (kept as audit trail). The topic comes from the extractor (`topic | sentence`), with a regex fallback for home city, name, diet, seat, `my X is …`.
- **Retrieve.** Active items only. An item is returned only if it shares a content word (or its topic word) with the query. Home city and name are always offered. Facts already shown in a session are not re-injected, which saves tokens and cache.
- **Forget** (injected clock only): one-off remark, never recalled → 30 days. Reinforced (≥2 mentions) or ever recalled → 180 days. Topic-keyed → 365 days. Inactive items are purged after 90 days. Cap of 150 active per user.

## What I chose not to fix

- **Retrieval is lexical.** It handles "which diet…" through topic labels but not true paraphrase ("where am I based?" → "I live in Berlin"), apart from the always-on identity facts. An embedding index would fix it. I judged that a dependency and cost I could not justify for the time.
- **Negative constraints and retractions that carry no topic** ("you may ship to my office again") are not linked to the earlier constraint. The summary and recency catch most cases, but not deterministically.
- **Sub-agent output is still trusted by the main model.** The instructions tell it to check the findings against the user's statements; no mechanical check exists.
- Stuck tool threads cannot be killed from Python. They are abandoned after the timeout and stay in the pool.

## Trade-offs

- Token estimation is deliberately conservative (3 chars/token, scaled up if the API ever reports more). This wastes about 25% of the window but guarantees the hard budget. I chose the budget over capacity.
- Truncating to 60% means rarer, bigger cuts, so the cached prefix survives for many turns. It costs some recall of old turns, which the summary compensates for at a small haiku cost.
- Session date is frozen at session start so the cached prefix is stable. A session that crosses midnight shows the old date.
- The summariser and extractor use `claude-haiku-4-5` (`aux_model`, unchanged). The main model is unchanged (`claude-sonnet-5`). `max_steps` was lowered from 30 to 15 and `retrieval_k` from 8 to 5.
- Sessions persist as one JSON file per session, and long-term memory as one `memory.json`. Both are written atomically. This is fine for sequential processes, not for concurrent writers.

## What I'd instrument in production

- Per request: prompt tokens ÷ budget (alert at >90%), cache-read share per session (a drop means the prefix changed), and the `truncation` rate.
- **Tool error rate by tool and repeated-identical-call count per turn.** The CX-4540 pattern would have shown immediately.
- API 4xx/5xx counts by status, and retry exhaustion.
- Memory: active items per user, merge rate, supersession rate, share of retrieved facts the model actually used, and extractor output rejected by the parser (the aux model emitted chatter, as seen in my run).
- Share of turns that end in the fallback text; sub-agent calls per turn and the cost they add.

## Real measurements (`measure.py`, real API, sonnet-5 + haiku-4-5, budget 8,000)

65 turns over 3 sessions (session 2 was killed with no `end_session`, and each session ran in a fresh `Harness`). Flaky calendar tool on. The run used the code at commit "Extraction: reject non-fact chatter"; two later extractor/memory fixes (second-person filter, canonical `diet` label) came from this run's output and were not re-measured.

- **Cost $0.535 total, $0.0082 per turn**, against $0.031 per turn in the handed-over production logs (about 3.8× cheaper; the workloads differ, so treat it as indicative).
- **Cache hit rate 72.8% overall (87.1% on the main model)**, against 10.7% before. Tokens: 80.0k uncached, 321.9k cache-read, 40.5k cache-write, 28.6k output.
- Max prompt in any request was 5,613 tokens against the 8,000 budget. The 24% headroom is the cost of the conservative estimator. 11 truncations, 0 API errors, 1 tool failure out of 50 calls, no 400s.
- 177 API calls: 91 main, 65 extract, 10 sub-agent, 11 summary.
- **Behaved differently from expected:** (1) the aux model often emitted commentary instead of `NONE` ("The user made no durable statements…"), and the first parser stored it as facts. It even overwrote the home city once. That is why the parser now only accepts first-person, non-second-person lines. (2) On a first run (before the filter) the "which diet do I follow?" and "where do I live?" probes failed. After the fixes the memory dump looked right but I did not re-run the probes. The last run still had `s3` probes failing for the reasons above, so recall across sessions is the least verified part. (3) Extraction is a call every turn (65 of 177 calls) but only about 10% of cost.

## Found outside the bug reports

- Tool-result cache shared across users (a data-leak risk). Corrupt `memory.json` silently reset. Facts lost on kill. No 429/529 handling. Unbounded sub-agent context. `research` description pushed the model to over-use the most expensive tool. Extraction stored order ids the user was merely asking about, which polluted retrieval.
- **Looked like a bug, is not:** recalled facts are appended to the *user* message instead of the system prompt. That is deliberate. It keeps the cached system prefix stable, and the text is labelled as the user's own, possibly outdated statements. Also the duplicated phrase in the logged user messages ("just so you know, just so you know") is simulator noise, not harness duplication.
