# Memo: concierge harness handover

## What was wrong, architecturally

Five layers failed quietly; none raised an error anyone would notice.

1. **Context budgeting was not about tokens.** `fit()` counted characters, ignored system prompt and tools, and deleted messages two at a time. When one tool-heavy turn outgrew the budget, the first kept message was an orphaned `tool_result`: the 903 `API error 400`s in `logs/` (CX-4502). Dropped content was never summarised, so the order number and "not to the office" vanished (CX-4471).
2. **Tool failures became successes.** Any exception turned into `ok=True, content=""` and was cached, so the model saw an empty answer and retried until the step limit, then returned blank (CX-4540). The cache was engine-wide, so one user's lookup could be served to another.
3. **The prompt cache could never hit.** The system prompt began with the clock and turn number, and there was no message breakpoint. Production logs: 10.7% cache hit, **$0.031 per turn**.
4. **Memory was an append-only list.** No identity (duplicates), no "same topic, newer value" (CX-4518), no relevance cut-off (it always returned k items, CX-4561). Facts were saved only in `end_session`, so a kill lost them, and `memory.json` was written non-atomically (three logged parse warnings; a parse failure silently reset it).
5. **The sub-agent saw only a one-line task**, so it could not honour "vegan" or "Porto" (CX-4533). No budget check; blank output if it ran out of steps.

## Fixes → tests (`tests/test_regressions.py`, via `harness.api` + fake `model_client` only)

| Report | Fix | Tests |
|---|---|---|
| CX-4471 | Token estimate over system+tools+messages; whole turns dropped to a 60% low-water mark and folded into a rolling summary (haiku, chunked to fit the budget) | `test_cx4471_*`, `test_every_request_…fits_the_budget` |
| CX-4502 | Cut only at turn boundaries; if one turn is too big, stub old tool outputs but keep pairs; failed turns roll back | `test_cx4502_*` (2), `test_a_failed_turn_leaves_the_session_usable` |
| CX-4540 | Failures are `ok=False`, never cached; timeout; refuse an identical call after 2 failures; forced final answer (`tool_choice: none`) on step exhaustion | `test_cx4540_*` (3), `test_empty_model_reply_…` |
| CX-4533 | Sub-agent gets a digest of the user's statements and binding-constraint instructions, is budget-checked, fails honestly; `research` description no longer says "prefer it" | `test_cx4533_*` (2) |
| CX-4518 / 4561 | Memory policy below | `test_cx4518_*`, `test_cx4561_*` (2), `test_one_off_remarks_…`, `test_facts_survive_…killed…`, `test_a_corrupt_memory_file…`, `test_extractor_chatter…`, `test_a_fact_the_user_did_not_say…`, `test_topic_word_…` |
| Conditions | 429/529/5xx retry with backoff; sessions persisted per turn and resumable; stable cache prefix | `test_rate_limits_…`, `test_a_session_can_continue_in_a_new_process`, `test_the_cacheable_prefix_…` |

Against the handed-over code, 22 of the 24 current regression-test cases fail (checked in a worktree of the original commit). One of the two that pass, `test_cx4502_history_stays_valid_…`, does not reproduce the bug (`test_cx4502_a_single_turn_…` does).

## Memory policy (written before implementing; `harness/memory/longterm.py`)

- **Write-through** every turn; `end_session` is a final sweep.
- **Merge:** same topic and similar wording → one item, `mentions`+1.
- **Change over time:** same topic, different content → old item `active:false, superseded_by` (kept as audit). Topic comes from the extractor (`topic | sentence`) with regex fallback for home city, name, diet, seat, `my X is …`.
- **Retrieve:** active only, and only if it shares a content word (or topic word) with the query; home city and name always offered; facts already shown this session are not re-injected.
- **Forget** (injected clock only): one-off, never recalled → 30 days; reinforced or recalled → 180; topic-keyed → 365; expired and superseded items stay in the file as inactive, never retrieved.

## Not fixed, and why

- **Retrieval is lexical**: "where am I based?" will not find "I live in Berlin" except via the always-on identity facts. Embeddings would fix it; I judged the dependency and cost not worth it.
- **Retractions without a topic** ("you may ship to my office again") are not linked to the earlier constraint; summary and recency catch most cases, not deterministically.
- **Sub-agent output is still trusted** by the main model; the prompt says to check it, nothing enforces it.
- Hung tool threads cannot be killed in Python; they are abandoned after the timeout.

## Trade-offs

- Token estimate is conservative (3 chars/token, scaled up if the API reports more): wastes about 25% of the window, guarantees the hard budget.
- Cutting to 60% means rarer, bigger cuts (stable cached prefix) at the cost of some old-turn detail, covered by the summary.
- Session date frozen at session start (cache stability); a session crossing midnight shows the old date. State is JSON files, atomic but single-writer.
- Models unchanged (`claude-sonnet-5`, aux `claude-haiku-4-5`). `max_steps` 30→15, `retrieval_k` 8→5.

## What I'd instrument in production

Prompt tokens ÷ budget (alert >90%); cache-read share per session (a drop means the prefix changed); truncation rate; tool errors and repeated-identical-call count per turn (CX-4540 would have shown at once); API errors by status; memory size, merge/supersession rates, extractor lines rejected; share of turns ending in the fallback text; sub-agent cost per turn.

## Real measurements (`measure.py`, real API, budget 8,000)

65 turns, 3 sessions (session 2 killed without `end_session`, each session in a fresh `Harness`), flaky calendar on. Run three times, fixing what each run showed; final run on the final code:

- **Cost $0.631, $0.0097/turn** vs $0.031/turn in the production logs (~3×; workloads differ, indicative only). Earlier runs: $0.535, $0.696, so expect ±15% run to run.
- **Cache hit 67.7% overall, 84.3% on main** (was 10.7%). 100k uncached, 317k cache-read, 51k cache-write, 32k output tokens.
- Max prompt 5,753 vs 8,000 budget (7,620 in the previous run); 13 truncations; 0 API errors; 1 tool failure in 66 calls; no 400s. 186 calls: 94 main, 65 extract, 14 sub-agent, 13 summary.
- **Cross-session recall: 5 of 6 probes pass** (home city, diet, seat, shipping rule). The sixth ("what is my first order id") was ambiguous: the model gave the earliest order in the order history, not the id the user stated. I reworded the probe but did not re-run it.
- **Surprises, each fixed:** (1) haiku often answered with commentary instead of `NONE`; the first parser stored it as facts and once overwrote the home city. Extraction now uses a forced `record_facts` tool call, and every fact must carry a quote that occurs in the user's own message (this replaced the regex filters). (2) A user *comparing* Berlin and Porto for a trip made it emit `home_city | I live in Porto.`, which superseded the real Berlin ("you live in Porto"). The quote requirement drops facts the user did not say (`test_comparing_places_for_a_trip_does_not_change_the_home_city`). It cannot catch a fabricated fact that quotes real words, which the prompt discourages. The final run kept Berlin. (3) Extraction runs every turn (65 of 186 calls) but is about 10% of cost.

## Found outside the bug reports

Tool cache shared across users (leak risk); corrupt `memory.json` silently reset; facts lost on kill; no 429/529 handling; unbounded sub-agent context; `research` description over-promoted the most expensive tool; extraction stored order ids the user was merely asking about.
**Looked like a bug, is not:** recalled facts are appended to the *user* message, not the system prompt. Deliberate: it keeps the cached prefix stable, and they are labelled as the user's possibly outdated statements. The duplicated phrase in logged user messages ("just so you know, just so you know") is simulator noise.

*Latest simplification pass (after the runs above): removed the success replay in the per-turn tool guard (it returned stale reads after a write; `test_a_tool_call_made_after_a_write_…`), the capacity cap and purge, the second key set, dead helpers, and moved extraction to structured output. Checked with one real-API smoke run (4 correct facts, trip-comparison turn did not change the home city); the full 65-turn run was not repeated.*
