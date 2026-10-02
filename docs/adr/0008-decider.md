---
status: accepted
extends: ADR 0003 (the capture gate) and ADR 0006 (the Advisor); source: Laya (convaiinnovations/laya-multilingual, Apache-2.0), run from onnx-community/laya-multilingual-ONNX
---

# A local decision model makes memory's cheap decisions

Some of memoose's decisions pick one answer from a fixed list, and today a heuristic or a whole LLM
call makes them. A turn longer than 400 characters starts a capture keeper, and the keeper often
replies `NOTHING`. Every tool step starts an Advisor review, and the Advisor's own policy is to stay
silent mid-turn unless something is about to go wrong. The recall router is a set of English regex
rules, so a query that contains none of their cue words falls to `hybrid`.

A System One decision model (the category TypeSafe's Jev started) does not generate text. It reads
a state and a typed question and returns calibrated probabilities over the options in one forward
pass. Laya is an open, Apache-2.0 model of this kind. Its multilingual checkpoint (mmBERT-base and
a decision head) has an ONNX export that runs on onnxruntime with no API key, at about 80 ms a call
once loaded.

## Decision

1. **One module, three decisions.** `memoose.decider` holds the runner (laya's `build_sequence` and
   one ONNX pass), the questions, and the decision log. It decides:
   - `capture`: is this slice durable or transient? A transient slice never reaches the keeper.
   - `advise`: is the agent about to do something wrong or risky in this step? A mid-turn step the
     model calls safe is deferred, not dropped. The cursor stays, so the review at the end of the
     turn sees it. The end-of-turn review is never gated.
   - `route`: which recall mode fits the query? The model is asked only when no rule matched, and
     an answer below 0.5 keeps `hybrid`.
2. **On by default.** `MEMOOSE_DECIDER=on|shadow|off`. `shadow` runs the model next to the
   heuristic and only records the result. `onnxruntime` and `tokenizers` are core dependencies.
   `memoose install` fetches the model (~0.65 GB fp16, `--no-model` skips it), and so does
   `memoose decide --fetch`. Nothing downloads it from inside a hook.
3. **Failures are recorded, not silent.** If the model cannot run, the heuristic decides and an
   `error` row goes to `~/.memoose/decisions.jsonl`. `memoose decide --report` shows the rows,
   errors and agreement for each decision.
4. **Every decision is a labelled example.** In shadow mode the log stores what actually happened
   next to each answer: the keeper's verdict for `capture`, whether the Advisor emitted advice for
   `advise`, and the rules' mode for `route`. That gives agreement figures now and fine-tuning data
   later.
5. **Hooks stay stdlib.** A hook calls `memoose decide <name>` with the state on stdin, as the
   keeper already calls the CLI. Only the package imports onnxruntime.

## Why only three

We probed the zero-shot model on memoose's own decisions before wiring anything:

| decision | correct |
| --- | --- |
| advisor gate | 4/4 |
| router (final option wording) | 8/8 |
| capture gate, asked as a choice | 7/8 (3/8 asked as a yes/no) |
| two triples: duplicate, update or coexist | 0/4, always "contradict" |
| same entity under two names | 4/8, says "same" to every pair |
| is this relation functional | 2/6 |
| user dataset or project dataset | 2/4 |
| would memory help this prompt | 4/8 |

The model is good at situational judgments and at chance on questions about the schema itself.
Comparing two triples is plain string equality and stays in code. The other weak decisions stay
with the agent or the keeper until a fine-tune on the log beats them.

## Consequences

- Capture spends a keeper call only on slices the model calls durable. Mid-turn Advisor reviews
  run only on steps it calls risky.
- A concern the Advisor would have raised mid-turn now arrives at the end of the turn, unless the
  model rates the step risky. That trade buys skipping most mid-turn reviews.
- A recall the rules cannot route loads the model, ~1.4 s more in a fresh CLI process. A
  long-running MCP server loads it once.
- The gates judge at most the newest 12,000 characters (four windows). The Advisor gate sees the
  delta, not the user's earlier instruction, unless that instruction is in the delta.
