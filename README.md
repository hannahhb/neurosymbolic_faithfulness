# Phase 0 — calibrating the tool-use decision

Part of the experiment asking whether a model's decision to call a tool is
**pre-encoded** at the end of the prompt (H_pre — the reasoning is post-hoc) or
**generated** during decoding (H_gen — the reasoning is causally upstream).

Phase 0 answers a prerequisite question: **is there any difficulty band where
the decision is genuinely uncertain?** If the model always calls or never calls,
there is no variance for the later variance decomposition or the prefix
cross-grafting to explain, and the project stops here.

## Setup

```bash
pip install -r requirements.txt
```

## Running

Smoke-test the whole pipeline on a laptop with the scripted mock backend
(no GPU, no model download — outputs are synthetic and stamped as such):

```bash
python run_phase0.py --backend mock --n-prompts 12 --out-dir runs/smoke_mock
```

The real run (1× A100 is enough for a 7B; ~10–20 min):

```bash
python run_phase0.py --model Qwen/Qwen2.5-7B-Instruct --out-dir runs/phase0_qwen7b
```

Parser and sandbox tests:

```bash
python test_parsing.py
```

The process exits `0` only if the gate passes **and** the malformed rate is
within budget; `2` otherwise, so it can be chained in a script.

## What it does

Six difficulty levels of `chain_sum`, 50 prompts each, two conditions:

| condition | tool in context | samples | temp | measures |
|---|---|---|---|---|
| `forced_no_tool` | no | 1 | 0.0 | unaided accuracy = **true tool necessity** |
| `free_choice` | yes | 8 | 0.7 | **tool-call rate** |

Note that `forced_no_tool` removes the tool from the context entirely — it is
not a prompt telling the model to abstain, which would confound refusal with
capability. If a model in that condition invents a tool call anyway, it is
recorded but **never executed**, so the condition stays unaided.

## Outputs

```
runs/<name>/
  config.json                        exact configuration of the run
  raw/items.jsonl                    every task instance + reasoning-gym metadata
  raw/rollouts_free_choice.jsonl     one record per rollout, full transcripts
  raw/rollouts_forced_no_tool.jsonl
  summary/per_level.csv              the numbers, for hand-checking
  summary/calibration_summary.json   same + gate verdict + malformed breakdown
  plots/calibration_curves.png       the two curves vs difficulty, with the gate band
  plots/accuracy_by_choice.png       does the tool actually help?
  plots/decision_variance.png        between-prompt vs within-prompt variance
  transcripts/handread_sample.md     30 transcripts stratified over level × label
  transcripts/marker_stats.json      regex pointers for the hand-read
  transcripts/forced_no_tool_sample.md
```

Every rollout record keeps the exact prompt string and its sha256 for every
round, so Phase 1/2 forward passes can replay the identical context.

## Design decisions

1. **Task parameters were read from the installed source, not guessed.**
   `chain_sum` and `products` both take `min_terms`/`max_terms`/`min_digits`/
   `max_digits`/`allow_negation`/`seed`/`size`. **`GALLERY.md` is stale for
   `products`** — it documents `min_factors`/`max_factors`, which the current
   `ProductsConfig` rejects. `tasks.py::self_check()` constructs every rung of
   every ladder, so a future rename fails loudly instead of silently.

2. **One prompt shape, two conditions.** `chat.py::build_prompt` is the only
   place prompts are assembled. The system prompt is `"You are a helpful
   assistant."` — nothing about when or whether to use the tool. The tool's
   description says what it does, never when to reach for it. The only
   difference between conditions is `tools=` being present or absent.

3. **`\boxed{}` instruction is in both conditions**, so it cannot bias the tool
   decision; it exists purely to make answer extraction reliable. Extraction
   method is logged per rollout (`boxed` / `answer_is` / `last_number` / `none`)
   so extraction failures can be audited rather than silently scored wrong.

4. **The scorer is handed an extracted answer, never raw prose.** Reasoning
   Gym's default `score_answer` gives partial credit for substring containment
   (`len(oracle)/len(answer)`), so passing a full transcript would inflate
   scores. Accuracy uses strict `score == 1.0`.

5. **Difficulty levels are pinned** (`min_terms == max_terms`, same for digits)
   so that within-level difficulty variance does not contaminate the curves.
   Each level gets its own dataset seed.

6. **Generation stops at `</tool_call>`.** Without this, models sometimes
   hallucinate the tool's *response* and answer from an invented number, which
   would silently corrupt the TOOL condition.

7. **8 samples per prompt are not 8 independent observations.** Every rate is
   reported twice: pooled over rollouts, and as a mean over per-prompt rates
   with a bootstrap CI over prompts. **The gate is evaluated on the item-mean
   rate**, which is the honest one. `decision_variance.png` shows how much of
   the variance is between prompts vs within a prompt — if nearly all prompts
   are unanimous across their 8 samples, the decision is already
   prompt-determined, which is itself evidence bearing on H_pre.

8. **The sandbox never calls `eval`.** `calculator.py` parses with
   `ast.parse(mode="eval")` and walks the tree, refusing every node type that
   is not pure arithmetic — names, calls, attributes, subscripts, comprehensions
   and string constants are all rejected, with bounds on expression length,
   exponent size and operand width.

9. **Model choice is an assumption, not a finding.** The default is
   `Qwen/Qwen2.5-7B-Instruct`: it fits the GPU budget and has a native
   Hermes-style tool-calling template. Any Hermes-format model drops in
   unchanged; for Llama 3.1 pass `--tool-format llama31`. If a different model
   is intended, change `--model` and confirm `--tool-format` matches its
   template.

## Parsing categories

A rollout is `TOOL` if any well-formed call appears, `MALFORMED` if only
malformed attempts appear, else `NO_TOOL`. Malformed attempts are recorded with
a reason (bad JSON, wrong tool name, missing `expression`, call-shaped JSON
outside the tags, plain-text `calculator(...)`, unclosed tag).

Two distinctions worth keeping straight:

* A call that is well-formed but whose *expression* fails to evaluate is a
  `tool_error`, not a malformed call — tracked separately.
* An unclosed `<tool_call>` where generation hit `max_tokens` is `truncated`,
  not malformed — that is a budget problem, not a model failure. Raise
  `--max-new-tokens` if `truncated_call_rate` is non-trivial.

**The malformed rate is reported first and gates everything else.** If
`any_malformed_rate` exceeds 0.05, the run reports over-budget and the parsing
must be fixed before the calibration numbers mean anything.

## The gate

Phase 0 passes only if some level's item-mean tool-call rate lands in
`[0.25, 0.75]`. Three failure modes, each with a different remedy:

| failure | remedy |
|---|---|
| never calls (all rates < 0.25) | try `--dataset products` — multiplication gets hard far faster per digit than addition |
| always calls (all rates > 0.75) | add easier rungs, e.g. `--custom-levels "2x1,2x2,3x2,3x3"` |
| rate jumps across the band between adjacent rungs | refine between them, e.g. `--custom-levels "4x3,4x4,5x4,5x5"` |

If none of these produce a band, **stop and report it**. The rest of the
experiment has nothing to explain.

## The hand-read (do not skip)

`transcripts/handread_sample.md` contains 30 rollouts stratified over
level × label. Each shows a **decision prefix** — everything the model emitted
before its first call attempt. The question to answer by reading, not by regex:

> Does the model explicitly deliberate ("this is too big to do in my head"),
> or does it simply start calling?

This determines whether "the formulation" — the reasoning that supposedly
justifies the decision — is a well-defined object at all. If the mean decision
prefix is a handful of characters and most rollouts open straight into
`<tool_call>`, there is no verbal deliberation to be post-hoc *about*, and the
framing of Phase 1/2 needs revisiting before running them.

`marker_stats.json` gives regex hit-rates for deliberation phrases split by
label. It is a **pointer to where to look, not a result** — the write-up should
quote transcripts a human actually read.

## Known caveats

* `products` questions end with *"Give only the result as your final answer."*
  while `chain_sum` questions do not. That instruction discourages showing work
  and may suppress exactly the deliberation text the hand-read is looking for.
  If the ladder is switched to `products`, re-read transcripts before comparing
  decision prefixes across datasets.
* `accuracy_by_choice.png` is observational: rollouts were not randomised into
  TOOL/NO_TOOL, so the gap between the two curves is confounded by which
  problems the model chose the tool for. It is a sanity check that the tool
  helps, not an effect estimate.
* Levels differ in both term count and digit width, so the x-axis is an ordinal
  ladder, not a linear difficulty scale. `work = terms × digits` is logged per
  level as a scalar proxy if a continuous axis is wanted.
