# Neurosymbolic faithfulness

Does symbolic (program-of-thought) reasoning actually move computation out of
the model's weights, or does it just relabel it?

## The question

Standard framing: "is PoT more faithful than CoT?" That is hard to operationalise,
because faithfulness is several different claims wearing one word.

Sharper framing, and the one this repo tests. **In PoT the model does not execute
the code — the interpreter does.** The answer is produced outside the forward
pass. So a linear probe on the model's own activations gives a clean read:

- If the eventual answer **cannot** be decoded from PoT activations before the
  program is finished, the model genuinely offloaded the computation. There is no
  internal answer for the code to be a post-hoc rationalisation *of*.
- If it **can** be decoded early and accurately, the model already knew the answer
  and emitted code as decoration. That is the interesting failure mode.

CoT is the comparison arm: same problems, same model, natural-language reasoning
that the model must carry out itself.

## The headline metric is `prompt_end`, not the trajectory

The probe reads the residual stream at the **last prompt token**, before a single
reasoning token exists. Decodability there is evidence the answer was fixed in
advance.

Decodability *during* generation is not evidence of anything on its own. Eighty
percent of the way through a CoT the intermediate results are literally in the
context window — a probe recovering the answer there is the reasoning working as
designed. The generation deciles are reported as context for the `prompt_end`
number, not as the result.

## Probe target

Answers are free-form (GSM8K integers, MATH LaTeX), so a k-way probe needs a
projection onto a small label set. Measured class balance on the gold answers:

| scheme | GSM8K | MATH |
|---|---|---|
| `first_token` | 10 classes, 29% majority | 41 classes, 26% majority |
| `parity` | 70/30 — unusable | 34% of answers undefined |
| `magnitude` | 48% majority | 29% undefined |

`first_token` is the default. Its GSM8K skew is Benford's law on leading digits;
29% is the baseline a probe has to clear. `parity` is not viable on either set.

Because CoT and PoT produce *different* answers, their label distributions and
majority baselines differ. **Compare `delta_majority` (accuracy − majority
baseline), never raw accuracy.**

## Pipeline

```bash
python scripts/01_build_dataset.py --dataset gsm8k --n 1000 --out runs/dev
python scripts/02_generate.py       --run runs/dev --backend vllm --model Qwen/Qwen2.5-7B-Instruct
python scripts/03_execute.py        --run runs/dev
python scripts/04_harvest_activations.py --run runs/dev --device cuda \
    --components resid_post attn_out mlp_out
python scripts/05_train_probes.py   --run runs/dev --scheme first_token --component resid_post
```

To look at one problem end to end — exact prompt, raw completion, how the answer
was recovered, and the token indices the harvester would probe:

```bash
# display what a run already produced (no model needed)
python scripts/inspect_one.py --run runs/dev --index 0

# or generate fresh for a single item
python scripts/inspect_one.py --dataset gsm8k --index 0 --generate \
    --backend hf --model Qwen/Qwen2.5-0.5B-Instruct
```

This is the fastest way to catch a misalignment before spending GPU hours, and
it is how both of the extraction bugs above were found.

Steps 2 and 4 need the GPU box. Steps 1, 3 and 5 run anywhere. Swap
`--backend mock` (no model) or `--backend hf` (small model, MPS/CPU) to exercise
the plumbing locally.

## Activations: TransformerLens

`04` replays each rollout through a `HookedTransformer` and stores one `.npy`
per (condition, component). Components:

| component | what it is |
|---|---|
| `resid_pre` / `resid_post` | the residual stream entering / leaving block L |
| `attn_out` | what attention wrote into the stream at block L |
| `mlp_out` | what the MLP wrote into the stream at block L |

Having the writers separately is the point: `resid_post` answers *"is the answer
present at layer L"*, while `attn_out` / `mlp_out` answer *"which component put
it there"*. Probe one at a time with `--component`.

Hooks slice to the readout positions **inside** the forward pass rather than
caching everything — on a 7B with a 1300-token sequence that is ~10 MB per
rollout instead of ~800 MB.

Two verified facts worth not rediscovering:

- `resid_post` at layer L is numerically identical to HF `hidden_states[L+1]`
  for every layer **except the last**, where HF applies the final RMSNorm before
  appending. Ours is the raw stream.
- `resid_pre + attn_out + mlp_out == resid_post` to fp16 rounding. That identity
  is the cheapest check that hook names and position indices line up.

`from_pretrained` applies TransformerLens weight processing (`fold_ln`,
`center_writing_weights`, `center_unembed`) by default. Centering makes the
residual stream mean-zero along `d_model`, which removes the large common
component in raw streams — distinct prompts correlate at ~0.93 without it — and
that helps a linear probe. It is applied identically to both conditions. Pass
`--no-fold-ln` if you need numbers comparable to raw HF instead.

**Do not tokenise with `model.to_tokens`.** It defaults to `prepend_bos=True`,
Qwen's chat template carries no BOS, and a prepended token shifts every readout
index by one — silently probing the wrong positions. `Harvester.encode` passes
explicit ids from the raw tokenizer to sidestep this.

## Things that will silently ruin the result

- **Prompt drift.** vLLM generates; HF harvests activations. If the prompt string
  differs by one character between the two, every readout index shifts and the
  probe reads the wrong tokens. The exact rendered string is stored on each
  rollout and replayed verbatim; `04` re-renders one prompt and hard-fails on
  mismatch. Chat templating lives in `nsf/prompts.py` and nowhere else — in
  particular vLLM's own `.chat()` helper is deliberately unused.
- **Qwen's implicit system prompt.** Its chat template injects
  "You are Qwen, created by Alibaba Cloud..." even when you pass only a user
  message. Identical across both conditions, so it does not confound the
  contrast, but it does shift every token index.
- **Accuracy confound.** The two conditions will not have equal accuracy, and both
  probe decodability and every faithfulness metric correlate with correctness, so
  report probe results conditioned on correct/incorrect rather than pooled. Do not
  assume the direction: on a 0.5B smoke run CoT beat PoT on GSM8K (46.7% vs 28.3%).
  Measure it on your model.
- **Leakage across samples.** With `n_samples > 1`, folds must split on
  `item_id`. `nsf/probe.py` uses `StratifiedGroupKFold` for this.
- **Asymmetric label attrition.** A CoT rollout is labelled if its `Answer:` line
  parses; a PoT rollout is labelled only if its program also *ran*. Failed
  executions drop out of the PoT probe entirely, so the two conditions are
  trained on differently-selected subsets. `05` prints `labelled=` per condition
  — if PoT attrition is large, report it, and consider restricting both arms to
  items where each condition produced an answer.
- **The two prompts differ.** `prompt_end` is the last token of a *different*
  prompt in each condition, because the instructions differ. That is inherent to
  the design — the question is "how much does the model already know at the start
  of a CoT run vs a PoT run" — but it means the contrast is between two settings,
  not between two readouts of one setting.
- **CoT ignores the output format.** The 0.5B smoke run emitted a literal
  `Answer:` line on 6/60 GSM8K items, writing "Therefore, the answer is: $4500"
  instead; PoT complied 59/60, because there the format lives inside a `print()`
  call the model copies rather than prose it must obey. Parsing strictly does not
  fix this — it silently narrows the CoT arm to format-compliant completions, a
  biased subset. `nsf/answers.extract_answer_lenient` falls back through
  `answer_line -> boxed -> phrase -> last_number` and records which tier fired.
  Restoring the dropped 90% moved measured CoT accuracy from 1.7% to 46.7%.
  **Always read the tier distribution `03` prints**: a CoT arm resting mostly on
  `last_number` is measuring the parser, not the model. PoT stdout is still
  parsed strictly — guessing at a number in stdout would invent an answer the
  program never produced.
- **PoT prompt wording decides whether the arm is even PoT.** A rollout with no
  code in it is CoT with a different preamble, and it will silently contaminate
  the symbolic condition. Measured on 60 GSM8K items (Qwen2.5-0.5B, greedy):
  `"Output only Python code."` -> 49/60 executed; `"Reason in code only."` ->
  29/60, with only half the completions containing `print()` at all; both clauses
  together -> 57/60. Check the `exec:` counter `03` prints before trusting any
  PoT result, and treat a low `ok` rate as a prompt bug rather than a model
  limitation.
- **Chance is not 1/k.** Skewed classes and grouped folds make the theoretical
  baseline wrong. The permutation null (labels shuffled at group level, same fold
  structure) is the reference. Use ≥50 permutations before believing a z-score.

## Executing model-generated code

`nsf/execute.py` runs each PoT program in a fresh subprocess, in a scratch cwd,
under a wall-clock timeout and best-effort CPU/memory/file-size rlimits. That
contains runaway loops and memory bombs, which is what models actually produce.
It is **not** a security sandbox — the code can still reach the network and the
filesystem. Run it on a machine you are willing to have execute arbitrary Python.

macOS refuses `RLIMIT_AS`, so limits are applied best-effort and the local
fallback is wall-clock only; all three limits apply on the Linux GPU box.

## Layout

```
nsf/prompts.py      CoT/PoT templates + the only chat-template call site
nsf/data.py         MATH + GSM8K -> one Item schema; brace-matched \boxed extraction
nsf/answers.py      extraction, LaTeX/numeric normalisation, equivalence, class schemes
nsf/execute.py      sandboxed execution; produces the PoT answer
nsf/generate.py     vLLM / HF / mock backends
nsf/activations.py  TransformerLens harvest; hook points + readout position arithmetic
nsf/probe.py        (position x layer) sweep, grouped CV, permutation null
```
