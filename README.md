# Is code generation more faithful than chain-of-thought?

Two conditions, held identical except for the instruction:

* **cot**  — `"Let's think step by step"` (after the problem)
* **code** — `"write a python program for the following problem"` (before it)

Two measurements applied to both, which must agree:

* **Counterfactual importance** (causal, black-box) — resample each step of the
  trace and measure how much the final-answer distribution moves. Method from
  *Thought Anchors* (Bogdan, Macar, Nanda & Conmy, arXiv:2506.19143).
* **Linear probes** (correlational) — decode the eventual answer from the
  residual stream at each step boundary. If it is decodable early, the visible
  trace is post-hoc.

## Data

| source | role |
|---|---|
| `apple/GSM-Symbolic` | 100 templates x 50 instances. Both arms generated here; `original_id` is the clustering unit. |
| `uzaymacar/math-rollouts` | Thought Anchors companion data. **Its `chunks_labeled.json` already contains `resampling_importance_kl`, `counterfactual_importance_kl`, `forced_importance_kl`, 8-category `function_tags`, `depends_on`, `overdeterminedness`** for R1-Distill-Qwen-14B and Llama-8B on MATH. |

The CoT arm on math-rollouts problems is therefore free — *provided you use the
same model*. `run.py` uses the precomputed metrics by default; pass
`--regenerate-cot` to ignore them.

## Running

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python test_faithfulness.py                      # 21 tests, no GPU
python -m neurosymbolic_faithfulness.run resample --backend mock --embedder hash --n-problems 3
```

Real runs:

```bash
python -m neurosymbolic_faithfulness.run resample --model <hf-id> --dataset gsm_symbolic --n-problems 10 --n-rollouts 100
python -m neurosymbolic_faithfulness.run probe    --model <hf-id> --dataset gsm_symbolic --n-problems 20
```

## Files

| file | role |
|---|---|
| `prompts.py` | the two prompts; `prefix=` re-enters a trace mid-way |
| `segment.py` | sentences (cot) / statements (code), byte-exact offsets |
| `data.py` | both datasets; `cot_reference_steps` flattens precomputed metrics |
| `resample.py` | counterfactual importance, KL, MiniLM similarity filter |
| `probe.py` | hidden-state capture, grouped-CV logistic probes |
| `execute.py` | sandboxed execution of generated programs |
| `engine.py` | HF / vLLM / mock backends |
| `run.py` | `resample` and `probe` entry points |

## Method notes

* Replacement acceptance follows the paper: all-MiniLM-L6-v2, cosine **< 0.8**
  counts as semantically different. `counterfactual_importance_kl` uses only
  those; `resampling_importance_kl` pools all resamples.
* KL uses additive smoothing so disjoint supports stay finite.
* `n_unreadable` / `unreadable_fraction` are recorded per step. Answer
  distributions are computed over parseable rollouts only, so a step with a high
  unreadable fraction is reporting on a biased subset — check it before trusting
  that row.
* Probe CV is grouped by problem. Rollouts of one problem share a prompt and
  would leak across a random split. A majority-class baseline accompanies every
  AUC, since a problem whose rollouts agree gives a trivially high one.

## Model

`Qwen/Qwen2.5-7B-Instruct` (the default). It is not a reasoning model, so the
prompt genuinely controls the output format and the cot/code contrast is clean.
An R1-Distill would emit a natural-language `<think>` block in *both* arms,
making them near-identical.

The cost is that math-rollouts' precomputed CoT metrics come from
R1-Distill-Qwen-14B / Llama-8B and **cannot** be paired with a Qwen2.5-7B code
arm -- that would compare two models, not two conditions. `run.py` refuses the
mismatch and regenerates the CoT arm instead, printing a `[guard]` line. So both
arms are generated here, and math-rollouts serves as a MATH problem source and as
a reference implementation to sanity-check our numbers against.

Answer extraction deserves attention on the cot side: `"Let's think step by
step"` carries no answer-format instruction, so extraction falls back through
boxed -> "answer is N" -> trailing `= N` -> last number. The rule that fired is
recorded per step in `answer_rules`. **A run dominated by `last_number` is not
trustworthy** -- the last number in a trace is often an intermediate, which
would corrupt every KL. Check that column before reading any result.

## Compute

100 rollouts x ~15 steps x 2 arms x 10 problems is ~30k generations of ~512
tokens. `--n-rollouts 30` cuts it substantially at some cost in KL precision;
start there and check whether the importance ranking is stable before paying for
100.

## Suggested first run

```bash
python -m neurosymbolic_faithfulness.run resample \
  --model Qwen/Qwen2.5-7B-Instruct --dataset gsm_symbolic \
  --n-problems 5 --n-rollouts 30 --max-steps 8 --out-dir runs/pilot
```

Before scaling up, check three things in `steps.jsonl`: `answer_rules` is mostly
`boxed`/`answer_is`/`executed` rather than `last_number`; `unreadable_fraction`
is low; and `different_trajectories_fraction` is neither ~0 (the similarity
filter is rejecting everything, so counterfactual KL is undefined) nor ~1 (it is
accepting everything, so the filter is doing no work).
