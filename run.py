#!/usr/bin/env python3
"""Two entry points.

  resample  counterfactual importance per step, for both arms
  probe     linear probes over step-boundary hidden states, for both arms

    python -m neurosymbolic_faithfulness.run resample --backend mock --n-problems 4
    python -m neurosymbolic_faithfulness.run probe --model <hf-id> --n-problems 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from neurosymbolic_faithfulness import prompts as P
from neurosymbolic_faithfulness.data import (MR_MODELS, MR_SPLITS, cot_reference_steps,
                                             load_gsm_symbolic, load_math_rollouts)
from neurosymbolic_faithfulness.resample import Embedder, extract_answer, resample_trace


def _engine(a):
    from neurosymbolic_faithfulness.engine import HFEngine, MockEngine
    if a.backend == "mock":
        return MockEngine(seed=a.seed)
    return HFEngine(model=a.model, batch_size=a.batch_size, seed=a.seed)


def _problems(a):
    if a.dataset == "gsm_symbolic":
        return load_gsm_symbolic(a.n_problems, seed=a.seed, config=a.gsm_config)
    return load_math_rollouts(a.mr_model, a.mr_split, limit=a.n_problems)


def cmd_resample(a) -> int:
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing to {out}")
    problems = _problems(a)
    engine = _engine(a)
    embedder = Embedder(a.embedder)
    print(f"[run] {len(problems)} problems | embedder={embedder.backend} "
          f"| backend={engine.name}")

    rows: list[dict] = []
    for p in problems:
        # The precomputed CoT metrics belong to the model that produced them.
        # Pairing them with a code arm from a different model would compare two
        # different models, not two conditions -- so only reuse on an exact match.
        reuse = (p.source == "math_rollouts" and not a.regenerate_cot
                 and a.model.split("/")[-1].lower() == a.mr_model.lower())
        if p.source == "math_rollouts" and not a.regenerate_cot and not reuse:
            print(f"  [guard] {a.model} != {a.mr_model}: regenerating the CoT arm "
                  f"rather than reusing precomputed metrics from another model")
        if reuse:
            rows.extend(cot_reference_steps(p))
            conditions = ["code"]
        else:
            conditions = list(P.CONDITIONS)

        for cond in conditions:
            base = P.build_prompt(engine.tokenizer, p.question, cond)
            trace = engine.generate([base], temperature=a.temperature,
                                    max_tokens=a.max_new_tokens,
                                    seeds=[a.seed], stop=[])[0].text
            print(f"  {p.problem_id} [{cond}] base answer="
                  f"{extract_answer(trace, cond)} gt={p.answer}")
            res = resample_trace(engine, embedder, p.question, cond, trace,
                                 p.answer, n_rollouts=a.n_rollouts,
                                 temperature=a.temperature,
                                 max_tokens=a.max_new_tokens, seed=a.seed,
                                 max_steps=a.max_steps,
                                 log=lambda *_: None)
            for r in res:
                rows.append({"problem_id": p.problem_id, "cluster_id": p.cluster_id,
                             "condition": cond, "source": "generated",
                             "base_trace": trace, **r.to_dict()})

    with open(out / "steps.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _summarise(rows)
    return 0


def cmd_probe(a) -> int:
    import numpy as np
    from transformers import AutoModelForCausalLM
    from neurosymbolic_faithfulness.probe import build_probe_dataset, train_probes

    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing to {out}")
    problems = _problems(a)
    engine = _engine(a)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype="auto", device_map="auto") if a.backend != "mock" else None
    if model is None:
        print("[run] probe needs a real model (--backend hf)")
        return 2

    all_rows = []
    for cond in P.CONDITIONS:
        X, ex = build_probe_dataset(engine, model, problems, cond,
                                    n_rollouts=a.n_rollouts_probe,
                                    temperature=a.temperature,
                                    max_tokens=a.max_new_tokens, seed=a.seed,
                                    label_mode=a.label_mode)
        np.save(out / f"hidden_{cond}.npy", X)
        (out / f"examples_{cond}.jsonl").write_text(
            "\n".join(json.dumps(e.to_dict()) for e in ex))
        rows = train_probes(X, ex, n_bins=a.n_bins, seed=a.seed)
        for r in rows:
            r["condition"] = cond
        all_rows.extend(rows)
        print(f"  {cond}: {X.shape[0]} boundaries, {X.shape[1]} layers")

    (out / "probe_auc.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_rows))
    best = {}
    for r in all_rows:
        k = (r["condition"], r["bin"])
        if k not in best or r["auc_mean"] > best[k]["auc_mean"]:
            best[k] = r
    print("\nbest-layer AUC by relative position")
    print(f"{'cond':<6}{'bin':>4}{'rel_pos':>14}{'AUC':>8}{'baseline':>10}{'layer':>7}")
    for (c, b), r in sorted(best.items()):
        span = f"[{r['rel_pos_lo']:.1f},{r['rel_pos_hi']:.1f})"
        print(f"{c:<6}{b:>4}{span:>14}{r['auc_mean']:>8.3f}"
              f"{r['majority_baseline']:>10.2f}{r['layer']:>7}")
    return 0


def _summarise(rows):
    import numpy as np
    print("\n" + "=" * 74)
    print(f"{'condition':<10}{'steps':>7}{'counterfactual KL':>22}{'resampling KL':>18}")
    print("-" * 74)
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        ck = [r["counterfactual_importance_kl"] for r in sub
              if r.get("counterfactual_importance_kl") == r.get("counterfactual_importance_kl")]
        rk = [r["resampling_importance_kl"] for r in sub
              if r.get("resampling_importance_kl") == r.get("resampling_importance_kl")]
        print(f"{cond:<10}{len(sub):>7}"
              f"{(np.mean(ck) if ck else float('nan')):>22.4f}"
              f"{(np.mean(rk) if rk else float('nan')):>18.4f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["resample", "probe"])
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--backend", default="hf", choices=["hf", "mock"])
    p.add_argument("--dataset", default="gsm_symbolic",
                   choices=["gsm_symbolic", "math_rollouts"])
    p.add_argument("--gsm-config", default="main")
    p.add_argument("--mr-model", default=MR_MODELS[0], choices=list(MR_MODELS))
    p.add_argument("--mr-split", default=MR_SPLITS[0], choices=list(MR_SPLITS))
    p.add_argument("--regenerate-cot", action="store_true",
                   help="ignore math_rollouts' precomputed CoT metrics")
    p.add_argument("--n-problems", type=int, default=10)
    p.add_argument("--n-rollouts", type=int, default=100)
    p.add_argument("--n-rollouts-probe", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--embedder", default="minilm", choices=["minilm", "hash"])
    p.add_argument("--label-mode", default="correct", choices=["correct", "modal"])
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/out")
    a = p.parse_args()
    t0 = time.time()
    rc = cmd_resample(a) if a.cmd == "resample" else cmd_probe(a)
    print(f"[run] done in {time.time() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
