#!/usr/bin/env python3
"""Run the 2x2 (mode x bias) on gsm_symbolic.

    python -m neurosymbolic_faithfulness.faith.run --pilot --out-dir runs/pilot
    python -m neurosymbolic_faithfulness.faith.run --n-items 300 --out-dir runs/full

--pilot runs the day-1 gate only: NL, unbiased vs wrong-hint, small n.  Its
question is "does the bias move answers at all?".  If it does not, there is no
effect to compare and nothing downstream is worth building.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_PKG_PARENT = str(Path(__file__).resolve().parent.parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from neurosymbolic_faithfulness.faith import prompts as P
from neurosymbolic_faithfulness.faith.data import build_items


def _seed(base: int, item_id: str, mode: str, bias: str, k: int) -> int:
    h = hashlib.sha256(f"{base}|{item_id}|{mode}|{bias}|{k}".encode()).hexdigest()
    return int(h[:8], 16)


def run_cells(engine, items, cells, n_samples, temperature, max_tokens,
              seed, log=print) -> list[dict]:
    records: list[dict] = []
    for mode, bias in cells:
        specs = [(it, k) for it in items for k in range(n_samples)]
        prompts = [P.build_prompt(engine.tokenizer, it, mode, bias) for it, _ in specs]
        seeds = [_seed(seed, it.item_id, mode, bias, k) for it, k in specs]
        log(f"  [{mode}/{bias}] {len(prompts)} rollouts")
        outs = engine.generate(prompts, temperature=temperature,
                               max_tokens=max_tokens, seeds=seeds, stop=[])
        for (it, k), prompt, out in zip(specs, prompts, outs):
            ex = P.extract(out.text, mode)
            records.append({
                "item_id": it.item_id, "template_id": it.template_id,
                "mode": mode, "bias": bias, "sample_idx": k,
                "question": it.question,
                "gt_answer": it.answer, "distractor": it.distractor,
                "distractor_kind": it.distractor_kind,
                "prompt": prompt, "completion": out.text,
                "finish_reason": out.finish_reason, "n_tokens": out.n_tokens,
                **ex.to_dict(),
                "correct": P.answers_match(ex.answer, it.answer),
                "equals_distractor": P.answers_match(ex.answer, it.distractor),
            })
    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--backend", default="hf", choices=["hf", "mock"])
    p.add_argument("--n-items", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--hf-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--pilot", action="store_true",
                   help="day-1 gate: NL only, none vs wrong, small n")
    p.add_argument("--out-dir", default="runs/faith")
    a = p.parse_args()

    if a.pilot:
        cells = [("nl", "none"), ("nl", "wrong")]
        a.n_items = min(a.n_items, 30)
    else:
        cells = [(m, b) for m in P.MODES for b in ("none", "wrong", "correct", "contentless")]

    out = Path(a.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "raw" / "config.json").write_text(json.dumps(vars(a), indent=2))

    items, _ = build_items(a.n_items, a.seed)
    (out / "raw" / "items.jsonl").write_text(
        "\n".join(json.dumps(i.to_dict()) for i in items))
    print(f"[faith] {len(items)} items x {len(cells)} cells x {a.n_samples} samples")

    from neurosymbolic_faithfulness.engine import HFEngine, MockEngine
    engine = MockEngine(seed=a.seed) if a.backend == "mock" else HFEngine(
        model=a.model, batch_size=a.hf_batch_size, seed=a.seed)

    t0 = time.time()
    records = run_cells(engine, items, cells, a.n_samples, a.temperature,
                        a.max_new_tokens, a.seed)
    print(f"[faith] generation finished in {time.time() - t0:.1f}s")
    with open(out / "raw" / "rollouts.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from neurosymbolic_faithfulness.faith.analyze import analyse, report
    rep = analyse(records)
    (out / "summary.json").write_text(json.dumps(rep, indent=2))
    report(rep, pilot=a.pilot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
