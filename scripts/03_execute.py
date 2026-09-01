"""Resolve each rollout to a final answer and a correctness label.

CoT: parse the `Answer:` line out of the completion.
PoT: execute the program and parse the `Answer:` line out of its stdout.  The
model never emits the answer itself in this condition, so this step is what
creates the PoT label.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsf import answers, data, execute, generate


def _resolve(payload: tuple[str, str, float]) -> dict:
    condition, completion, timeout = payload
    if condition == "pot":
        # A program's stdout is parsed strictly: it printed the format or it did
        # not, and guessing at a number in stdout would invent an answer the
        # program never produced.
        res = execute.run_completion(completion, timeout=timeout)
        return {"answer": res.answer, "exec_status": res.status,
                "stderr": res.stderr[:500], "tier": "stdout"}
    # CoT prose gets the tiered fallback; parsing strictly would restrict this
    # arm to format-compliant completions, which is a biased subset.
    value, tier = answers.extract_answer_lenient(completion)
    return {"answer": value, "exec_status": "n/a", "stderr": "", "tier": tier}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conditions", nargs="+", default=["cot", "pot"])
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    run = Path(args.run)
    gold = {it.item_id: it.gold for it in data.read_items(run / "raw" / "items.jsonl")}

    for condition in args.conditions:
        src = run / "raw" / f"rollouts_{condition}.jsonl"
        if not src.exists():
            print(f"[{condition}] no rollouts at {src}, skipping")
            continue
        rollouts = generate.read_rollouts(src)
        payloads = [(condition, r.completion, args.timeout) for r in rollouts]

        if condition == "pot" and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                resolved = list(pool.map(_resolve, payloads, chunksize=4))
        else:
            resolved = [_resolve(p) for p in payloads]

        rows = []
        for r, res in zip(rollouts, resolved):
            rows.append(
                {
                    "item_id": r.item_id,
                    "condition": condition,
                    "dataset": r.dataset,
                    "sample_idx": r.sample_idx,
                    "model_answer": res["answer"],
                    "model_answer_norm": answers.normalize(res["answer"]),
                    "gold": gold.get(r.item_id),
                    "correct": answers.is_correct(res["answer"], gold.get(r.item_id)),
                    "exec_status": res["exec_status"],
                    "extraction_tier": res["tier"],
                    "stderr": res["stderr"],
                }
            )

        out = run / "raw" / f"answers_{condition}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n if n else 0.0
        parsed = sum(r["model_answer"] is not None for r in rows) / n if n else 0.0
        print(f"[{condition}] n={n}  accuracy={acc:.2%}  answer_recovered={parsed:.2%}")
        if condition == "pot":
            print("           exec:", dict(collections.Counter(r["exec_status"] for r in rows)))
        else:
            tiers = collections.Counter(r["extraction_tier"] for r in rows)
            print("           extraction:", dict(tiers.most_common()))
            weak = (tiers["phrase"] + tiers["last_number"]) / n if n else 0.0
            if weak > 0.25:
                print(f"           WARNING: {weak:.0%} of CoT answers came from fallback "
                      f"tiers, not the `Answer:` line. The model is ignoring the output "
                      f"format; check a few completions before trusting this arm.")
        print(f"[{condition}] wrote -> {out}")


if __name__ == "__main__":
    main()
