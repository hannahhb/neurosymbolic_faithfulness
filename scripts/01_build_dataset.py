"""Build the item pool for a run.

    python scripts/01_build_dataset.py --dataset gsm8k --n 1000 --out runs/dev
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsf import answers, data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "math"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subjects", nargs="+", choices=list(data.MATH_CONFIGS),
                    help="MATH only: restrict to these subject configs")
    ap.add_argument("--n", type=int, default=1000, help="0 = keep everything")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="run directory")
    args = ap.parse_args()

    if args.dataset == "gsm8k":
        pool = data.load_gsm8k(split=args.split)
        scope = args.dataset
    else:
        configs = args.subjects or data.MATH_CONFIGS
        pool = data.load_math(split=args.split, configs=configs)
        scope = f"{args.dataset}[{','.join(configs)}]"
    print(f"loaded {len(pool)} items from {scope}/{args.split}")

    items = pool if args.n == 0 else data.stratified_sample(pool, args.n, seed=args.seed)
    out = data.write_items(items, Path(args.out) / "raw" / "items.jsonl")
    print(f"wrote {len(items)} -> {out}")

    # Report the probe label distribution now, not after a GPU run.  A first
    # token class that is 60% one value makes probe accuracy uninterpretable.
    for scheme in ("first_token", "parity", "magnitude"):
        counts = collections.Counter(answers.answer_class(it.gold, scheme) for it in items)
        total = sum(counts.values())
        top = counts.most_common(6)
        majority = top[0][1] / total if total else 0.0
        print(f"  {scheme:11} classes={len(counts):3d}  majority={majority:.2%}  top={top}")

    if args.dataset == "math":
        lv = collections.Counter(it.level for it in items)
        print("  levels:", dict(sorted(lv.items(), key=lambda kv: str(kv[0]))))


if __name__ == "__main__":
    main()
