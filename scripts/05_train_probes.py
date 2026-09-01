"""Train probes over the (position x layer) grid and write the results table.

    python scripts/05_train_probes.py --run runs/dev --scheme first_token

The headline row is position=prompt_end: decodability of the eventual answer
before any reasoning token exists.  Compare `delta_majority` across conditions,
never raw accuracy -- CoT and PoT answer differently, so their label
distributions and majority baselines differ.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nsf import activations, answers, probe


def load_labels(run: Path, condition: str, scheme: str) -> dict[str, str | None]:
    path = run / "raw" / f"answers_{condition}.jsonl"
    out: dict[str, str | None] = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['item_id']}::{row['sample_idx']}"
            out[key] = answers.answer_class(row["model_answer"], scheme)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conditions", nargs="+", default=["cot", "pot"])
    ap.add_argument("--scheme", default="first_token",
                    choices=["first_token", "parity", "magnitude"])
    ap.add_argument("--positions", nargs="+", default=None,
                    help="subset of position names; default all")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--max-dim", type=int, default=0, help="PCA dims; 0 = raw activations")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=20)
    ap.add_argument("--min-class-count", type=int, default=10)
    ap.add_argument("--min-rows", type=int, default=50,
                    help="refuse to fit below this many usable rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    all_rows = []

    for condition in args.conditions:
        acts_path = run / "acts" / f"acts_{condition}.npy"
        if not acts_path.exists():
            print(f"[{condition}] no activations at {acts_path}, skipping")
            continue
        acts, meta = activations.load(acts_path)
        labels_by_row = load_labels(run, condition, args.scheme)

        row_ids = meta["row_ids"]
        labels = [labels_by_row.get(rid) for rid in row_ids]
        groups = [rid.split("::")[0] for rid in row_ids]

        positions = meta["positions"]
        layers = meta["layers"]
        if args.positions:
            keep_pos = [i for i, p in enumerate(positions) if p in args.positions]
            acts = acts[:, keep_pos, :, :]
            positions = [positions[i] for i in keep_pos]

        n_labelled = sum(l is not None for l in labels)
        print(f"[{condition}] n={len(row_ids)} labelled={n_labelled} "
              f"positions={len(positions)} layers={len(layers)}")

        results = probe.sweep(
            acts,
            labels,
            groups,
            positions,
            layers,
            min_class_count=args.min_class_count,
            min_rows=args.min_rows,
            C=args.C,
            max_dim=args.max_dim or None,
            n_splits=args.n_splits,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        for r in results:
            d = r.to_dict()
            d["condition"] = condition
            d["scheme"] = args.scheme
            all_rows.append(d)

        best = max(results, key=lambda r: r.delta_majority)
        pe = [r for r in results if r.position == activations.PROMPT_END]
        print(f"[{condition}] best overall: {best.position} L{best.layer} "
              f"acc={best.accuracy:.3f} (+{best.delta_majority:.3f} over majority)")
        if pe:
            best_pe = max(pe, key=lambda r: r.delta_majority)
            print(f"[{condition}] PROMPT_END best: L{best_pe.layer} "
                  f"acc={best_pe.accuracy:.3f} maj={best_pe.majority:.3f} "
                  f"delta={best_pe.delta_majority:+.3f} z={best_pe.z_vs_null:.2f}")

    if not all_rows:
        raise SystemExit("no results; run 04_harvest_activations.py first")

    out = run / "summary" / f"probe_{args.scheme}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
