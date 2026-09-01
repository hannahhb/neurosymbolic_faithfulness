"""Replay rollouts through TransformerLens and store activation readouts.

    python scripts/04_harvest_activations.py --run runs/dev \
        --model Qwen/Qwen2.5-7B-Instruct --device cuda \
        --components resid_post attn_out mlp_out

One .npy per (condition, component), so probes can be fit per component and the
"is the answer there" question separated from "which component put it there".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nsf import activations, data, generate, prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--conditions", nargs="+", default=["cot", "pot"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n-deciles", type=int, default=10)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--components", nargs="+", default=["resid_post"],
                    choices=sorted(activations.COMPONENTS),
                    help="resid_post is the stream itself; attn_out/mlp_out are "
                         "what each sublayer writes into it")
    ap.add_argument("--no-fold-ln", action="store_true",
                    help="disable TransformerLens weight processing (fold_ln + "
                         "centering). Off by default because centering removes "
                         "the large common component in raw residual streams.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    spec = activations.HarvestSpec(
        n_deciles=args.n_deciles,
        layer_stride=args.layer_stride,
        components=tuple(args.components),
    )
    process = not args.no_fold_ln
    harvester = activations.Harvester(
        args.model, device=args.device, dtype=args.dtype, max_len=args.max_len,
        fold_ln=process, center_writing_weights=process, center_unembed=process,
    )
    layers = harvester.kept_layers(spec)
    positions = spec.position_names()
    print(f"device={harvester.device} layers={len(layers)} positions={len(positions)} "
          f"hidden={harvester.hidden} components={list(spec.components)} "
          f"weight_processing={process}")

    for condition in args.conditions:
        src = run / "raw" / f"rollouts_{condition}.jsonl"
        if not src.exists():
            print(f"[{condition}] missing {src}, skipping")
            continue
        rollouts = generate.read_rollouts(src)
        if args.limit:
            rollouts = rollouts[: args.limit]

        # The prompt must be byte-identical to generation time.  Re-render one
        # and compare rather than trusting that nothing drifted.
        _verify_prompt_roundtrip(harvester, run, rollouts[0], condition)

        rows: dict[str, list] = {c: [] for c in spec.components}
        row_ids, dropped = [], {"empty": 0, "too_long": 0, "dirty_seam": 0}
        for i, r in enumerate(rollouts):
            if not harvester.seam_is_clean(r.prompt_text, r.completion):
                dropped["dirty_seam"] += 1
                continue
            got = harvester.harvest_one(r.prompt_text, r.completion, spec)
            if got is None:
                _, n_p, n_c = harvester.encode(r.prompt_text, r.completion)
                dropped["empty" if n_c < 1 else "too_long"] += 1
                continue
            for comp, arr in got.items():
                rows[comp].append(arr)
            row_ids.append(f"{r.item_id}::{r.sample_idx}")
            if (i + 1) % 100 == 0:
                print(f"[{condition}] {i + 1}/{len(rollouts)}", flush=True)

        if not row_ids:
            print(f"[{condition}] nothing harvested; dropped={dropped}")
            continue

        print(f"[{condition}] kept {len(row_ids)}/{len(rollouts)} dropped={dropped}")
        for comp in spec.components:
            acts = np.stack(rows[comp])  # [N, P, L, H] float16
            out = activations.save(
                run / "acts" / f"acts_{condition}__{comp}",
                acts,
                row_ids,
                positions,
                layers,
                meta={
                    "model": args.model,
                    "condition": condition,
                    "component": comp,
                    "dropped": dropped,
                    "n_deciles": args.n_deciles,
                    "layer_stride": args.layer_stride,
                    "weight_processing": process,
                },
            )
            print(f"[{condition}/{comp}] {acts.shape} "
                  f"({acts.nbytes / 1024 ** 3:.2f} GB) -> {out}")


def _verify_prompt_roundtrip(harvester, run: Path, rollout, condition: str) -> None:
    """Fail loudly if the stored prompt no longer matches a fresh render.

    Generation and this pass must see the identical string.  If prompts.py or
    the tokenizer's chat template changed since the rollouts were produced,
    every readout index shifts and the probe silently reads the wrong tokens.
    """
    items = {it.item_id: it for it in data.read_items(run / "raw" / "items.jsonl")}
    item = items.get(rollout.item_id)
    if item is None:
        print(f"WARNING: {rollout.item_id} not in items.jsonl; cannot verify prompt.",
              file=sys.stderr)
        return
    fresh = prompts.render(harvester.tokenizer, item.problem, condition, item.dataset).text
    if fresh != rollout.prompt_text:
        stored = rollout.prompt_text
        # Show where they actually diverge.  Printing the tails is useless: a
        # prompt edit is almost always mid-instruction, and both tails end with
        # the identical chat-template suffix.
        i = next((k for k in range(min(len(stored), len(fresh)))
                  if stored[k] != fresh[k]), min(len(stored), len(fresh)))
        lo, hi = max(0, i - 40), i + 60
        raise SystemExit(
            "Stored prompt does not match a fresh render -- prompts.py or the chat "
            "template changed since generation. Re-generate the rollouts, or the "
            "readout positions will be misaligned.\n"
            f"  condition   : {condition}\n"
            f"  first differs at char {i} of {len(stored)} (stored) / {len(fresh)} (fresh)\n"
            f"  stored: ...{stored[lo:hi]!r}...\n"
            f"  fresh : ...{fresh[lo:hi]!r}..."
        )


if __name__ == "__main__":
    main()
