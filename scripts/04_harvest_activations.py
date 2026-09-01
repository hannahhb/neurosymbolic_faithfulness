"""Replay rollouts through HF transformers and store residual-stream readouts.

    python scripts/04_harvest_activations.py --run runs/dev \
        --model Qwen/Qwen2.5-7B-Instruct --device cuda --layer-stride 1
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
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    spec = activations.HarvestSpec(n_deciles=args.n_deciles, layer_stride=args.layer_stride)
    harvester = activations.Harvester(
        args.model, device=args.device, dtype=args.dtype, max_len=args.max_len
    )
    layers = harvester.kept_layers(spec)
    positions = spec.position_names()
    print(f"device={harvester.device} layers={len(layers)} positions={len(positions)} "
          f"hidden={harvester.hidden}")

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

        rows, row_ids, dropped = [], [], {"empty": 0, "too_long": 0, "dirty_seam": 0}
        for i, r in enumerate(rollouts):
            if not harvester.seam_is_clean(r.prompt_text, r.completion):
                dropped["dirty_seam"] += 1
                continue
            arr = harvester.harvest_one(r.prompt_text, r.completion, spec)
            if arr is None:
                _, n_p, n_c = harvester.encode(r.prompt_text, r.completion)
                dropped["empty" if n_c < 1 else "too_long"] += 1
                continue
            rows.append(arr)
            row_ids.append(f"{r.item_id}::{r.sample_idx}")
            if (i + 1) % 100 == 0:
                print(f"[{condition}] {i + 1}/{len(rollouts)}", flush=True)

        if not rows:
            print(f"[{condition}] nothing harvested; dropped={dropped}")
            continue

        acts = np.stack(rows)  # [N, P, L, H] float16
        out = activations.save(
            run / "acts" / f"acts_{condition}",
            acts,
            row_ids,
            positions,
            layers,
            meta={
                "model": args.model,
                "condition": condition,
                "dropped": dropped,
                "n_deciles": args.n_deciles,
                "layer_stride": args.layer_stride,
            },
        )
        gb = acts.nbytes / 1024 ** 3
        print(f"[{condition}] kept {len(rows)}/{len(rollouts)} dropped={dropped}")
        print(f"[{condition}] {acts.shape} ({gb:.2f} GB) -> {out}")


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
        raise SystemExit(
            "Stored prompt does not match a fresh render -- prompts.py or the chat "
            "template changed since generation. Re-generate, or the readout "
            "positions will be misaligned.\n"
            f"  stored tail: {rollout.prompt_text[-80:]!r}\n"
            f"  fresh  tail: {fresh[-80:]!r}"
        )


if __name__ == "__main__":
    main()
