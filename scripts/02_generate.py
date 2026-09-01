"""Generate CoT and PoT rollouts.

    python scripts/02_generate.py --run runs/dev --backend vllm \
        --model Qwen/Qwen2.5-7B-Instruct --conditions cot pot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsf import data, generate, prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], default="vllm")
    ap.add_argument("--conditions", nargs="+", default=list(prompts.CONDITIONS))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap item count")
    args = ap.parse_args()

    run = Path(args.run)
    items = data.read_items(run / "raw" / "items.jsonl")
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items x {len(args.conditions)} conditions")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backend = generate.build_backend(args.backend, args.model)
    cfg = generate.SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        seed=args.seed,
    )

    for condition in args.conditions:
        rendered = [
            prompts.render(tokenizer, it.problem, condition, it.dataset) for it in items
        ]
        texts = [r.text for r in rendered]
        print(f"[{condition}] generating…")
        completions = backend.generate(texts, cfg)

        rollouts = [
            generate.Rollout(
                item_id=it.item_id,
                condition=condition,
                dataset=it.dataset,
                sample_idx=k,
                prompt_text=text,
                completion=comp,
            )
            for it, text, comps in zip(items, texts, completions)
            for k, comp in enumerate(comps)
        ]
        out = generate.write_rollouts(rollouts, run / "raw" / f"rollouts_{condition}.jsonl")
        print(f"[{condition}] wrote {len(rollouts)} -> {out}")


if __name__ == "__main__":
    main()
