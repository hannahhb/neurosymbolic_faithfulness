"""Run ONE problem through both conditions and print everything, readably.

The JSONL files are for the pipeline; this is for you.  It shows the exact
prompt the model saw, the raw completion, how the answer was recovered (parsed
for CoT, executed for PoT), and the token indices the harvester would probe --
which is the fastest way to catch a misalignment before spending GPU hours.

Display what a run already produced (no model needed):

    python scripts/inspect_one.py --run runs/e2e --index 0

Generate fresh for a single item:

    python scripts/inspect_one.py --dataset gsm8k --index 0 \\
        --generate --backend hf --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsf import activations, answers, data, execute, generate, prompts

W = 88


def rule(title: str = "", char: str = "=") -> None:
    if not title:
        print(char * W)
        return
    pad = W - len(title) - 3
    print(f"{char * 2} {title} {char * max(pad, 0)}")


def block(text: str, limit: int = 0, indent: str = "  ") -> None:
    body = text if not limit or len(text) <= limit else (
        text[: limit // 2] + f"\n{indent}[... {len(text) - limit} chars elided ...]\n"
        + text[-limit // 2 :]
    )
    for line in body.splitlines() or [""]:
        print(indent + line)


def pick_item(args) -> data.Item:
    if args.run:
        items = data.read_items(Path(args.run) / "raw" / "items.jsonl")
    elif args.dataset == "gsm8k":
        items = data.load_gsm8k(split=args.split)
    else:
        items = data.load_math(split=args.split, configs=args.subjects or data.MATH_CONFIGS)
    if args.item_id:
        for it in items:
            if it.item_id == args.item_id:
                return it
        raise SystemExit(f"item_id {args.item_id!r} not found among {len(items)} items")
    if not 0 <= args.index < len(items):
        raise SystemExit(f"--index {args.index} out of range (0..{len(items) - 1})")
    return items[args.index]


def existing_completion(run: Path, condition: str, item_id: str) -> str | None:
    path = run / "raw" / f"rollouts_{condition}.jsonl"
    if not path.exists():
        return None
    for r in generate.read_rollouts(path):
        if r.item_id == item_id and r.sample_idx == 0:
            return r.completion
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="read items/rollouts from this run directory")
    ap.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--subjects", nargs="+", choices=list(data.MATH_CONFIGS),
                    help="MATH only: restrict to these subject configs")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--item-id")
    ap.add_argument("--conditions", nargs="+", default=list(prompts.CONDITIONS))
    ap.add_argument("--generate", action="store_true",
                    help="generate a fresh completion instead of reading the run")
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-chars", type=int, default=600,
                    help="elide the middle of the prompt; 0 = show all")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    item = pick_item(args)

    rule("PROBLEM")
    print(f"  item_id : {item.item_id}")
    print(f"  dataset : {item.dataset}"
          + (f"   subject: {item.subject}   {item.level}" if item.level else ""))
    print(f"  gold    : {item.gold!r}   normalised: {answers.normalize(item.gold)!r}")
    print()
    for line in textwrap.wrap(item.problem, W - 4):
        print("  " + line)
    print()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backend = generate.build_backend(args.backend, args.model) if args.generate else None

    for condition in args.conditions:
        rendered = prompts.render(tokenizer, item.problem, condition, item.dataset)

        if args.generate:
            cfg = generate.SamplingConfig(
                temperature=args.temperature, max_tokens=args.max_tokens, n=1
            )
            completion = backend.generate([rendered.text], cfg)[0][0]
            source = f"generated ({args.backend}/{args.model})"
        else:
            completion = existing_completion(Path(args.run), condition, item.item_id) if args.run else None
            source = f"from {args.run}"
            if completion is None:
                rule(f"{condition.upper()}", "-")
                print(f"  no rollout found ({source}); pass --generate to make one\n")
                continue

        rule(f"{condition.upper()}  [{source}]", "-")

        print("\n  --- PROMPT (exact string fed to the model) ---")
        block(rendered.text, limit=args.prompt_chars)

        print("\n  --- RAW COMPLETION ---")
        block(completion)

        print("\n  --- ANSWER RECOVERY ---")
        if condition == "pot":
            code = execute.strip_fences(completion)
            print("  extracted code:")
            block(code or "<none>", indent="    | ")
            res = execute.run_completion(completion, timeout=args.timeout)
            print(f"\n  exec status : {res.status}   returncode={res.returncode}")
            print("  stdout:")
            block(res.stdout or "<empty>", indent="    | ")
            if res.stderr.strip():
                print("  stderr:")
                block(res.stderr.strip()[:800], indent="    | ")
            raw = res.answer
        else:
            raw = answers.extract_answer(completion)
            print(f"  parsed from the last `Answer:` line")

        norm = answers.normalize(raw)
        ok = answers.is_correct(raw, item.gold)
        print(f"\n  model answer : {raw!r}")
        print(f"  normalised   : {norm!r}")
        print(f"  gold         : {answers.normalize(item.gold)!r}")
        print(f"  CORRECT      : {ok}")
        print(f"  probe label  : {answers.answer_class(raw, 'first_token')!r} (first_token)")

        print("\n  --- HARVEST ALIGNMENT ---")
        p_ids = tokenizer(rendered.text, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        joint = tokenizer(rendered.text + completion, add_special_tokens=False)["input_ids"]
        spec = activations.HarvestSpec()
        idx = activations.readout_indices(len(p_ids), len(c_ids), spec)
        print(f"  prompt tokens={len(p_ids)}  completion tokens={len(c_ids)}  "
              f"seam clean={p_ids + c_ids == joint}")
        names = spec.position_names()
        print(f"  {names[0]} -> token {idx[0]} = "
              f"{tokenizer.decode([p_ids[-1]])!r} (last prompt token)")
        for name, i in list(zip(names, idx))[1:]:
            tok = (p_ids + c_ids)[i]
            print(f"  {name}    -> token {i} = {tokenizer.decode([tok])!r}")
        print()


if __name__ == "__main__":
    main()
