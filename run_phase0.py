#!/usr/bin/env python3
"""Phase 0 -- calibration.

Finds the difficulty band where the tool-use decision is genuinely uncertain,
and refuses to declare success if no such band exists.

Two conditions per difficulty level:
  forced_no_tool : the tool is absent from the context.  1 sample, temp 0.
                   -> the model's unaided accuracy = true tool necessity.
  free_choice    : the tool is available.  8 samples, temp 0.7.
                   -> the tool-call rate.

Everything is written to disk: raw transcripts as JSONL, per-level metrics as
CSV and JSON, curves as PNG, and a stratified sample of transcripts as Markdown
for hand-reading.

    python run_phase0.py --backend mock --n-prompts 8 --out-dir runs/smoke
    python run_phase0.py --model Qwen/Qwen2.5-7B-Instruct --out-dir runs/phase0_qwen7b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `python run_phase0.py` work from inside the package directory.  Python
# puts only the *script's own* directory on sys.path, so `neurosymbolic_faithfulness.*`
# would not resolve.  Putting the parent on the path fixes that; running as
# `python -m neurosymbolic_faithfulness.run_phase0` from the parent also works.
import sys as _sys
from pathlib import Path as _Path

_PKG_PARENT = str(_Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in _sys.path:
    _sys.path.insert(0, _PKG_PARENT)


from neurosymbolic_faithfulness.analyze import analyse, read_jsonl, write_csv
from neurosymbolic_faithfulness.config import Config
from neurosymbolic_faithfulness.engine import make_engine
from neurosymbolic_faithfulness.rollouts import run_condition
from neurosymbolic_faithfulness.tasks import LADDERS, build_items, parse_custom_levels


def parse_args() -> Config:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=d.model)
    p.add_argument("--tool-format", default=d.tool_format, choices=["hermes", "llama31"])
    p.add_argument("--backend", default=d.backend, choices=["vllm", "mock"])
    p.add_argument("--tensor-parallel-size", type=int, default=d.tensor_parallel_size)
    p.add_argument("--max-model-len", type=int, default=d.max_model_len)
    p.add_argument("--gpu-memory-utilization", type=float, default=d.gpu_memory_utilization)
    p.add_argument("--dtype", default=d.dtype)
    p.add_argument("--dataset", default=d.dataset, choices=sorted(LADDERS))
    p.add_argument("--n-prompts", type=int, default=d.n_prompts)
    p.add_argument("--levels", type=int, nargs="+", default=d.levels)
    p.add_argument("--custom-levels", default=d.custom_levels,
                   help='ad-hoc ladder, e.g. "4x3,4x4,5x4,5x5" (terms x digits); '
                        "use this to refine the ladder when the gate fails")
    p.add_argument("--free-n-samples", type=int, default=d.free_n_samples)
    p.add_argument("--free-temperature", type=float, default=d.free_temperature)
    p.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--max-tool-rounds", type=int, default=d.max_tool_rounds)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out-dir", default=d.out_dir)
    a = p.parse_args()
    return Config(
        model=a.model, tool_format=a.tool_format, backend=a.backend,
        tensor_parallel_size=a.tensor_parallel_size, max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization, dtype=a.dtype,
        dataset=a.dataset, n_prompts=a.n_prompts, levels=a.levels,
        custom_levels=a.custom_levels,
        free_n_samples=a.free_n_samples, free_temperature=a.free_temperature,
        max_new_tokens=a.max_new_tokens, max_tool_rounds=a.max_tool_rounds,
        seed=a.seed, out_dir=a.out_dir,
    )


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    cfg = parse_args()
    out = Path(cfg.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")

    if cfg.custom_levels:
        levels = parse_custom_levels(cfg.custom_levels)
        print(f"[phase0] custom ladder: {[lv.name for lv in levels]}")
    else:
        levels = [lv for lv in LADDERS[cfg.dataset] if lv.index in cfg.levels]
    items, handles = build_items(cfg.dataset, cfg.n_prompts, cfg.seed, levels)
    write_jsonl([i.to_dict() for i in items], out / "raw" / "items.jsonl")
    print(f"[phase0] {len(items)} items across {len(levels)} levels -> {out}")

    engine = make_engine(cfg)
    if cfg.backend == "mock":
        print("[phase0] *** MOCK BACKEND -- results are synthetic ***")

    t0 = time.time()
    forced = run_condition(
        engine, items, handles,
        condition="forced_no_tool", with_tools=False,
        n_samples=cfg.forced_n_samples, temperature=cfg.forced_temperature,
        max_new_tokens=cfg.max_new_tokens, max_tool_rounds=0,
        tool_format=cfg.tool_format, seed=cfg.seed,
    )
    write_jsonl(forced, out / "raw" / "rollouts_forced_no_tool.jsonl")

    free = run_condition(
        engine, items, handles,
        condition="free_choice", with_tools=True,
        n_samples=cfg.free_n_samples, temperature=cfg.free_temperature,
        max_new_tokens=cfg.max_new_tokens, max_tool_rounds=cfg.max_tool_rounds,
        tool_format=cfg.tool_format, seed=cfg.seed + 7,
    )
    write_jsonl(free, out / "raw" / "rollouts_free_choice.jsonl")
    print(f"[phase0] generation finished in {time.time() - t0:.1f}s")

    meta = cfg.to_dict() | {"backend_name": engine.name, "n_items": len(items)}
    report = analyse(free, forced, meta, seed=cfg.seed)
    (out / "summary" / "calibration_summary.json").write_text(json.dumps(report, indent=2))
    write_csv(report["per_level"], out / "summary" / "per_level.csv")

    from neurosymbolic_faithfulness.plot_calibration import plot_calibration
    plot_calibration(report, out / "plots")

    from neurosymbolic_faithfulness.sample_transcripts import dump_transcripts
    dump_transcripts(free, forced, out / "transcripts", n=30, seed=cfg.seed)

    print_report(report)
    return 0 if report["gate"]["passed"] and report["parsing"]["within_budget"] else 2


def print_report(report: dict) -> None:
    print("\n" + "=" * 78)
    print("PHASE 0 CALIBRATION")
    print("=" * 78)
    print(f"{'level':<14} {'terms':>5} {'digits':>6} | "
          f"{'unaided acc [95% CI]':>22} | {'tool rate [95% CI]':>22}")
    print("-" * 78)
    for d in report["per_level"]:
        acc = (f"{d['unaided_accuracy']:.2f} "
               f"[{d['unaided_accuracy_lo']:.2f},{d['unaided_accuracy_hi']:.2f}]")
        tr = (f"{d['tool_call_rate_item_mean']:.2f} "
              f"[{d['tool_call_rate_item_lo']:.2f},{d['tool_call_rate_item_hi']:.2f}]")
        star = "  <== in band" if d["level_index"] in report["gate"]["levels_in_band"] else ""
        print(f"{d['level_name']:<14} {d['num_terms']:>5} {d['num_digits']:>6} | "
              f"{acc:>22} | {tr:>22}{star}")

    p = report["parsing"]
    print(f"\nmalformed: label-rate {p['malformed_label_rate']:.3f} | "
          f"any-malformed {p['any_malformed_rate']:.3f} | "
          f"budget {p['malformed_budget']:.2f} -> "
          f"{'OK' if p['within_budget'] else 'OVER BUDGET: fix parsing before proceeding'}")
    if p["reasons"]:
        for reason, n in list(p["reasons"].items())[:8]:
            print(f"    {n:>5}  {reason}")

    g = report["gate"]
    print("\n" + "-" * 78)
    if g["passed"]:
        print(f"GATE PASSED: levels {g['level_names_in_band']} sit inside "
              f"[{g['band_low']}, {g['band_high']}].")
        print("Proceed to Phase 1 using these levels.")
    else:
        print(f"GATE FAILED: no level has a tool-call rate inside "
              f"[{g['band_low']}, {g['band_high']}].")
        print(f"Failure mode: {g['failure_mode']}")
        print("STOP. Do not proceed to Phase 1 -- there is no variance to explain.")
    print("-" * 78)


if __name__ == "__main__":
    sys.exit(main())
