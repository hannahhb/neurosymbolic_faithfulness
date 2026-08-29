"""Phase 0 metrics, confidence intervals and the go/no-go gate.

Two things worth flagging about the statistics:

* The free-choice condition draws 8 samples per prompt, so those 8 are NOT
  independent observations.  Every rate is therefore reported twice: pooled over
  rollouts (`*_pooled`) and as a mean over per-item rates with a bootstrap CI
  over items (`*_item_mean`).  The item-level CI is the one to trust; the pooled
  one is reported because it is what a naive count would give.
* The gate is evaluated on the item-mean rate.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; the right choice for rates near 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Iterable[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap over items (the clustering unit)."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def _by_item(rows: list[dict], key) -> dict[str, list]:
    out = defaultdict(list)
    for r in rows:
        out[r["item_id"]].append(key(r))
    return out


def summarise_level(free: list[dict], forced: list[dict], seed: int = 0) -> dict[str, Any]:
    """All Phase 0 numbers for a single difficulty level."""
    lvl = (free or forced)[0]
    out: dict[str, Any] = {
        "level_index": lvl["level_index"],
        "level_name": lvl["level_name"],
        "num_terms": lvl["num_terms"],
        "num_digits": lvl["num_digits"],
        "work": lvl["work"],
        "n_items_free": len({r["item_id"] for r in free}),
        "n_rollouts_free": len(free),
        "n_rollouts_forced": len(forced),
    }

    # --- forced no-tool: true tool necessity -------------------------------
    k = sum(r["correct"] for r in forced)
    n = len(forced)
    out["unaided_accuracy"] = k / n if n else float("nan")
    out["unaided_accuracy_lo"], out["unaided_accuracy_hi"] = wilson(k, n)
    out["unaided_n_correct"] = k

    # --- free choice: tool-call rate ---------------------------------------
    k = sum(r["label"] == "TOOL" for r in free)
    n = len(free)
    out["tool_call_rate_pooled"] = k / n if n else float("nan")
    out["tool_call_rate_pooled_lo"], out["tool_call_rate_pooled_hi"] = wilson(k, n)

    per_item = _by_item(free, lambda r: float(r["label"] == "TOOL"))
    item_rates = [float(np.mean(v)) for v in per_item.values()]
    out["tool_call_rate_item_mean"] = float(np.mean(item_rates)) if item_rates else float("nan")
    lo, hi = bootstrap_ci(item_rates, seed=seed)
    out["tool_call_rate_item_lo"], out["tool_call_rate_item_hi"] = lo, hi
    # How much of the variance is between prompts vs within a prompt?  A high
    # between-item share is the first hint that the decision is prompt-determined.
    out["tool_rate_item_std"] = float(np.std(item_rates)) if item_rates else float("nan")
    out["frac_items_unanimous"] = (
        float(np.mean([r in (0.0, 1.0) for r in item_rates])) if item_rates else float("nan")
    )

    # --- free-choice accuracy, split by what the model chose ----------------
    out["free_accuracy"] = float(np.mean([r["correct"] for r in free])) if free else float("nan")
    for lab in ("TOOL", "NO_TOOL", "MALFORMED"):
        sub = [r for r in free if r["label"] == lab]
        out[f"free_accuracy_{lab.lower()}"] = (
            float(np.mean([r["correct"] for r in sub])) if sub else float("nan")
        )
        out[f"n_{lab.lower()}"] = len(sub)

    # --- parsing health ----------------------------------------------------
    out["malformed_rate"] = (
        float(np.mean([r["label"] == "MALFORMED" for r in free])) if free else float("nan")
    )
    out["any_malformed_rate"] = (
        float(np.mean([r["n_malformed_calls"] > 0 for r in free])) if free else float("nan")
    )
    out["truncated_call_rate"] = (
        float(np.mean([r["n_truncated_calls"] > 0 for r in free])) if free else float("nan")
    )
    out["tool_error_rate"] = (
        float(np.mean([r["n_tool_errors"] > 0 for r in free])) if free else float("nan")
    )
    out["round_cap_rate"] = (
        float(np.mean([r["hit_round_cap"] for r in free])) if free else float("nan")
    )
    out["length_finish_rate_free"] = (
        float(np.mean([r["finish_reason_last"] == "length" for r in free])) if free else float("nan")
    )
    out["length_finish_rate_forced"] = (
        float(np.mean([r["finish_reason_last"] == "length" for r in forced])) if forced else float("nan")
    )
    return out


def malformed_reasons(free: list[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in free:
        for t in r["turns"]:
            for m in t["malformed"]:
                c[m["reason"]] += 1
    return dict(c.most_common())


def extraction_methods(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(r["extraction_method"] for r in rows).most_common())


def analyse(free: list[dict], forced: list[dict], cfg: dict, seed: int = 0) -> dict[str, Any]:
    levels = sorted({r["level_index"] for r in free} | {r["level_index"] for r in forced})
    per_level = [
        summarise_level(
            [r for r in free if r["level_index"] == L],
            [r for r in forced if r["level_index"] == L],
            seed=seed,
        )
        for L in levels
    ]

    lo, hi = cfg["gate_low"], cfg["gate_high"]
    in_band = [d for d in per_level if lo <= d["tool_call_rate_item_mean"] <= hi]

    overall_malformed = float(np.mean([r["label"] == "MALFORMED" for r in free])) if free else 0.0
    overall_any_malformed = (
        float(np.mean([r["n_malformed_calls"] > 0 for r in free])) if free else 0.0
    )

    gate = {
        "band_low": lo,
        "band_high": hi,
        "levels_in_band": [d["level_index"] for d in in_band],
        "level_names_in_band": [d["level_name"] for d in in_band],
        "passed": len(in_band) > 0,
        "tool_rate_by_level": {
            d["level_index"]: d["tool_call_rate_item_mean"] for d in per_level
        },
    }
    if not gate["passed"]:
        rates = [d["tool_call_rate_item_mean"] for d in per_level]
        gate["failure_mode"] = (
            "model never calls the tool" if max(rates) < lo
            else "model always calls the tool" if min(rates) > hi
            else "rate jumps across the band between adjacent levels; "
                 "refine the ladder between the bracketing levels"
        )

    parsing = {
        "malformed_budget": cfg["malformed_budget"],
        "malformed_label_rate": overall_malformed,
        "any_malformed_rate": overall_any_malformed,
        "within_budget": overall_any_malformed <= cfg["malformed_budget"],
        "reasons": malformed_reasons(free),
    }

    return {
        "config": cfg,
        "per_level": per_level,
        "gate": gate,
        "parsing": parsing,
        "extraction_methods_free": extraction_methods(free),
        "extraction_methods_forced": extraction_methods(forced),
    }


CSV_COLUMNS = [
    "level_index", "level_name", "num_terms", "num_digits", "work",
    "n_items_free", "n_rollouts_free", "n_rollouts_forced",
    "unaided_accuracy", "unaided_accuracy_lo", "unaided_accuracy_hi",
    "tool_call_rate_item_mean", "tool_call_rate_item_lo", "tool_call_rate_item_hi",
    "tool_call_rate_pooled", "tool_call_rate_pooled_lo", "tool_call_rate_pooled_hi",
    "tool_rate_item_std", "frac_items_unanimous",
    "free_accuracy", "free_accuracy_tool", "free_accuracy_no_tool",
    "n_tool", "n_no_tool", "n_malformed",
    "malformed_rate", "any_malformed_rate", "truncated_call_rate",
    "tool_error_rate", "round_cap_rate",
    "length_finish_rate_free", "length_finish_rate_forced",
]


def write_csv(per_level: list[dict], path: Path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in per_level:
            w.writerow({k: row.get(k) for k in CSV_COLUMNS})
