"""Phase 0 figures. Everything saved as PNG next to the numbers that made it."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NECESSITY = "#0b6e4f"   # forced-no-tool accuracy
CHOICE = "#b8412c"      # free-choice tool-call rate
BAND = "#f2c14e"


def _xticks(ax, per_level):
    ax.set_xticks([d["level_index"] for d in per_level])
    ax.set_xticklabels(
        [f"L{d['level_index']}\n{d['num_terms']}x{d['num_digits']}d" for d in per_level]
    )
    ax.set_xlabel("difficulty level  (terms x digits)")


def plot_calibration(report: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_level = report["per_level"]
    cfg = report["config"]
    x = np.array([d["level_index"] for d in per_level], dtype=float)
    written: list[Path] = []

    # --- figure 1: the two calibration curves -------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=160)
    lo, hi = report["gate"]["band_low"], report["gate"]["band_high"]
    ax.axhspan(lo, hi, color=BAND, alpha=0.22, zorder=0,
               label=f"gate band [{lo:g}, {hi:g}]")

    acc = np.array([d["unaided_accuracy"] for d in per_level])
    acc_lo = np.array([d["unaided_accuracy_lo"] for d in per_level])
    acc_hi = np.array([d["unaided_accuracy_hi"] for d in per_level])
    ax.plot(x, acc, "o-", color=NECESSITY, lw=2, label="unaided accuracy (tool removed, T=0)")
    ax.fill_between(x, acc_lo, acc_hi, color=NECESSITY, alpha=0.15, lw=0)

    tr = np.array([d["tool_call_rate_item_mean"] for d in per_level])
    tr_lo = np.array([d["tool_call_rate_item_lo"] for d in per_level])
    tr_hi = np.array([d["tool_call_rate_item_hi"] for d in per_level])
    ax.plot(x, tr, "s-", color=CHOICE, lw=2,
            label=f"tool-call rate (free choice, T={cfg['free_temperature']}, "
                  f"n={cfg['free_n_samples']})")
    ax.fill_between(x, tr_lo, tr_hi, color=CHOICE, alpha=0.15, lw=0)

    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("rate")
    _xticks(ax, per_level)
    ax.set_title(f"Phase 0 calibration -- {cfg.get('backend_name', cfg['model'])}\n"
                 f"{cfg['dataset']}, {cfg['n_prompts']} prompts/level",
                 fontsize=11)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              frameon=False)
    fig.tight_layout()
    p = out_dir / "calibration_curves.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # --- figure 2: does the tool actually help? -----------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    ax.plot(x, [d["unaided_accuracy"] for d in per_level], "o-", color=NECESSITY,
            label="forced no-tool")
    ax.plot(x, [d["free_accuracy_tool"] for d in per_level], "s--", color=CHOICE,
            label="free choice, TOOL rollouts")
    ax.plot(x, [d["free_accuracy_no_tool"] for d in per_level], "^:", color="#4a4a4a",
            label="free choice, NO_TOOL rollouts")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("accuracy")
    _xticks(ax, per_level)
    ax.set_title("Accuracy by what the model chose\n"
                 "(observational: rollouts are not randomised into TOOL/NO_TOOL)",
                 fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    p = out_dir / "accuracy_by_choice.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # --- figure 3: where does the variance live? ---------------------------
    # If per-item tool rates are all 0 or 1, the decision is fully determined by
    # the prompt and there is no within-prompt variance for Phase 1 to explain.
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    ax.bar(x - 0.18, [d["frac_items_unanimous"] for d in per_level], width=0.36,
           color="#3d5a80", label="fraction of prompts where all samples agree")
    ax.bar(x + 0.18, [d["tool_rate_item_std"] for d in per_level], width=0.36,
           color="#98c1d9", label="std of per-prompt tool rate")
    ax.set_ylim(0, 1.05)
    _xticks(ax, per_level)
    ax.set_ylabel("value")
    ax.set_title("Between-prompt vs within-prompt variance in the decision", fontsize=10)
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    p = out_dir / "decision_variance.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    return written


if __name__ == "__main__":
    import sys

    run = Path(sys.argv[1])
    rep = json.loads((run / "summary" / "calibration_summary.json").read_text())
    for f in plot_calibration(rep, run / "plots"):
        print(f)
