"""Dump raw transcripts for the mandatory hand-read, plus a crude marker count.

The marker counts are a *pointer*, not a result: they say where to look, and the
Phase 0 write-up should quote what a human actually read, not these regexes.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# Phrases that would indicate the model explicitly deliberates about the choice
# before making it.  Kept deliberately narrow; a miss here is better than a
# false positive that inflates the "it deliberates" story.
MARKERS = {
    "size_too_big": r"\b(too (?:big|large|long|many)|large numbers?|many digits|"
                    r"multi-?digit)\b",
    "mental_capacity": r"\b(in my head|mentally|by hand|manually|without a calculator)\b",
    "reliability": r"\b(error-prone|prone to (?:error|mistakes)|risk of (?:an )?error|"
                   r"to (?:be|make) sure|double[- ]check|accurat\w+|reliab\w+|avoid mistakes)\b",
    "tool_intent": r"\b(use the calculator|call the calculator|use a calculator|"
                   r"let me (?:compute|calculate)|I'?ll (?:compute|calculate|use))\b",
    "easy_enough": r"\b(simple|straightforward|easy|trivial|I can (?:do|compute) this)\b",
}
COMPILED = {k: re.compile(v, re.I) for k, v in MARKERS.items()}


def marker_counts(text: str) -> dict[str, int]:
    return {k: len(r.findall(text)) for k, r in COMPILED.items()}


def _fmt(rec: dict) -> str:
    lines = [
        f"### {rec['item_id']}  sample={rec['sample_idx']}  "
        f"condition={rec['condition']}  label={rec['label']}",
        "",
        f"- level: {rec['level_name']} ({rec['num_terms']} terms x {rec['num_digits']} digits)",
        f"- question: `{rec['question']}`",
        f"- ground truth: `{rec['gt_answer']}`",
        f"- extracted: `{rec['extracted_answer']}` via `{rec['extraction_method']}` "
        f"-> score {rec['score']:.2f}",
        f"- prompt sha256[:16]: `{rec['prompt_sha256']}`",
        "",
        "**Decision prefix** (everything the model emitted before its first call attempt):",
        "",
        "```",
        rec["decision_prefix"].strip() or "(empty -- the model called the tool immediately)",
        "```",
        "",
    ]
    for t in rec["turns"]:
        lines += [f"**Assistant turn {t['round']}** (finish={t['finish_reason']}):", "",
                  "```", t["text"].strip(), "```", ""]
        for r in t["tool_results"]:
            lines += [f"> tool <- `{r['expression']}`",
                      f"> tool -> `{r['value'] if r['ok'] else 'Error: ' + str(r['error'])}`",
                      ""]
        for m in t["malformed"]:
            lines += [f"> MALFORMED: {m['reason']}", "```", m["raw"].strip(), "```", ""]
    lines.append("---\n")
    return "\n".join(lines)


def dump_transcripts(free: list[dict], forced: list[dict], out_dir: Path,
                     n: int = 30, seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Stratify over (level, label) so the read is not dominated by whichever
    # cell happens to be largest.
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in free:
        buckets[(r["level_index"], r["label"])].append(r)
    keys = sorted(buckets)
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            picked.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
        i += 1
        if i > 10000:
            break

    body = ["# Phase 0 -- transcripts for hand-reading",
            "",
            f"{len(picked)} free-choice rollouts, stratified over "
            f"(difficulty level x TOOL/NO_TOOL/MALFORMED).",
            "",
            "Read the **decision prefix** of each: is there explicit deliberation "
            "about whether to use the tool, or does the model simply start calling?",
            "", "---", ""]
    body += [_fmt(r) for r in picked]
    path = out_dir / "handread_sample.md"
    path.write_text("\n".join(body))

    # marker frequencies over ALL free-choice rollouts, split by label
    stats: dict = {}
    for label in ("TOOL", "NO_TOOL", "MALFORMED"):
        sub = [r for r in free if r["label"] == label]
        agg = Counter()
        n_with_any = 0
        empty_prefix = 0
        prefix_chars = []
        for r in sub:
            c = marker_counts(r["decision_prefix"])
            agg.update({k: v for k, v in c.items() if v})
            n_with_any += int(any(c.values()))
            empty_prefix += int(not r["decision_prefix"].strip())
            prefix_chars.append(len(r["decision_prefix"]))
        stats[label] = {
            "n": len(sub),
            "marker_hits": dict(agg),
            "frac_with_any_marker": (n_with_any / len(sub)) if sub else None,
            "frac_empty_decision_prefix": (empty_prefix / len(sub)) if sub else None,
            "mean_decision_prefix_chars": (sum(prefix_chars) / len(sub)) if sub else None,
        }
    (out_dir / "marker_stats.json").write_text(json.dumps(
        {"note": "Heuristic regex counts over the decision prefix. A pointer for "
                 "the hand-read, not a finding.",
         "markers": MARKERS, "by_label": stats}, indent=2))

    forced_path = out_dir / "forced_no_tool_sample.md"
    fsample = rng.sample(forced, min(10, len(forced)))
    forced_path.write_text("\n".join(
        ["# Phase 0 -- forced no-tool transcripts (spot check)", "", "---", ""]
        + [_fmt(r) for r in fsample]))
    return path


if __name__ == "__main__":
    import sys

    run = Path(sys.argv[1])
    free = [json.loads(l) for l in open(run / "raw" / "rollouts_free_choice.jsonl")]
    forced = [json.loads(l) for l in open(run / "raw" / "rollouts_forced_no_tool.jsonl")]
    print(dump_transcripts(free, forced, run / "transcripts", n=30))
