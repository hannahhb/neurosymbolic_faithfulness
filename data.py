"""The two problem sources.

apple/GSM-Symbolic -- 100 templates x 50 instances.  Both arms are generated
    ourselves here, and template identity (`original_id`) is the clustering unit,
    since instances of one template differ only in names and numbers.

uzaymacar/math-rollouts -- the Thought Anchors companion data.  Its
    chunks_labeled.json ALREADY carries per-chunk resampling_importance_kl,
    counterfactual_importance_kl, forced_importance_kl, the 8-category
    function_tags, depends_on and overdeterminedness, for
    R1-Distill-Qwen-14B and R1-Distill-Llama-8B on MATH.  So the CoT arm on
    those problems is free: only the code arm has to be generated, and only if
    the same model is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Any

GSM_CONFIGS = ("main", "p1", "p2")
MR_REPO = "uzaymacar/math-rollouts"
MR_MODELS = ("deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-llama-8b")
MR_TEMP = "temperature_0.6_top_p_0.95"
MR_SPLITS = ("correct_base_solution", "incorrect_base_solution")

_FINAL = re.compile(r"####\s*([-\d,\.]+)")


@dataclass
class Problem:
    problem_id: str
    source: str              # "gsm_symbolic" | "math_rollouts"
    question: str
    answer: str              # ground truth, bare
    cluster_id: str          # template id (GSM) or problem id (MATH)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_number(s: str) -> str:
    return s.strip().replace(",", "").rstrip(".")


def load_gsm_symbolic(n: int, seed: int = 0, config: str = "main",
                      instances_per_template: int = 1) -> list[Problem]:
    """Sample problems, spreading across templates rather than clustering on a
    few, since templates are the unit the statistics cluster on."""
    from datasets import load_dataset

    ds = load_dataset("apple/GSM-Symbolic", config, split="test")
    by_tpl: dict[int, list[int]] = {}
    for i, tid in enumerate(ds["original_id"]):
        by_tpl.setdefault(int(tid), []).append(i)

    import random
    rng = random.Random(seed)
    tpls = sorted(by_tpl)
    rng.shuffle(tpls)

    picked: list[int] = []
    for t in tpls:
        rows = by_tpl[t][:]
        rng.shuffle(rows)
        picked.extend(rows[:instances_per_template])
        if len(picked) >= n:
            break
    picked = picked[:n]

    out = []
    for i in picked:
        r = ds[i]
        m = _FINAL.search(r["answer"])
        out.append(Problem(
            problem_id=f"gsm:{config}:{r['id']}:{r['instance']}",
            source="gsm_symbolic",
            question=r["question"],
            answer=_clean_number(m.group(1)) if m else "",
            cluster_id=f"tpl:{r['original_id']}",
            meta={"worked_solution": r["answer"], "original_id": int(r["original_id"]),
                  "instance": int(r["instance"])},
        ))
    return out


def list_math_rollout_problems(model: str, split: str) -> list[str]:
    from huggingface_hub import HfApi

    prefix = f"{model}/{MR_TEMP}/{split}/"
    files = HfApi().list_repo_files(MR_REPO, repo_type="dataset")
    return sorted({f[len(prefix):].split("/")[0]
                   for f in files
                   if f.startswith(prefix) and f[len(prefix):].startswith("problem_")})


def load_math_rollouts(model: str = MR_MODELS[0], split: str = MR_SPLITS[0],
                       limit: int | None = None) -> list[Problem]:
    """Problems plus the pre-computed per-chunk Thought Anchors metrics."""
    from huggingface_hub import hf_hub_download

    ids = list_math_rollout_problems(model, split)
    if limit:
        ids = ids[:limit]
    out = []
    for pid in ids:
        base = f"{model}/{MR_TEMP}/{split}/{pid}"
        prob = json.load(open(hf_hub_download(MR_REPO, f"{base}/problem.json",
                                              repo_type="dataset")))
        try:
            chunks = json.load(open(hf_hub_download(
                MR_REPO, f"{base}/chunks_labeled.json", repo_type="dataset")))
        except Exception:
            chunks = []
        out.append(Problem(
            problem_id=f"mr:{model}:{split}:{pid}",
            source="math_rollouts",
            question=prob["problem"],
            answer=_clean_number(str(prob.get("gt_answer", ""))),
            cluster_id=f"mr:{pid}",
            meta={"level": prob.get("level"), "type": prob.get("type"),
                  "nickname": prob.get("nickname"), "model": model, "split": split,
                  "cot_chunks": chunks},
        ))
    return out


# --- reference CoT metrics, for the comparison table -------------------------
COT_METRIC_KEYS = (
    "resampling_importance_kl", "counterfactual_importance_kl",
    "forced_importance_kl", "resampling_importance_accuracy",
    "counterfactual_importance_accuracy", "forced_importance_accuracy",
    "accuracy", "overdeterminedness", "different_trajectories_fraction",
)


def cot_reference_steps(p: Problem) -> list[dict]:
    """Flatten a math_rollouts problem's pre-computed CoT chunk metrics into the
    same shape our own code-arm measurements produce."""
    rows = []
    for c in p.meta.get("cot_chunks", []):
        rows.append({
            "problem_id": p.problem_id, "cluster_id": p.cluster_id,
            "condition": "cot", "source": "math_rollouts_precomputed",
            "step_idx": c.get("chunk_idx"), "text": c.get("chunk"),
            "function_tags": c.get("function_tags"), "depends_on": c.get("depends_on"),
            **{k: c.get(k) for k in COT_METRIC_KEYS},
        })
    return rows
