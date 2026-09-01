"""Dataset loading: MATH and GSM8K into one Item schema.

Answers stay free-form.  MATH golds are dug out of the \\boxed{...} in the
reference solution; GSM8K golds follow the `#### N` marker.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


@dataclass
class Item:
    item_id: str
    dataset: str  # "gsm8k" | "math"
    problem: str
    gold: str
    level: Optional[str] = None   # MATH: "Level 1".."Level 5"
    subject: Optional[str] = None  # MATH: config name

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _item_id(dataset: str, problem: str) -> str:
    h = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}-{h}"


def extract_boxed(solution: str) -> Optional[str]:
    """Return the content of the last \\boxed{...}, brace-matched.

    Regex is not enough here: MATH solutions nest braces inside \\boxed, e.g.
    \\boxed{\\frac{1}{2}}, so we scan and count depth.
    """
    idx = solution.rfind("\\boxed")
    if idx == -1:
        # A few solutions use \fbox instead.
        idx = solution.rfind("\\fbox")
        if idx == -1:
            return None
    start = solution.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1 : i].strip()
    return None


def load_gsm8k(split: str = "test") -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    items: list[Item] = []
    for row in ds:
        answer = row["answer"]
        if "####" not in answer:
            continue
        gold = answer.split("####")[-1].strip().replace(",", "")
        problem = row["question"].strip()
        items.append(
            Item(
                item_id=_item_id("gsm8k", problem),
                dataset="gsm8k",
                problem=problem,
                gold=gold,
            )
        )
    return items


def load_math(split: str = "test", configs: Iterable[str] = MATH_CONFIGS) -> list[Item]:
    from datasets import load_dataset

    items: list[Item] = []
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
        for row in ds:
            gold = extract_boxed(row["solution"])
            if gold is None:
                continue  # no recoverable reference answer; unusable as a label
            problem = row["problem"].strip()
            items.append(
                Item(
                    item_id=_item_id("math", problem),
                    dataset="math",
                    problem=problem,
                    gold=gold,
                    level=row.get("level"),
                    subject=cfg,
                )
            )
    return items


def stratified_sample(items: list[Item], n: int, seed: int = 0) -> list[Item]:
    """Sample `n` items, balanced across (subject, level) where those exist.

    Balance matters for the probe: MATH's levels are the natural hardness axis,
    and an unbalanced draw makes probe accuracy a proxy for difficulty mix
    rather than for decodability.
    """
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    strata: dict[tuple, list[Item]] = {}
    for it in items:
        strata.setdefault((it.subject, it.level), []).append(it)
    for bucket in strata.values():
        rng.shuffle(bucket)

    out: list[Item] = []
    keys = sorted(strata, key=lambda k: (str(k[0]), str(k[1])))
    while len(out) < n:
        progressed = False
        for k in keys:
            if strata[k]:
                out.append(strata[k].pop())
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
    rng.shuffle(out)
    return out


def write_items(items: list[Item], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for it in items:
            fh.write(it.to_json() + "\n")
    return path


def read_items(path: str | Path) -> list[Item]:
    with Path(path).open() as fh:
        return [Item(**json.loads(line)) for line in fh if line.strip()]
