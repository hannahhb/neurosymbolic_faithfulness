"""gsm_symbolic items plus the biasing distractor X.

X is chosen to be *plausible*: wherever possible it is an intermediate value
from the problem's own solution -- the number you would land on by stopping a
step early -- rather than a random wrong number.  gsm_symbolic exposes every
intermediate in metadata["variables"], and metadata["answer_cot"] shows which
of them are computed rather than given.

Consequence to state in the write-up: intermediate-value hints are more
seductive than random ones, so absolute susceptibility rates are inflated.  The
NL/SYM comparison is unaffected because X is identical across conditions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, field
from typing import Any

import reasoning_gym

# gsm_symbolic asserts on any difficulty other than 1.0 (verified against the
# installed package); there is no difficulty knob to sweep here.
DIFFICULTY = 1.0


@dataclass
class Item:
    item_id: str
    source_index: int
    template_id: int            # which of the 100 generators produced this item
    question: str
    answer: str                 # ground truth, as a string
    answer_value: float
    distractor: str             # X, the wrong answer injected as a hint
    distractor_value: float
    distractor_kind: str        # "intermediate" | "perturbed"
    variables: dict[str, Any]
    answer_cot: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(round(v, 4))


def _question_numbers(question: str) -> set[float]:
    return {float(m) for m in re.findall(r"\d+(?:\.\d+)?", question)}


def choose_distractor(question: str, answer: float, variables: dict,
                      answer_cot: str) -> tuple[float, str]:
    """Pick a plausible wrong answer.

    Preference order:
      1. a *computed* intermediate -- appears in the worked solution but is not
         one of the numbers handed to the model in the question
      2. any other intermediate
      3. a perturbation of the answer (last resort, recorded as such)

    Among candidates, the one closest to the answer in log-magnitude wins: a
    hint three orders of magnitude off is not a temptation, it is a typo.
    """
    given = _question_numbers(question)
    cands: list[tuple[int, float, float]] = []   # (tier, distance, value)
    for v in variables.values():
        x = _numeric(v)
        if x is None or x == answer or x == 0:
            continue
        computed = (_fmt(x) in answer_cot) and (x not in given)
        dist = abs(math.log10(abs(x) + 1) - math.log10(abs(answer) + 1))
        cands.append((0 if computed else 1, dist, x))
    if cands:
        cands.sort(key=lambda t: (t[0], t[1]))
        return cands[0][2], "intermediate"

    # nothing usable in the metadata: perturb the answer in a way a model
    # plausibly might, and mark it so these items can be split out in analysis
    for factor in (2.0, 0.5, 1.1):
        x = round(answer * factor)
        if x != answer and x != 0:
            return float(x), "perturbed"
    return float(answer + 1), "perturbed"


def build_items(n: int, seed: int) -> tuple[list[Item], Any]:
    ds = reasoning_gym.create_dataset(
        "gsm_symbolic", size=n, seed=seed, difficulty=DIFFICULTY
    )
    items: list[Item] = []
    for i in range(n):
        e = ds[i]
        m = e["metadata"]
        ans = float(m["answer_value"])
        x, kind = choose_distractor(e["question"], ans, m.get("variables", {}),
                                    m.get("answer_cot", ""))
        items.append(Item(
            item_id=f"gsm:{seed}:{i:05d}",
            source_index=i,
            template_id=int(ds.task_indices[i]),
            question=e["question"],
            answer=str(e["answer"]),
            answer_value=ans,
            distractor=_fmt(x),
            distractor_value=x,
            distractor_kind=kind,
            variables=m.get("variables", {}),
            answer_cot=m.get("answer_cot", ""),
        ))
    return items, ds
