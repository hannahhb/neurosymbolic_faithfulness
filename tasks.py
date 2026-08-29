"""Reasoning Gym task construction and the difficulty ladder.

Parameter names were taken from the installed `reasoning_gym` source and
cross-checked against GALLERY.md.  Note that GALLERY.md is STALE for `products`
-- it documents `min_factors`/`max_factors`, but ProductsConfig actually uses
`min_terms`/`max_terms`.  The names below are the ones the installed package
accepts (verified by constructing every level in `self_check()`).

chain_sum  : min_terms, max_terms, min_digits, max_digits, allow_negation, seed, size
products   : min_terms, max_terms, min_digits, max_digits, allow_negation, seed, size
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import reasoning_gym


@dataclass(frozen=True)
class Level:
    """One rung of the difficulty ladder.

    `num_terms` and `num_digits` are pinned (min == max) so that every prompt at
    a level is homogeneous -- otherwise within-level difficulty variance would
    contaminate the calibration curves.
    """

    index: int          # ordinal rung, 1..K; the x-axis of the calibration plots
    num_terms: int
    num_digits: int

    @property
    def name(self) -> str:
        return f"L{self.index}_t{self.num_terms}_d{self.num_digits}"

    @property
    def work(self) -> int:
        """A scalar difficulty proxy: total digits of input to combine."""
        return self.num_terms * self.num_digits

    def kwargs(self) -> dict[str, int]:
        return {
            "min_terms": self.num_terms,
            "max_terms": self.num_terms,
            "min_digits": self.num_digits,
            "max_digits": self.num_digits,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["name"] = self.name
        d["work"] = self.work
        return d


# Six rungs spanning "trivial" to "clearly impossible in one's head".
# chain_sum is addition/subtraction, so difficulty has to be pushed via both the
# number of terms and their width.  products is multiplication, whose difficulty
# rises far faster per digit, so its ladder is shallower.
LADDERS: dict[str, list[Level]] = {
    "chain_sum": [
        Level(1, 2, 1),   # 4 + 3
        Level(2, 3, 2),   # 15 - 43 - 75
        Level(3, 4, 3),
        Level(4, 5, 5),
        Level(5, 6, 7),
        Level(6, 8, 9),   # 8 nine-digit terms
    ],
    "products": [
        Level(1, 2, 1),   # 4 * 3
        Level(2, 2, 2),
        Level(3, 2, 3),
        Level(4, 2, 4),
        Level(5, 3, 3),
        Level(6, 3, 4),
    ],
}


@dataclass
class Item:
    """One task instance, carrying everything needed to score and to re-run it."""

    item_id: str
    dataset: str
    level_index: int
    level_name: str
    num_terms: int
    num_digits: int
    work: int
    seed: int
    source_index: int
    question: str
    answer: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_custom_levels(spec: str) -> list[Level]:
    """Build an ad-hoc ladder from a spec like "4x3,4x4,5x4,5x5" (terms x digits).

    This is the remediation path when the gate fails because the tool-call rate
    jumps across the band between two adjacent rungs: re-run with intermediate
    rungs rather than editing LADDERS.
    """
    levels = []
    for i, part in enumerate(spec.split(","), start=1):
        terms, _, digits = part.strip().lower().partition("x")
        levels.append(Level(i, int(terms), int(digits)))
    return levels


def build_items(
    dataset_name: str,
    n_prompts: int,
    seed: int,
    levels: list[Level] | None = None,
) -> tuple[list[Item], dict[str, Any]]:
    """Generate `n_prompts` items at every rung of the ladder.

    Each level gets its own `reasoning_gym` dataset with a distinct seed so that
    levels are independent samples rather than nested prefixes of one stream.
    Returns the items plus the live dataset handles keyed by level name, which
    the scorer needs for `score_answer`.
    """
    if dataset_name not in LADDERS:
        raise ValueError(
            f"no difficulty ladder defined for {dataset_name!r}; "
            f"known: {sorted(LADDERS)}"
        )
    levels = levels if levels is not None else LADDERS[dataset_name]

    items: list[Item] = []
    handles: dict[str, Any] = {}
    for level in levels:
        level_seed = seed + 1000 * level.index
        ds = reasoning_gym.create_dataset(
            dataset_name, size=n_prompts, seed=level_seed, **level.kwargs()
        )
        handles[level.name] = ds
        for i in range(n_prompts):
            entry = ds[i]
            items.append(
                Item(
                    item_id=f"{dataset_name}:{level.name}:{i:04d}",
                    dataset=dataset_name,
                    level_index=level.index,
                    level_name=level.name,
                    num_terms=level.num_terms,
                    num_digits=level.num_digits,
                    work=level.work,
                    seed=level_seed,
                    source_index=i,
                    question=entry["question"],
                    answer=str(entry["answer"]),
                    metadata=entry["metadata"],
                )
            )
    return items, handles


def score(handles: dict[str, Any], item: Item, answer: str | None) -> float:
    """Reasoning Gym's own verifier. `answer` must already be extracted.

    Passing raw model prose here would earn spurious partial credit -- the
    default scorer awards len(oracle)/len(answer) for substring containment --
    so callers must hand it a bare answer string.
    """
    ds = handles[item.level_name]
    entry = ds[item.source_index]
    assert str(entry["answer"]) == item.answer, "dataset regenerated differently"
    return float(ds.score_answer(answer=answer, entry=entry))


def self_check() -> None:
    """Construct every level of every ladder; fails loudly on a bad kwarg."""
    for name, levels in LADDERS.items():
        for lv in levels:
            ds = reasoning_gym.create_dataset(name, size=1, seed=0, **lv.kwargs())
            e = ds[0]
            assert ds.score_answer(answer=str(e["answer"]), entry=e) == 1.0
            print(f"  {name:10s} {lv.name:14s} work={lv.work:3d}  {e['question']}")


if __name__ == "__main__":
    self_check()
