"""Linear probes over the (position x layer) grid.

The question each probe answers: given the residual stream at position p, layer
l, can a linear map recover the first token of the answer this rollout ends up
producing?

Three things this module insists on, because each of them is a way to get a
result that looks strong and means nothing:

1. Grouped splits.  Folds are split by item_id, so multiple samples of the same
   problem never straddle train and test.
2. An empirical null.  With ~10 skewed classes, "above chance" is not 1/k and
   not obviously the majority rate either.  We permute labels within the same
   fold structure and report the null distribution.
3. Baseline-relative reporting.  CoT and PoT produce different answers, so their
   label distributions differ and their raw accuracies are not comparable.  What
   compares is accuracy above that condition's own majority baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    position: str
    layer: int
    n: int
    n_classes: int
    accuracy: float
    macro_f1: float
    majority: float
    null_mean: float
    null_std: float
    delta_majority: float  # accuracy - majority; the headline number
    z_vs_null: float

    def to_dict(self) -> dict:
        return asdict(self)


def _make_clf(C: float, max_dim: Optional[int], seed: int):
    steps = [StandardScaler()]
    if max_dim:
        steps.append(PCA(n_components=max_dim, random_state=seed))
    steps.append(
        # multinomial is sklearn's default for >2 classes since 1.5; passing
        # multi_class explicitly is deprecated and removed in 1.7.
        LogisticRegression(C=C, max_iter=3000, n_jobs=-1, random_state=seed)
    )
    return make_pipeline(*steps)


def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    C: float = 0.01,
    max_dim: Optional[int] = None,
    n_permutations: int = 20,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """Cross-validated accuracy, macro-F1, majority rate, and permutation null."""
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def cv_accuracy(labels: np.ndarray) -> float:
        preds = np.empty_like(labels)
        for tr, te in splitter.split(X, labels, groups):
            clf = _make_clf(C, max_dim, seed)
            clf.fit(X[tr], labels[tr])
            preds[te] = clf.predict(X[te])
        return accuracy_score(labels, preds), preds

    acc, preds = cv_accuracy(y)
    macro = f1_score(y, preds, average="macro", zero_division=0)
    _, counts = np.unique(y, return_counts=True)
    majority = counts.max() / len(y)

    nulls = []
    for _ in range(n_permutations):
        # Permute labels at the group level so the null keeps the same grouped
        # structure; permuting rows would leak item identity into the null.
        perm = _permute_within_groups(y, groups, rng)
        try:
            null_acc, _ = cv_accuracy(perm)
        except ValueError:
            continue
        nulls.append(null_acc)
    null_mean = float(np.mean(nulls)) if nulls else float("nan")
    null_std = float(np.std(nulls)) if nulls else float("nan")

    return acc, macro, majority, null_mean, null_std


def _permute_within_groups(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """Shuffle the label attached to each group, keeping group sizes intact.

    Assumes one label per group, which holds at n_samples=1 (greedy decoding,
    the default).  With multiple samples per item that disagree on the answer,
    this collapses the group to its first label, so the null is slightly
    optimistic; prefer per-sample groups in that regime.
    """
    uniq = np.unique(groups)
    labels_by_group = {g: y[groups == g][0] for g in uniq}
    shuffled = rng.permutation(list(labels_by_group.values()))
    mapping = dict(zip(uniq, shuffled))
    return np.array([mapping[g] for g in groups])


def sweep(
    acts: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[str],
    positions: Sequence[str],
    layers: Sequence[int],
    min_class_count: int = 10,
    min_rows: int = 50,
    **kw,
) -> list[ProbeResult]:
    """Fit a probe at every (position, layer).

    Classes rarer than `min_class_count` are dropped: a 10-way probe with
    singleton classes reports noise as structure.
    """
    y = np.asarray(labels)
    g = np.asarray(groups)

    keep = np.array([lab is not None and lab == lab for lab in y])
    vals, counts = np.unique(y[keep], return_counts=True)
    frequent = set(vals[counts >= min_class_count])
    keep &= np.array([lab in frequent for lab in y])

    if keep.sum() < min_rows or len(frequent) < 2:
        raise ValueError(
            f"not enough usable rows/classes after filtering: n={keep.sum()} "
            f"(need {min_rows}), classes={len(frequent)} (need 2). "
            f"Started from {len(y)} rows; {(~keep).sum()} dropped as unlabelled or "
            f"in classes rarer than {min_class_count}. Lower --min-class-count / "
            f"--min-rows for a debug run, or collect more items for a real one."
        )

    y, g = y[keep], g[keep]
    results: list[ProbeResult] = []
    for pi, pos in enumerate(positions):
        for li, layer in enumerate(layers):
            X = np.asarray(acts[keep, pi, li, :])
            acc, macro, majority, null_mean, null_std = fit_probe(X, y, g, **kw)
            z = (acc - null_mean) / null_std if null_std and null_std > 0 else float("nan")
            results.append(
                ProbeResult(
                    position=pos,
                    layer=int(layer),
                    n=int(len(y)),
                    n_classes=len(frequent),
                    accuracy=float(acc),
                    macro_f1=float(macro),
                    majority=float(majority),
                    null_mean=null_mean,
                    null_std=null_std,
                    delta_majority=float(acc - majority),
                    z_vs_null=float(z),
                )
            )
    return results
