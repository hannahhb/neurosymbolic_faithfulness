"""Linear probes: is the final answer already decodable before the trace ends?

Construction.  For one problem we sample M complete rollouts per condition.
Each rollout is segmented into steps.  At every step boundary we take the
residual stream at the last prompt token, for every layer.  The label is that
*same rollout's* final answer, binarised.  Different rollouts diverge, so the
label genuinely varies at a given boundary -- which is what makes the probe
informative rather than a constant.

Reading it.  Plot AUC against relative position (i / n_steps).  If AUC is
already high near position 0, the answer was fixed before the visible reasoning
happened, and that reasoning is post-hoc.  The comparison of interest is the
CoT curve against the code curve: whichever rises later is the arm whose visible
trace is doing more of the causal work.

Two things kept honest here:
  * cross-validation is grouped by problem, never by rollout -- rollouts of one
    problem share a prompt and would leak straight across a random split.
  * a majority-class baseline is reported alongside every AUC, because a problem
    whose rollouts nearly all agree makes a trivially high AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

from neurosymbolic_faithfulness import prompts as P
from neurosymbolic_faithfulness.resample import extract_answer
from neurosymbolic_faithfulness.segment import split


@dataclass
class ProbeExample:
    problem_id: str
    cluster_id: str
    condition: str
    rollout_idx: int
    step_idx: int
    n_steps: int
    rel_pos: float
    label: int
    final_answer: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_hidden(model, tokenizer, prompts: Sequence[str],
                   batch_size: int = 8, layers: Sequence[int] | None = None
                   ) -> np.ndarray:
    """Residual stream at the final token of each prompt.

    Returns [n_prompts, n_layers, d_model] as float32 on CPU.
    """
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = []
    for s in range(0, len(prompts), batch_size):
        batch = list(prompts[s: s + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                    # tuple[n_layers+1] of [b, t, d]
        sel = range(len(hs)) if layers is None else layers
        # left padding => the final token is the last position for every row
        stack = torch.stack([hs[i][:, -1, :] for i in sel], dim=1)
        chunks.append(stack.float().cpu().numpy())
        del out, hs
    return np.concatenate(chunks, axis=0)


def build_probe_dataset(engine, model, problems, condition: str,
                        n_rollouts: int = 16, temperature: float = 0.6,
                        max_tokens: int = 512, seed: int = 0,
                        label_mode: str = "correct", log=print):
    """Generate rollouts, segment them, and capture a hidden state per boundary.

    label_mode:
      "correct"  -- 1 if the rollout's final answer matches ground truth
      "modal"    -- 1 if it matches the problem's most common answer across
                    rollouts (usable when accuracy is near 0 or 1)
    """
    tok = engine.tokenizer
    examples: list[ProbeExample] = []
    prompts_to_embed: list[str] = []

    for p in problems:
        log(f"  {p.problem_id} ({condition})")
        base = P.build_prompt(tok, p.question, condition)
        outs = engine.generate([base] * n_rollouts, temperature=temperature,
                               max_tokens=max_tokens,
                               seeds=[seed + k for k in range(n_rollouts)], stop=[])
        finals = [extract_answer(o.text, condition) for o in outs]
        modal = None
        vals = [f for f in finals if f is not None]
        if vals:
            modal = max(set(vals), key=vals.count)

        for r, (o, fin) in enumerate(zip(outs, finals)):
            steps = split(o.text, condition)
            if not steps:
                continue
            label = int(fin == p.answer) if label_mode == "correct" else int(
                fin is not None and fin == modal)
            for st in steps:
                prompts_to_embed.append(
                    P.build_prompt(tok, p.question, condition, o.text[: st.start]))
                examples.append(ProbeExample(
                    problem_id=p.problem_id, cluster_id=p.cluster_id,
                    condition=condition, rollout_idx=r, step_idx=st.idx,
                    n_steps=len(steps),
                    rel_pos=st.idx / max(1, len(steps) - 1) if len(steps) > 1 else 0.0,
                    label=label, final_answer=fin))

    X = capture_hidden(model, tok, prompts_to_embed)
    return X, examples


def train_probes(X: np.ndarray, examples: list[ProbeExample], n_bins: int = 5,
                 seed: int = 0) -> list[dict]:
    """One probe per (layer, relative-position bin), grouped CV by problem."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y = np.array([e.label for e in examples])
    groups = np.array([e.problem_id for e in examples])
    rel = np.array([e.rel_pos for e in examples])
    bins = np.clip((rel * n_bins).astype(int), 0, n_bins - 1)

    rows = []
    for layer in range(X.shape[1]):
        for b in range(n_bins):
            m = bins == b
            yb, gb = y[m], groups[m]
            if m.sum() < 20 or len(set(yb)) < 2 or len(set(gb)) < 3:
                continue
            Xb = X[m, layer, :]
            n_splits = min(5, len(set(gb)))
            aucs = []
            for tr, te in GroupKFold(n_splits=n_splits).split(Xb, yb, gb):
                if len(set(yb[tr])) < 2 or len(set(yb[te])) < 2:
                    continue
                sc = StandardScaler().fit(Xb[tr])
                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
                clf.fit(sc.transform(Xb[tr]), yb[tr])
                s = clf.predict_proba(sc.transform(Xb[te]))[:, 1]
                aucs.append(roc_auc_score(yb[te], s))
            if not aucs:
                continue
            rows.append({
                "layer": layer, "bin": b,
                "rel_pos_lo": b / n_bins, "rel_pos_hi": (b + 1) / n_bins,
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "n": int(m.sum()), "n_problems": len(set(gb)),
                "majority_baseline": float(max(yb.mean(), 1 - yb.mean())),
                "positive_rate": float(yb.mean()),
            })
    return rows
