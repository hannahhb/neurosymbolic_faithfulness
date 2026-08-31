"""Counterfactual importance by resampling (Bogdan, Macar, Nanda & Conmy 2025).

For each step i of a trace, continue generation from the prefix ending just
before step i, N times.  The first step of each continuation is the replacement
for step i.  Partition the continuations by whether that replacement is
semantically similar to the original step (all-MiniLM-L6-v2 cosine >= 0.8), then
compare the resulting final-answer distributions:

    resampling importance      D_KL[ p(A' | any resample)        || p(A | kept) ]
    counterfactual importance  D_KL[ p(A' | replacement differs) || p(A | kept) ]

The counterfactual version is the one that isolates the step's content: if
replacing the step with something that *means something else* barely moves the
answer distribution, the step was not doing causal work.

The same machinery runs on both arms.  The only difference is the unit --
sentences for CoT, statements for code -- which `segment.split` handles.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from neurosymbolic_faithfulness import prompts as P
from neurosymbolic_faithfulness.execute import run_expression
from neurosymbolic_faithfulness.segment import Step, split

SIM_THRESHOLD = 0.8          # paper's median cosine over sentence pairs
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cue appended to a prefix to force immediate answer emission.
FORCE_CUE = {
    "cot": "\n\nTherefore, the final answer is",
    "code": "\n\n# The final answer is",
}


# --- embeddings -------------------------------------------------------------
class Embedder:
    """all-MiniLM-L6-v2. `backend="hash"` is a deterministic bag-of-words
    stand-in for offline tests only -- runs record which backend was used, so a
    test run can never be mistaken for a real one."""

    def __init__(self, backend: str = "minilm"):
        self.backend = backend
        self._m = None
        if backend == "minilm":
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(EMBED_MODEL)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "minilm":
            v = self._m.encode(list(texts), normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)
        vs = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[hash(w) % 256] += 1.0
            n = np.linalg.norm(v)
            vs.append(v / n if n else v)
        return np.stack(vs)

    def similarity(self, a: str, bs: Sequence[str]) -> np.ndarray:
        if not bs:
            return np.zeros(0, dtype=np.float32)
        e = self.encode([a] + list(bs))
        return e[1:] @ e[0]


# --- answer handling --------------------------------------------------------
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


_ANSWER_IS = re.compile(
    r"(?:final answer|answer)\s*(?:is|:|=)\s*\**\s*\$?(-?[\d,]+(?:\.\d+)?)", re.I)
_TRAILING_EQ = re.compile(r"=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\.?\s*$")


def extract_answer_detailed(trace: str, condition: str) -> tuple[str | None, str]:
    """Returns (answer, rule).  The rule is recorded because the CoT prompt is
    just "Let's think step by step" -- there is no answer-format instruction, so
    the fallbacks matter and a run dominated by `last_number` should be treated
    with suspicion rather than read as a clean measurement.
    """
    if condition == "code":
        a = extract_answer(trace, condition)
        return a, ("executed" if a is not None else "none")
    b = _BOXED.findall(trace)
    if b:
        n = _NUM.findall(b[-1])
        if n:
            return n[-1].replace(",", ""), "boxed"
    m = list(_ANSWER_IS.finditer(trace))
    if m:
        return m[-1].group(1).replace(",", ""), "answer_is"
    tail = trace.strip().splitlines()[-1] if trace.strip() else ""
    m2 = _TRAILING_EQ.search(tail)
    if m2:
        return m2.group(1).replace(",", ""), "trailing_equals"
    n = _NUM.findall(trace)
    if n:
        return n[-1].replace(",", ""), "last_number"
    return None, "none"


def extract_answer(trace: str, condition: str) -> str | None:
    """CoT: boxed, then "answer is", then a trailing '= N', then the last
    number. Code: execute the program's final print expression."""
    if condition == "code":
        body = _CODE_FENCE.search(trace)
        src = body.group(1) if body else trace
        lines = [l for l in src.splitlines() if l.strip()]
        for line in reversed(lines):
            m = re.search(r"print\s*\((.+)\)\s*$", line.strip())
            expr = m.group(1) if m else None
            if expr:
                r = run_expression(expr)
                if r.ok:
                    return r.value
        nums = _NUM.findall(src)
        return nums[-1].replace(",", "") if nums else None
    return extract_answer_detailed(trace, condition)[0]


def distribution(answers: Sequence[str | None]) -> dict[str, float]:
    """Unreadable answers are dropped and the rest renormalised.  Callers must
    record how many were dropped -- a step whose rollouts are mostly
    unparseable yields a distribution over a biased subset."""
    vals = [a for a in answers if a is not None]
    if not vals:
        return {}
    c = Counter(vals)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-3) -> float:
    """D_KL[p || q] with additive smoothing over the union of supports, so a
    zero in q cannot produce an infinite divergence."""
    if not p or not q:
        return float("nan")
    keys = sorted(set(p) | set(q))
    pv = np.array([p.get(k, 0.0) + eps for k in keys])
    qv = np.array([q.get(k, 0.0) + eps for k in keys])
    pv, qv = pv / pv.sum(), qv / qv.sum()
    return float(np.sum(pv * np.log(pv / qv)))


# --- the measurement --------------------------------------------------------
@dataclass
class StepResult:
    step_idx: int
    text: str
    n_rollouts: int
    n_kept: int
    n_different: int
    n_unreadable: int          # rollouts whose final answer could not be parsed
    unreadable_fraction: float
    different_trajectories_fraction: float
    resampling_importance_kl: float
    counterfactual_importance_kl: float
    resampling_importance_accuracy: float
    counterfactual_importance_accuracy: float
    accuracy: float
    answers_kept: dict
    answers_different: dict
    answer_rules: dict         # which extraction rule fired, and how often

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def resample_step(engine, embedder: Embedder, question: str, condition: str,
                  trace: str, steps: list[Step], i: int, gt_answer: str,
                  n_rollouts: int, temperature: float, max_tokens: int,
                  seed: int) -> StepResult:
    prefix = trace[: steps[i].start]
    prompt = P.build_prompt(engine.tokenizer, question, condition, prefix)
    outs = engine.generate([prompt] * n_rollouts, temperature=temperature,
                           max_tokens=max_tokens,
                           seeds=[seed + k for k in range(n_rollouts)], stop=[])

    replacements, answers, rules = [], [], []
    for o in outs:
        cont = o.text
        sub = split(cont, condition)
        replacements.append(sub[0].text if sub else "")
        a, rule = extract_answer_detailed(prefix + cont, condition)
        answers.append(a)
        rules.append(rule)

    sims = embedder.similarity(steps[i].text, replacements)
    kept_idx = [j for j, s in enumerate(sims) if s >= SIM_THRESHOLD]
    diff_idx = [j for j, s in enumerate(sims) if s < SIM_THRESHOLD]

    p_kept = distribution([answers[j] for j in kept_idx])
    p_diff = distribution([answers[j] for j in diff_idx])
    p_all = distribution(answers)

    def acc(idx):
        vals = [answers[j] == gt_answer for j in idx]
        return float(np.mean(vals)) if vals else float("nan")

    return StepResult(
        step_idx=i, text=steps[i].text, n_rollouts=len(outs),
        n_kept=len(kept_idx), n_different=len(diff_idx),
        n_unreadable=sum(a is None for a in answers),
        unreadable_fraction=sum(a is None for a in answers) / max(1, len(answers)),
        different_trajectories_fraction=len(diff_idx) / max(1, len(outs)),
        resampling_importance_kl=kl(p_all, p_kept),
        counterfactual_importance_kl=kl(p_diff, p_kept),
        resampling_importance_accuracy=acc(range(len(answers))) - acc(kept_idx)
        if kept_idx else float("nan"),
        counterfactual_importance_accuracy=acc(diff_idx) - acc(kept_idx)
        if kept_idx and diff_idx else float("nan"),
        accuracy=acc(range(len(answers))),
        answers_kept=p_kept, answers_different=p_diff,
        answer_rules=dict(Counter(rules)),
    )


def resample_trace(engine, embedder: Embedder, question: str, condition: str,
                   trace: str, gt_answer: str, n_rollouts: int = 100,
                   temperature: float = 0.6, max_tokens: int = 512,
                   seed: int = 0, max_steps: int | None = None,
                   log=print) -> list[StepResult]:
    steps = split(trace, condition)
    if max_steps:
        steps = steps[:max_steps]
    out = []
    for i in range(len(steps)):
        log(f"    step {i + 1}/{len(steps)} ({condition})")
        out.append(resample_step(engine, embedder, question, condition, trace,
                                 steps, i, gt_answer, n_rollouts, temperature,
                                 max_tokens, seed + 10_000 * i))
    return out


def forced_answer_distribution(engine, question: str, condition: str, trace: str,
                               steps: list[Step], i: int, n_rollouts: int,
                               temperature: float, seed: int) -> dict[str, float]:
    """Interrupt at the boundary before step i and force an answer immediately."""
    prefix = trace[: steps[i].start] + FORCE_CUE[condition]
    prompt = P.build_prompt(engine.tokenizer, question, condition, prefix)
    outs = engine.generate([prompt] * n_rollouts, temperature=temperature,
                           max_tokens=24,
                           seeds=[seed + k for k in range(n_rollouts)], stop=[])
    return distribution([extract_answer(o.text, "cot") for o in outs])
