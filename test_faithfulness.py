#!/usr/bin/env python3
"""Tests for the pieces the comparison depends on. Run: python test_faithfulness.py"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import numpy as np

from neurosymbolic_faithfulness import prompts as P
from neurosymbolic_faithfulness.execute import run_expression
from neurosymbolic_faithfulness.resample import (Embedder, distribution,
                                                 extract_answer,
                                                 extract_answer_detailed, kl)
from neurosymbolic_faithfulness.segment import prefix_for, split


# --- prompts ----------------------------------------------------------------
def test_exactly_two_conditions():
    assert P.CONDITIONS == ("cot", "code")


def test_prompt_text_is_verbatim():
    assert P.COT == "Let's think step by step"
    assert P.CODE == "write a python program for the following problem"


def test_both_prompts_contain_the_problem_unchanged():
    q = "A whale is 210 feet long. How long is it?"
    for c in P.CONDITIONS:
        assert q in P.build_messages(q, c)[1]["content"]


def test_prefix_is_appended_verbatim():
    class T:
        def apply_chat_template(self, m, add_generation_prompt=True, tokenize=False):
            return "PROMPT>"
    assert P.build_prompt(T(), "q", "cot", "partial trace") == "PROMPT>partial trace"


# --- segmentation -----------------------------------------------------------
def test_sentence_offsets_are_byte_exact():
    t = "First step. Second step here. Third one.\nFourth after newline."
    st = split(t, "cot")
    assert len(st) >= 4
    assert all(t[s.start:s.end].strip() == s.text for s in st)


def test_code_lines_attach_comments_to_next_statement():
    t = "a = 1\n\n# a comment\nb = 2\nprint(b)\n"
    st = split(t, "code")
    assert [s.text for s in st] == ["a = 1", "# a comment\nb = 2", "print(b)"]


def test_prefix_reconstructs_exactly():
    t = "One. Two. Three."
    st = split(t, "cot")
    assert prefix_for(t, st, 0) == ""
    assert t.startswith(prefix_for(t, st, 2))


def test_decimals_do_not_split_sentences():
    st = split("The value is 3.14 exactly. Done.", "cot")
    assert len(st) == 2


# --- answers ----------------------------------------------------------------
def test_extract_cot_prefers_boxed():
    assert extract_answer("blah 99 then \\boxed{20}", "cot") == "20"


def test_extract_code_executes_the_print():
    assert extract_answer("x = 7\nprint(72*7/12/210*100)", "code") == "20"


def test_extract_code_handles_fences():
    assert extract_answer("```python\nprint(504/12)\n```", "code") == "42"


def test_sandbox_still_blocks_escapes():
    for bad in ['__import__("os").system("ls")', "open('/etc/passwd')", "(1).__class__"]:
        assert not run_expression(bad).ok, bad


def test_cot_extraction_rule_hierarchy():
    """The cot prompt carries no answer-format instruction, so the fallback
    order matters and the rule that fired must be recorded."""
    assert extract_answer_detailed("we get \\boxed{20}", "cot") == ("20", "boxed")
    assert extract_answer_detailed("the final answer is 42.", "cot") == ("42", "answer_is")
    assert extract_answer_detailed("total = 7*72 = 504", "cot") == ("504", "trailing_equals")
    assert extract_answer_detailed("3 then 17", "cot") == ("17", "last_number")
    assert extract_answer_detailed("no digits", "cot") == (None, "none")


def test_boxed_beats_a_later_stray_number():
    a, rule = extract_answer_detailed("\\boxed{20} (that took 5 steps)", "cot")
    assert (a, rule) == ("20", "boxed")


def test_answer_rules_are_recorded_per_step():
    from neurosymbolic_faithfulness.resample import StepResult
    assert "answer_rules" in StepResult.__dataclass_fields__


def test_precomputed_cot_metrics_are_not_reused_across_models():
    """Pairing R1-Distill CoT metrics with a Qwen2.5 code arm would compare two
    models, not two conditions. run.py must refuse the mismatch."""
    src = (Path(__file__).resolve().parent / "run.py").read_text()
    assert "a.model.split" in src and "mr_model" in src
    assert "regenerating the CoT arm" in src


# --- divergence -------------------------------------------------------------
def test_kl_zero_for_identical():
    assert kl({"20": 1.0}, {"20": 1.0}) == 0.0


def test_kl_positive_and_ordered():
    near = kl({"20": 0.6, "42": 0.4}, {"20": 1.0})
    far = kl({"42": 1.0}, {"20": 1.0})
    assert 0 < near < far


def test_kl_is_finite_on_disjoint_support():
    assert np.isfinite(kl({"1": 1.0}, {"2": 1.0}))


def test_distribution_drops_unreadable_and_renormalises():
    p = distribution(["20", None, "20", "42"])
    assert abs(p["20"] - 2 / 3) < 1e-9 and abs(p["42"] - 1 / 3) < 1e-9
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_unreadable_rollouts_are_counted():
    """A biased subset must be visible in the record, not silent."""
    from neurosymbolic_faithfulness.resample import StepResult
    assert "n_unreadable" in StepResult.__dataclass_fields__
    assert "unreadable_fraction" in StepResult.__dataclass_fields__


def test_embedder_hash_is_deterministic_and_ordered():
    e = Embedder("hash")
    s = e.similarity("the total is 504 inches",
                     ["the total is 504 inches", "bananas are yellow"])
    assert s[0] > 0.99 and s[1] < 0.5


# --- probe ------------------------------------------------------------------
def test_probe_recovers_planted_positional_signal():
    from neurosymbolic_faithfulness.probe import ProbeExample, train_probes

    rng = np.random.default_rng(0)
    X, ex = [], []
    for prob in range(8):
        for r in range(12):
            lab = int(rng.random() < 0.5)
            n = 10
            for i in range(n):
                rel = i / (n - 1)
                X.append(np.stack([rng.normal(lab * rel * 3.0, 1, 16),
                                   rng.normal(0, 1, 16)]))
                ex.append(ProbeExample(f"p{prob}", f"c{prob}", "cot", r, i, n,
                                       rel, lab, str(lab)))
    rows = train_probes(np.stack(X), ex, n_bins=5)
    early = [r["auc_mean"] for r in rows if r["layer"] == 0 and r["bin"] == 0]
    late = [r["auc_mean"] for r in rows if r["layer"] == 0 and r["bin"] == 4]
    noise = [r["auc_mean"] for r in rows if r["layer"] == 1]
    assert late[0] > early[0] + 0.2, (early, late)
    assert abs(np.mean(noise) - 0.5) < 0.08, np.mean(noise)


def test_probe_cv_groups_by_problem():
    """Rollouts of one problem must never straddle a CV split."""
    from neurosymbolic_faithfulness.probe import ProbeExample, train_probes

    rng = np.random.default_rng(1)
    X, ex = [], []
    for prob in range(2):                       # only 2 groups -> too few
        for r in range(20):
            lab = prob
            X.append(np.stack([rng.normal(lab, 1, 8)]))
            ex.append(ProbeExample(f"p{prob}", f"c{prob}", "cot", r, 0, 1, 0.0,
                                   lab, str(lab)))
    assert train_probes(np.stack(X), ex, n_bins=1) == []


# --- data -------------------------------------------------------------------
def test_gsm_symbolic_loads_and_spreads_across_templates():
    from neurosymbolic_faithfulness.data import load_gsm_symbolic
    ps = load_gsm_symbolic(8, seed=1)
    assert len(ps) == 8
    assert len({p.cluster_id for p in ps}) == 8
    assert all(p.answer and p.answer.lstrip("-").replace(".", "").isdigit() for p in ps)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1; print(f"  FAIL  {name}  {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"  ERROR {name}  {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
