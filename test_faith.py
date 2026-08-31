#!/usr/bin/env python3
"""Tests for the pieces the 2x2 result depends on.

Run: python test_faith.py   (or pytest test_faith.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from neurosymbolic_faithfulness import prompts as P
from neurosymbolic_faithfulness.data import build_items, choose_distractor
from neurosymbolic_faithfulness.execute import run_expression


# --- sandbox ----------------------------------------------------------------
def test_sandbox_real_gsm_expressions():
    assert run_expression("700000 - 700000*3/100 - 700000*4/100 - 560000").value == "91000"
    assert run_expression("85*30*(12/2)").value == "15300"
    assert run_expression("236/2 + 972").value == "1090"


def test_sandbox_allows_pure_helpers():
    for e, want in [("round(10/3)", "3"), ("int(7/2)", "3"), ("max(3,9)-min(1,4)", "8"),
                    ("floor(7/2)+ceil(1/3)", "4"), ("sum([1,2,3])", "6")]:
        assert run_expression(e).value == want, e


def test_sandbox_blocks_escapes():
    for bad in ['__import__("os").system("ls")', 'open("/etc/passwd")', 'exec("y=1")',
                "(1).__class__", "lambda: 1", "x+1", "[1,2,3]"]:
        assert not run_expression(bad).ok, bad


def test_sandbox_errors_do_not_raise():
    r = run_expression("1/0")
    assert not r.ok and "division by zero" in r.error


# --- distractors ------------------------------------------------------------
def test_distractor_is_never_the_answer():
    items, _ = build_items(120, 7)
    assert all(i.distractor_value != i.answer_value for i in items)


def test_distractor_prefers_computed_intermediates():
    items, _ = build_items(120, 7)
    frac = sum(i.distractor_kind == "intermediate" for i in items) / len(items)
    assert frac > 0.9, frac


def test_distractor_falls_back_when_metadata_is_empty():
    x, kind = choose_distractor("q with 5", answer=48.0, variables={}, answer_cot="")
    assert kind == "perturbed" and x != 48.0


def test_template_id_is_recorded():
    items, _ = build_items(20, 7)
    assert all(isinstance(i.template_id, int) for i in items)
    assert len({i.template_id for i in items}) > 1


# --- prompt invariants ------------------------------------------------------
def _item():
    return build_items(1, 7)[0][0]


def test_bias_sentence_is_identical_across_modes():
    it = _item()
    nl = P.build_messages(it, "nl", "wrong")[1]["content"]
    sym = P.build_messages(it, "sym", "wrong")[1]["content"]
    hint = P.bias_text("wrong", it)
    assert hint in nl and hint in sym and it.distractor in hint


def test_only_the_mode_instruction_differs():
    it = _item()
    nl = P.build_messages(it, "nl", "wrong")[1]["content"]
    sym = P.build_messages(it, "sym", "wrong")[1]["content"]
    assert nl.replace(P.MODE_INSTRUCTION["nl"], "") == sym.replace(
        P.MODE_INSTRUCTION["sym"], "")


def test_unbiased_prompt_has_no_hint():
    it = _item()
    msg = P.build_messages(it, "nl", "none")[1]["content"]
    assert "colleague" not in msg and it.distractor not in msg


def test_contentless_hint_carries_no_number():
    it = _item()
    msg = P.build_messages(it, "nl", "contentless")[1]["content"]
    assert "colleague" in msg and it.distractor not in msg


# --- extraction -------------------------------------------------------------
def test_sym_answer_comes_from_execution_not_assertion():
    """The model claims 33; its expression computes 57. Execution must win."""
    ex = P.extract("<expr>11*3 + 4*6</expr> so the answer is \\boxed{33}", "sym")
    assert ex.answer == "57" and ex.asserted == "33" and ex.exec_ok


def test_sym_records_format_failures():
    assert P.extract("the answer is 57", "sym").method == "no_expr_tags"
    assert P.extract("<expr>foo(1)</expr>", "sym").method == "exec_failed"


def test_nl_extraction():
    assert P.extract("...\\boxed{-1,234}", "nl").answer == "-1234"
    assert P.extract("I get 5 then 17", "nl").method == "last_number"


def test_answers_match_is_numeric():
    assert P.answers_match("57", "57.0") and not P.answers_match("57", "33")
    assert not P.answers_match(None, "57")


# --- the metric recovers a planted effect -----------------------------------
def test_susceptibility_recovers_planted_bias_rate():
    from neurosymbolic_faithfulness.analyze import analyse
    from neurosymbolic_faithfulness.engine import MockEngine
    from neurosymbolic_faithfulness.run import run_cells

    items, _ = build_items(40, 7)
    eng = MockEngine(seed=0, bias_rate=0.4)
    recs = run_cells(eng, items, [("nl", "none"), ("nl", "wrong")], n_samples=4,
                     temperature=0.7, max_tokens=32, seed=1, log=lambda *_: None)
    d = analyse(recs)["susceptibility"]["nl"]["delta_p_equals_X"]
    assert 0.30 <= d <= 0.50, d


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
