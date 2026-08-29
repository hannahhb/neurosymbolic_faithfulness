#!/usr/bin/env python3
"""Tests for the pieces the TOOL / NO_TOOL / MALFORMED labels depend on.

Run with `python test_parsing.py` or `pytest test_parsing.py`.
"""

from __future__ import annotations

# Make `python run_phase0.py` work from inside the package directory.  Python
# puts only the *script's own* directory on sys.path, so `neurosymbolic_faithfulness.*`
# would not resolve.  Putting the parent on the path fixes that; running as
# `python -m neurosymbolic_faithfulness.run_phase0` from the parent also works.
import sys as _sys
from pathlib import Path as _Path

_PKG_PARENT = str(_Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in _sys.path:
    _sys.path.insert(0, _PKG_PARENT)


from neurosymbolic_faithfulness.calculator import run_calculator
from neurosymbolic_faithfulness.chat import extract_answer, parse_assistant

H = "hermes"


def _p(text, fmt=H, finish="stop"):
    return parse_assistant(text, fmt, finish)


# --- well-formed ------------------------------------------------------------
def test_hermes_wellformed():
    t = ('I should compute this.\n<tool_call>\n'
         '{"name": "calculator", "arguments": {"expression": "12 + 34"}}\n</tool_call>')
    r = _p(t)
    assert len(r.calls) == 1 and not r.malformed
    assert r.calls[0].arguments["expression"] == "12 + 34"
    assert r.prefix.strip() == "I should compute this."


def test_hermes_arguments_as_json_string():
    t = ('<tool_call>{"name": "calculator", "arguments": '
         '"{\\"expression\\": \\"2*3\\"}"}</tool_call>')
    r = _p(t)
    assert len(r.calls) == 1 and r.calls[0].arguments["expression"] == "2*3"


def test_hermes_two_calls():
    t = ('<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>\n'
         '<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>')
    assert len(_p(t).calls) == 2


def test_llama31_wellformed():
    t = '<|python_tag|>{"name": "calculator", "parameters": {"expression": "7*8"}}'
    r = _p(t, "llama31")
    assert len(r.calls) == 1 and r.calls[0].arguments["expression"] == "7*8"


def test_llama31_bare_json():
    r = _p('{"name": "calculator", "parameters": {"expression": "7*8"}}', "llama31")
    assert len(r.calls) == 1 and not r.malformed


# --- no tool ----------------------------------------------------------------
def test_no_tool():
    r = _p("Adding step by step gives 46.\n\n\\boxed{46}")
    assert not r.calls and not r.malformed
    assert r.prefix == r.text


def test_word_calculator_alone_is_not_a_call():
    # mentioning the tool is not calling it
    r = _p("I could use the calculator here, but 2+2 is easy.\n\\boxed{4}")
    assert not r.calls and not r.malformed


# --- malformed --------------------------------------------------------------
def test_malformed_bad_json():
    r = _p('<tool_call>\n{"name": "calculator", "arguments": {expression: 1+1}}\n</tool_call>')
    assert not r.calls and len(r.malformed) == 1
    assert "not valid JSON" in r.malformed[0].reason


def test_malformed_wrong_arg_key():
    r = _p('<tool_call>{"name":"calculator","arguments":{"expr":"1+1"}}</tool_call>')
    assert not r.calls and "expression" in r.malformed[0].reason


def test_malformed_wrong_tool_name():
    r = _p('<tool_call>{"name":"calc","arguments":{"expression":"1+1"}}</tool_call>')
    assert not r.calls and "unknown tool name" in r.malformed[0].reason


def test_malformed_untagged_json():
    r = _p('I will call it: {"name": "calculator", "arguments": {"expression": "1+1"}}')
    assert not r.calls and "outside <tool_call>" in r.malformed[0].reason


def test_malformed_pseudo_call():
    r = _p('Let me run calculator("123 + 456") to check.')
    assert not r.calls and "pseudo-call" in r.malformed[0].reason


def test_unclosed_tag_is_malformed_when_not_truncated():
    r = _p('<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+1"}}')
    assert not r.calls and "unclosed" in r.malformed[0].reason


def test_unclosed_tag_from_truncation_is_not_malformed():
    r = _p('<tool_call>\n{"name": "calculator", "argum',
           finish="length")
    assert not r.calls and not r.malformed and r.truncated_call


def test_prefix_stops_at_malformed_too():
    r = _p('Thinking hard.\ncalculator("1+1")')
    assert r.prefix.strip() == "Thinking hard."


# --- answer extraction ------------------------------------------------------
def test_extract_boxed():
    assert extract_answer("stuff \\boxed{-1,234}") == ("-1234", "boxed")


def test_extract_boxed_nested_braces():
    assert extract_answer("\\boxed{\\text{42}}")[0] == "42"


def test_extract_last_boxed_wins():
    assert extract_answer("\\boxed{1} then \\boxed{2}")[0] == "2"


def test_extract_answer_is():
    assert extract_answer("The final answer is 987.")[0] == "987"


def test_extract_last_number_fallback():
    a, how = extract_answer("I get 5 then 17")
    assert (a, how) == ("17", "last_number")


def test_extract_none():
    assert extract_answer("no numbers here") == (None, "none")


# --- sandbox ----------------------------------------------------------------
def test_sandbox_blocks_code():
    for bad in ['__import__("os").system("ls")', "open('/etc/passwd')", "x", "[1,2]",
                "'a'*3", "(lambda: 1)()", "1 if True else 2"]:
        assert not run_calculator(bad).ok, bad


def test_sandbox_arithmetic():
    assert run_calculator("143469773 - 378009742").value == "-234539969"
    assert run_calculator("1663 * 5242 * 9376").value == "81734773696"
    assert run_calculator("15 - 43 - 75 =").value == "-103"   # trailing '=' tolerated


def test_sandbox_errors_are_reported_not_raised():
    r = run_calculator("1/0")
    assert not r.ok and r.to_model_string() == "Error: division by zero"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}  {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}  {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
