"""Sandboxed execution of the SYM condition's expressions.

Extends the Phase-0 calculator sandbox with a small allowlist of pure numeric
functions, because GSM8K-style problems occasionally need rounding or a
min/max.  Still no `eval`: the expression is parsed with ast.parse(mode="eval")
and every node type is checked against an allowlist, and calls are dispatched to
a fixed dict of Python builtins rather than resolved from any namespace.
"""

from __future__ import annotations

import ast
import math
import sys
from dataclasses import dataclass, asdict
from typing import Any

MAX_EXPR_CHARS = 2000
MAX_POW_EXPONENT = 64
MAX_OPERAND_BITS = 4096
MAX_NODES = 2000

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(8000)

# Pure, side-effect-free, no attribute access. Dispatch is by name against this
# dict only -- the expression can never reach a real namespace.
ALLOWED_FUNCS: dict[str, Any] = {
    "round": round,
    "int": int,
    "float": float,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call, ast.Name,
    ast.Load, ast.Tuple, ast.List,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)


class ExecError(Exception):
    pass


@dataclass
class ExecResult:
    ok: bool
    expression: str
    value: str | None
    numeric: float | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(v: Any) -> Any:
    if isinstance(v, bool):
        raise ExecError("boolean literals are not allowed")
    if isinstance(v, int) and v.bit_length() > MAX_OPERAND_BITS:
        raise ExecError("operand too large")
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        raise ExecError("non-finite float")
    if not isinstance(v, (int, float)):
        raise ExecError(f"unsupported value of type {type(v).__name__}")
    return v


def _eval(node: ast.AST, budget: list[int]) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise ExecError("expression too complex")
    if not isinstance(node, _ALLOWED_NODES):
        raise ExecError(f"disallowed syntax: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _eval(node.body, budget)
    if isinstance(node, ast.Constant):
        return _check(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(x, budget) for x in node.elts]
    if isinstance(node, ast.Name):
        raise ExecError(f"undefined name {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExecError("only direct calls to allowed functions")
        fn = ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ExecError(f"function {node.func.id!r} is not allowed")
        if node.keywords:
            raise ExecError("keyword arguments are not allowed")
        args = [_eval(a, budget) for a in node.args]
        try:
            return _check(fn(*args))
        except ExecError:
            raise
        except Exception as exc:
            raise ExecError(f"{node.func.id}: {exc}") from None
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, budget)
        return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, budget), _eval(node.right, budget)
        op = node.op
        try:
            if isinstance(op, ast.Add):
                r = a + b
            elif isinstance(op, ast.Sub):
                r = a - b
            elif isinstance(op, ast.Mult):
                r = a * b
            elif isinstance(op, ast.Div):
                r = a / b
            elif isinstance(op, ast.FloorDiv):
                r = a // b
            elif isinstance(op, ast.Mod):
                r = a % b
            elif isinstance(op, ast.Pow):
                if abs(b) > MAX_POW_EXPONENT:
                    raise ExecError("exponent too large")
                r = a**b
            else:
                raise ExecError(f"disallowed operator: {type(op).__name__}")
        except ZeroDivisionError:
            raise ExecError("division by zero") from None
        except OverflowError:
            raise ExecError("numeric overflow") from None
        return _check(r)
    raise ExecError(f"disallowed syntax: {type(node).__name__}")


def _format(v: Any) -> str:
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def run_expression(expr: Any) -> ExecResult:
    """Evaluate a single arithmetic expression. Never raises."""
    if not isinstance(expr, str):
        return ExecResult(False, str(expr), None, None, "expression must be a string")
    e = expr.strip().rstrip("=").strip()
    if not e:
        return ExecResult(False, str(expr), None, None, "empty expression")
    if len(e) > MAX_EXPR_CHARS:
        return ExecResult(False, expr, None, None, "expression too long")
    try:
        tree = ast.parse(e, mode="eval")
    except SyntaxError as exc:
        return ExecResult(False, expr, None, None, f"syntax error: {exc.msg}")
    try:
        v = _eval(tree, [MAX_NODES])
    except ExecError as exc:
        return ExecResult(False, expr, None, None, str(exc))
    except RecursionError:
        return ExecResult(False, expr, None, None, "too deeply nested")
    if isinstance(v, list):
        return ExecResult(False, expr, None, None, "expression returned a sequence")
    return ExecResult(True, expr, _format(v), float(v), None)
