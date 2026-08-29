"""The single tool exposed to the model: a sandboxed arithmetic evaluator.

Security model: we never call `eval` on the model's string. We parse the
expression with `ast.parse(mode="eval")` and walk the tree, refusing any node
type that is not pure arithmetic. Names, calls, attributes, subscripts,
comprehensions, f-strings and string/bytes constants are all rejected, so there
is no reachable path to the interpreter's namespace.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, asdict
from typing import Any

# --- limits -----------------------------------------------------------------
MAX_EXPR_CHARS = 512        # reject absurdly long expressions outright
MAX_POW_EXPONENT = 64       # stop `9**9**9`-style resource exhaustion
MAX_OPERAND_BITS = 4096     # ~1233 decimal digits; far above any task operand
MAX_NODES = 500             # bound on tree size

# Python >= 3.11 refuses to str() huge ints; raise the cap a little so that a
# legitimately large but bounded result can still be returned to the model.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(8000)

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    # binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # unary operators
    ast.UAdd,
    ast.USub,
)


class CalculatorError(Exception):
    """Raised for any expression we refuse to, or cannot, evaluate."""


@dataclass
class ToolResult:
    ok: bool
    expression: str
    value: str | None          # stringified result, as returned to the model
    error: str | None          # human-readable error, as returned to the model

    def to_model_string(self) -> str:
        """Exactly the text that is fed back into the conversation."""
        return self.value if self.ok else f"Error: {self.error}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_operand(v: Any) -> int | float:
    if isinstance(v, bool):  # bool is a subclass of int; not arithmetic input
        raise CalculatorError("boolean literals are not allowed")
    if not isinstance(v, (int, float)):
        raise CalculatorError(f"unsupported constant of type {type(v).__name__}")
    if isinstance(v, int) and v.bit_length() > MAX_OPERAND_BITS:
        raise CalculatorError("operand too large")
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        raise CalculatorError("non-finite float")
    return v


def _eval(node: ast.AST, budget: list[int]) -> int | float:
    budget[0] -= 1
    if budget[0] < 0:
        raise CalculatorError("expression too complex")
    if not isinstance(node, _ALLOWED_NODES):
        raise CalculatorError(f"disallowed syntax: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _eval(node.body, budget)

    if isinstance(node, ast.Constant):
        return _check_operand(node.value)

    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, budget)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise CalculatorError(f"disallowed unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _eval(node.left, budget)
        right = _eval(node.right, budget)
        op = node.op
        try:
            if isinstance(op, ast.Add):
                out = left + right
            elif isinstance(op, ast.Sub):
                out = left - right
            elif isinstance(op, ast.Mult):
                out = left * right
            elif isinstance(op, ast.Div):
                out = left / right
            elif isinstance(op, ast.FloorDiv):
                out = left // right
            elif isinstance(op, ast.Mod):
                out = left % right
            elif isinstance(op, ast.Pow):
                if abs(right) > MAX_POW_EXPONENT:
                    raise CalculatorError("exponent too large")
                out = left**right
            else:
                raise CalculatorError(
                    f"disallowed binary operator: {type(op).__name__}"
                )
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("numeric overflow") from None
        return _check_operand(out)

    raise CalculatorError(f"disallowed syntax: {type(node).__name__}")


def _format(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def run_calculator(expression: Any) -> ToolResult:
    """Evaluate `expression`. Never raises; failures come back as ToolResult."""
    if not isinstance(expression, str):
        return ToolResult(False, str(expression), None,
                          "expression must be a string")
    expr = expression.strip()
    if not expr:
        return ToolResult(False, expression, None, "empty expression")
    if len(expr) > MAX_EXPR_CHARS:
        return ToolResult(False, expression, None,
                          f"expression longer than {MAX_EXPR_CHARS} characters")
    # A trailing '=' is a very common model habit ("1+2="); strip it rather than
    # counting it as a model error, and record nothing special.
    if expr.endswith("="):
        expr = expr[:-1].strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return ToolResult(False, expression, None, f"syntax error: {exc.msg}")
    try:
        value = _eval(tree, [MAX_NODES])
    except CalculatorError as exc:
        return ToolResult(False, expression, None, str(exc))
    except RecursionError:
        return ToolResult(False, expression, None, "expression too deeply nested")
    return ToolResult(True, expression, _format(value), None)


# --- tool schema handed to the chat template --------------------------------
# Deliberately neutral: it says what the tool does, never when to use it.
CALCULATOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluates a Python arithmetic expression and returns the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python arithmetic expression, for example \"12345 + 6789\".",
                }
            },
            "required": ["expression"],
        },
    },
}

TOOL_NAME = "calculator"
TOOLS = [CALCULATOR_TOOL]
