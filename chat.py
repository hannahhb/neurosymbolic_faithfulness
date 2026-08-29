"""Prompt construction and tool-call parsing.

Two things live here because both must stay byte-exact across phases:

1. `build_prompt` produces the *string* that is fed to the sampler.  The later
   activation work (variance decomposition, prefix cross-grafting) has to run
   forward passes on exactly this string, so nothing else in the codebase is
   allowed to assemble prompts.
2. `parse_assistant` decides what counts as a well-formed tool call.  The
   TOOL / NO_TOOL / MALFORMED classification is only as trustworthy as this
   function, so it reports *why* something was malformed rather than silently
   dropping it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from typing import Any

from neurosymbolic_faithfulness.calculator import TOOL_NAME, TOOLS

# Neutral. It must not hint at when, or whether, to use the tool.
SYSTEM_PROMPT = "You are a helpful assistant."

# Appended to every question in *both* conditions, so it cannot bias the tool
# decision; it exists only to make answer extraction reliable.
ANSWER_SUFFIX = "\n\nPut your final answer in \\boxed{}."


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def build_messages(question: str, answer_suffix: str = ANSWER_SUFFIX) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question + answer_suffix},
    ]


def build_prompt(tokenizer, messages: list[dict], with_tools: bool) -> str:
    """The exact prompt string. `with_tools=False` removes the tool from the
    context entirely -- this is the forced-no-tool condition, not a prompt that
    tells the model to abstain."""
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS if with_tools else None,
        add_generation_prompt=True,
        tokenize=False,
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# tool-call parsing
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str
    start: int          # character offset in the assistant turn

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Malformed:
    reason: str
    raw: str
    start: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedTurn:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    malformed: list[Malformed] = field(default_factory=list)
    truncated_call: bool = False   # ran out of tokens mid-call; not the model's fault

    @property
    def prefix(self) -> str:
        """Text emitted before the first call attempt of any kind.

        This is the 'reasoning before the decision' that Phase 0's hand-read and
        the later grafting experiment both operate on.
        """
        starts = [c.start for c in self.calls] + [m.start for m in self.malformed]
        return self.text[: min(starts)] if starts else self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "calls": [c.to_dict() for c in self.calls],
            "malformed": [m.to_dict() for m in self.malformed],
            "truncated_call": self.truncated_call,
        }


_HERMES_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_HERMES_OPEN = re.compile(r"<tool_call>")
# a JSON object that looks like a call but sits outside the tags
_LOOSE_JSON = re.compile(r'\{[^{}]*"name"\s*:\s*"[A-Za-z_][A-Za-z_0-9]*"[^{}]*(?:\{[^{}]*\}[^{}]*)?\}')
# a plain-text pseudo-call, e.g. calculator("1+2")
_PSEUDO = re.compile(r"\b" + re.escape(TOOL_NAME) + r"\s*\(")
_PYTHON_TAG = "<|python_tag|>"


def _validate(obj: Any, raw: str, start: int) -> ToolCall | Malformed:
    """Turn a decoded JSON blob into a ToolCall, or say why it is not one."""
    if not isinstance(obj, dict):
        return Malformed("call payload is not a JSON object", raw, start)
    name = obj.get("name")
    if name is None:
        return Malformed("call has no 'name' field", raw, start)
    if name != TOOL_NAME:
        return Malformed(f"unknown tool name {name!r}", raw, start)
    args = obj.get("arguments", obj.get("parameters"))
    if args is None:
        return Malformed("call has no 'arguments' field", raw, start)
    if isinstance(args, str):
        # some models emit arguments as a JSON-encoded string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return Malformed("'arguments' is a string that is not JSON", raw, start)
    if not isinstance(args, dict):
        return Malformed("'arguments' is not an object", raw, start)
    if "expression" not in args:
        return Malformed("'arguments' has no 'expression' key", raw, start)
    if not isinstance(args["expression"], str):
        return Malformed("'expression' is not a string", raw, start)
    return ToolCall(name=name, arguments=args, raw=raw, start=start)


def _parse_hermes(text: str, out: ParsedTurn) -> None:
    """Qwen / Hermes / Mistral-style `<tool_call>{json}</tool_call>`."""
    consumed: list[tuple[int, int]] = []
    for m in _HERMES_BLOCK.finditer(text):
        consumed.append(m.span())
        body = m.group(1).strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            out.malformed.append(
                Malformed(f"tool_call body is not valid JSON: {exc.msg}",
                          m.group(0), m.start())
            )
            continue
        res = _validate(obj, m.group(0), m.start())
        (out.calls if isinstance(res, ToolCall) else out.malformed).append(res)

    # an opening tag with no matching close
    n_open = len(_HERMES_OPEN.findall(text))
    if n_open > len(consumed):
        idx = text.rfind("<tool_call>")
        if out.truncated_call:
            pass  # generation hit max_tokens mid-call; counted separately
        else:
            out.malformed.append(
                Malformed("unclosed <tool_call> tag", text[idx:][:400], idx)
            )

    def _inside(pos: int) -> bool:
        return any(a <= pos < b for a, b in consumed)

    for m in _LOOSE_JSON.finditer(text):
        if _inside(m.start()):
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            out.malformed.append(
                Malformed("call-shaped JSON emitted outside <tool_call> tags",
                          m.group(0), m.start())
            )

    for m in _PSEUDO.finditer(text):
        if not _inside(m.start()):
            out.malformed.append(
                Malformed("plain-text pseudo-call, not the tool-call format",
                          text[m.start(): m.start() + 120], m.start())
            )


def _parse_llama31(text: str, out: ParsedTurn) -> None:
    """Llama 3.1 emits a bare JSON object, optionally after <|python_tag|>."""
    body = text
    offset = 0
    if _PYTHON_TAG in text:
        offset = text.index(_PYTHON_TAG) + len(_PYTHON_TAG)
        body = text[offset:]
    stripped = body.strip()
    if stripped.startswith("{"):
        # the object may be followed by <|eom_id|> or trailing prose
        dec = json.JSONDecoder()
        try:
            obj, _ = dec.raw_decode(stripped)
        except json.JSONDecodeError as exc:
            if _PYTHON_TAG in text:
                out.malformed.append(
                    Malformed(f"python_tag payload is not valid JSON: {exc.msg}",
                              stripped[:400], offset)
                )
            return
        start = offset + body.index("{")
        res = _validate(obj, stripped[:400], start)
        (out.calls if isinstance(res, ToolCall) else out.malformed).append(res)
        return
    if _PYTHON_TAG in text:
        out.malformed.append(
            Malformed("python_tag present but payload is not a JSON object",
                      stripped[:400], offset)
        )
    for m in _PSEUDO.finditer(text):
        out.malformed.append(
            Malformed("plain-text pseudo-call, not the tool-call format",
                      text[m.start(): m.start() + 120], m.start())
        )


PARSERS = {"hermes": _parse_hermes, "llama31": _parse_llama31}

# Which chat-template role carries a tool result back to the model.
TOOL_ROLE = {"hermes": "tool", "llama31": "ipython"}


def parse_assistant(
    text: str, tool_format: str, finish_reason: str = "stop"
) -> ParsedTurn:
    out = ParsedTurn(text=text)
    out.truncated_call = finish_reason == "length" and (
        "<tool_call>" in text or _PYTHON_TAG in text
    ) and not text.rstrip().endswith("</tool_call>")
    if tool_format not in PARSERS:
        raise ValueError(f"unknown tool_format {tool_format!r}; known: {sorted(PARSERS)}")
    PARSERS[tool_format](text, out)
    return out


def tool_message(result: str, tool_format: str) -> dict:
    role = TOOL_ROLE.get(tool_format, "tool")
    return {"role": role, "name": TOOL_NAME, "content": result}


# ---------------------------------------------------------------------------
# final-answer extraction
# ---------------------------------------------------------------------------
_BOXED = re.compile(r"\\boxed\s*\{")
_ANSWER_IS = re.compile(
    r"(?:final answer|answer)\s*(?:is|:|=)\s*\**\s*(-?[\d,]+(?:\.\d+)?)", re.I
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _boxed_content(text: str) -> str | None:
    """Last \\boxed{...}, brace-matched so nested braces survive."""
    last = None
    for m in _BOXED.finditer(text):
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end(): i - 1]
    return last


def extract_answer(text: str) -> tuple[str | None, str]:
    """Return (answer, how_it_was_found). `how` is logged so that extraction
    failures can be audited rather than silently scored as wrong."""
    boxed = _boxed_content(text)
    if boxed is not None:
        nums = _NUMBER.findall(boxed)
        if nums:
            return nums[-1].replace(",", ""), "boxed"
        return boxed.strip(), "boxed_nonnumeric"
    m = list(_ANSWER_IS.finditer(text))
    if m:
        return m[-1].group(1).replace(",", ""), "answer_is"
    nums = _NUMBER.findall(text)
    if nums:
        return nums[-1].replace(",", ""), "last_number"
    return None, "none"
