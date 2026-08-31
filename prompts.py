"""Exactly two prompts. Everything else is held constant."""

from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."

COT = "Let's think step by step"
CODE = "write a python program for the following problem"

CONDITIONS = ("cot", "code")

INSTRUCTION = {"cot": COT, "code": CODE}


def build_messages(question: str, condition: str) -> list[dict]:
    """The instruction precedes the problem for `code` and follows it for `cot`,
    matching how each is conventionally phrased.  Both are otherwise identical:
    same system prompt, same problem text, no extra formatting hints."""
    assert condition in CONDITIONS, condition
    if condition == "cot":
        user = f"{question}\n\n{COT}"
    else:
        user = f"{CODE}\n\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_prompt(tokenizer, question: str, condition: str,
                 prefix: str = "") -> str:
    """`prefix` is a partial assistant turn to continue from -- this is how
    resampling re-enters the trace at a chosen step."""
    p = tokenizer.apply_chat_template(
        build_messages(question, condition),
        add_generation_prompt=True, tokenize=False,
    )
    return p + prefix
