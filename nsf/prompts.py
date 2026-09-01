"""Prompt construction for the CoT and PoT conditions.

Both conditions are built by the same function so that generation and the
teacher-forced activation pass cannot drift apart.  The chat template is
applied here and nowhere else; downstream code consumes the rendered string.

Convention: the instruction and the problem go in a SINGLE user message; we
never add a system message ourselves.  Note that Qwen's chat template injects
its own default system prompt regardless ("You are Qwen, created by Alibaba
Cloud..."), so the rendered string does contain a system turn.  That prefix is
identical across both conditions, so it does not confound the CoT/PoT contrast,
but it does shift every token index -- which is exactly why the rendered string
is stored on each rollout and replayed verbatim by the activation pass rather
than re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass

# The two instruction blocks are deliberately parallel in structure and length.
# Any asymmetry here is a confound in the CoT-vs-PoT comparison.

COT_INSTRUCTION = """You are solving a math problem. Reason through it step by step in plain English.
End your response with your final answer on its own line, in exactly this format:
Answer: <value>
where <value> is {answer_spec}."""

POT_INSTRUCTION = """You are solving a math problem. Write a Python program that computes the answer.
Reaon in  code only. Do not write any explanation outside the code.
Do not write explanatory comments - the code should stand alone.
The program must compute the answer from the problem's constraints and print it
on the last line in exactly this format:
Answer: <value>
where <value> is {answer_spec}."""

# Per-dataset description of what a well-formed answer looks like.  Kept out of
# the instruction bodies so the two conditions share it exactly.
ANSWER_SPEC = {
    "gsm8k": "a plain number with no units, commas, or currency symbols",
    "math": "the final expression in LaTeX, with no surrounding text",
}

CONDITIONS = ("cot", "pot")


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt in both its raw and chat-templated forms."""

    condition: str
    user_message: str
    text: str  # what actually gets fed to the model


def build_user_message(problem: str, condition: str, dataset: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")
    if dataset not in ANSWER_SPEC:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(ANSWER_SPEC)}")

    template = COT_INSTRUCTION if condition == "cot" else POT_INSTRUCTION
    instruction = template.format(answer_spec=ANSWER_SPEC[dataset])
    return f"{instruction}\n\n{problem.strip()}"


def render(tokenizer, problem: str, condition: str, dataset: str) -> RenderedPrompt:
    """Render a prompt to the exact string the model will see.

    `add_generation_prompt=True` appends the assistant turn header, so the last
    token of `text` is the position immediately before the model's first
    generated token.  That position is the probe's most important readout point.
    """
    user_message = build_user_message(problem, condition, dataset)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return RenderedPrompt(condition=condition, user_message=user_message, text=text)
