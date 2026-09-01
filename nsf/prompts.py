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

# The "Output only Python code" clause is load-bearing, not stylistic.  Measured
# on the same 60 GSM8K items, Qwen2.5-0.5B, greedy:
#
#   "Output only Python code."              49/60 executed, 28.3% accuracy
#   "Reason in code only."                  29/60 executed, 16.7% accuracy
#   both, as below                          57/60 executed, 31.7% accuracy
#
# Under the middle wording only half the completions contained print() at all and
# some were pure prose.  A PoT rollout containing no code is not the PoT
# condition -- it is CoT with a different preamble -- so compliance here decides
# whether the CoT/PoT contrast measures what it claims to.  Keep the explicit
# output constraint AND the anti-prose line; the "reason in code" framing rides
# alongside them rather than replacing them.
POT_INSTRUCTION = """You are solving a math problem. Write a Python program that computes the answer.
Output only Python code - reason in code, not in prose.
Do not write any explanation outside the code.
Do not write explanatory comments - the code should stand alone.
The program must compute the answer from the problem's constraints and print it
on the last line in exactly this format:
Answer: <value>
where <value> is {answer_spec}."""

# Per-dataset description of what a well-formed answer looks like.  Kept out of
# the instruction bodies so the two conditions share it exactly.
# Kept deliberately free of medium-specific language.  An earlier MATH spec read
# "the final expression in LaTeX", which broke the PoT arm outright: told that
# its output must be LaTeX, Qwen2.5-7B emitted `Answer: |$\len(asymptotes)$|`
# as a bare statement instead of a print() call, and the program died with a
# SyntaxError.  On GSM8K, whose spec asks for a plain number, the same model
# complied on 59/60 items.  Anything that reads as a formatting demand on the
# *medium* rather than the *value* pushes PoT toward emitting markup instead of
# code.  LaTeX golds still match: nsf.answers normalises \frac, \sqrt, \boxed
# and math-mode delimiters if the model produces them anyway.
ANSWER_SPEC = {
    "gsm8k": "a plain number with no units, commas, or currency symbols",
    "math": "the final answer, with no surrounding text, units, or explanation",
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
