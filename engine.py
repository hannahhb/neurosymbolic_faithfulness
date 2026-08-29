"""Generation backends.

`VLLMEngine` is what runs on the A100 box.  `MockEngine` is a scripted stand-in
with the same interface so the whole pipeline -- parsing, the tool loop,
scoring, the gate check, the plots -- can be exercised on a laptop before
burning GPU time.  Mock runs are stamped `"backend": "mock"` in every output
file so they can never be mistaken for real results.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Sequence

# Trailing template tokens that survive `skip_special_tokens=False`.
_ENDERS = ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "</s>",
           "<|endoftext|>"]


@dataclass
class GenOut:
    text: str
    finish_reason: str          # "stop" | "length"
    n_tokens: int = 0


def strip_enders(text: str) -> tuple[str, bool]:
    """Remove a trailing end-of-turn token. Returns (text, saw_ender)."""
    saw = False
    changed = True
    while changed:
        changed = False
        for e in _ENDERS:
            if text.rstrip().endswith(e):
                text = text.rstrip()[: -len(e)]
                saw = changed = True
    return text, saw


class Engine:
    """Interface implemented by both backends."""

    tokenizer: Any
    name: str

    def generate(
        self,
        prompts: Sequence[str],
        temperature: float,
        max_tokens: int,
        seeds: Sequence[int],
        stop: Sequence[str],
    ) -> list[GenOut]:
        raise NotImplementedError


class VLLMEngine(Engine):
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        seed: int = 0,
        enforce_eager: bool = False,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            enforce_eager=enforce_eager,
        )

    def generate(self, prompts, temperature, max_tokens, seeds, stop):
        from vllm import SamplingParams

        assert len(prompts) == len(seeds)
        params = [
            SamplingParams(
                n=1,
                temperature=temperature,
                top_p=1.0 if temperature == 0.0 else 0.95,
                max_tokens=max_tokens,
                seed=int(s),
                stop=list(stop),
                include_stop_str_in_output=True,
                skip_special_tokens=False,
            )
            for s in seeds
        ]
        outs = self.llm.generate(list(prompts), sampling_params=params)
        results: list[GenOut] = []
        for o in outs:
            c = o.outputs[0]
            text, saw_ender = strip_enders(c.text)
            reason = c.finish_reason or "stop"
            if reason == "length" and saw_ender:
                reason = "stop"
            results.append(GenOut(text=text, finish_reason=reason,
                                  n_tokens=len(c.token_ids)))
        return results


# ---------------------------------------------------------------------------
# mock backend
# ---------------------------------------------------------------------------
class MockTokenizer:
    """Minimal Qwen-shaped chat template, enough to exercise the pipeline."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False):
        import json as _json

        parts = []
        sys_msg = ""
        rest = list(messages)
        if rest and rest[0]["role"] == "system":
            sys_msg = rest[0]["content"]
            rest = rest[1:]
        if tools:
            sys_msg += (
                "\n\n# Tools\nYou may call one or more functions to assist with "
                "the user query.\n<tools>\n"
                + "\n".join(_json.dumps(t) for t in tools)
                + "\n</tools>"
            )
        parts.append(f"<|im_start|>system\n{sys_msg}<|im_end|>\n")
        for m in rest:
            if m["role"] == "tool":
                parts.append(
                    f"<|im_start|>user\n<tool_response>\n{m['content']}\n"
                    f"</tool_response><|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


_EXPR = re.compile(r":\s*(.+?)\s*=?\s*$", re.M)


class MockEngine(Engine):
    """Deterministic pseudo-model.

    Tool-call propensity rises with the number of digits in the question, so the
    calibration curves have the shape a real run might have.  It also emits a
    small rate of malformed calls so the malformed-rate gate is exercised.
    """

    name = "mock"

    def __init__(self, malformed_rate: float = 0.03, seed: int = 0):
        self.tokenizer = MockTokenizer()
        self.malformed_rate = malformed_rate
        self.seed = seed

    @staticmethod
    def _expression(prompt: str) -> str | None:
        m = re.search(r"(?:problem|multiplication):\s*([-\d\s+*]+?)\s*=?\s*\.?\s*\n",
                      prompt + "\n")
        return m.group(1).strip() if m else None

    def generate(self, prompts, temperature, max_tokens, seeds, stop):
        from neurosymbolic_faithfulness.calculator import run_calculator

        out: list[GenOut] = []
        for prompt, seed in zip(prompts, seeds):
            rng = random.Random(
                int(hashlib.sha256(f"{seed}|{prompt}".encode()).hexdigest()[:12], 16)
            )
            expr = self._expression(prompt)
            digits = sum(c.isdigit() for c in (expr or ""))
            has_tool = "<tools>" in prompt
            already_called = "<tool_response>" in prompt

            if already_called:
                m = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>",
                               prompt, re.DOTALL)
                out.append(GenOut(f"The result is {m[-1]}.\n\n\\boxed{{{m[-1]}}}",
                                  "stop", 20))
                continue

            p_tool = 0.0 if not has_tool else min(0.98, max(0.02, (digits - 4) / 30))
            if has_tool and rng.random() < p_tool:
                if rng.random() < self.malformed_rate:
                    body = '{"name": "calculator", "arguments": {"expr": "%s"}}' % expr
                    out.append(GenOut(
                        f"This is large; I'll compute it.\n<tool_call>\n{body}\n"
                        f"</tool_call>", "stop", 40))
                else:
                    body = '{"name": "calculator", "arguments": {"expression": "%s"}}' % expr
                    out.append(GenOut(
                        f"That is too many digits to add reliably in my head, so "
                        f"I'll use the calculator.\n<tool_call>\n{body}\n"
                        f"</tool_call>", "stop", 45))
                continue

            truth = run_calculator(expr or "0")
            val = truth.value if truth.ok else "0"
            # unaided accuracy decays with digit count
            p_correct = max(0.02, 1.0 - digits / 45.0)
            if rng.random() >= p_correct:
                val = str(int(val) + rng.choice([1, -1, 10, -100, 1000]))
            out.append(GenOut(f"Let me work through it step by step.\n\n"
                              f"\\boxed{{{val}}}", "stop", 30))
        return out


def make_engine(cfg) -> Engine:
    if cfg.backend == "mock":
        return MockEngine(seed=cfg.seed)
    return VLLMEngine(
        model=cfg.model,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype=cfg.dtype,
        seed=cfg.seed,
    )
