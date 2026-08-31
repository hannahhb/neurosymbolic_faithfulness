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
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class MockEngine(Engine):
    """Scripted stand-in for offline testing of the CoT / code pipelines.

    It emits a multi-sentence CoT or a multi-line program, with the final answer
    drawn from a small distribution that shifts partway through the trace.  That
    gives resampling something real to measure and the probe a signal that
    appears at a known position.  Runs record backend="mock".
    """

    name = "mock"

    def __init__(self, seed: int = 0, answer_pool=("20", "42"), flip_at: float = 0.5):
        self.tokenizer = MockTokenizer()
        self.seed = seed
        self.answer_pool = tuple(answer_pool)
        self.flip_at = flip_at

    def generate(self, prompts, temperature, max_tokens, seeds, stop):
        out: list[GenOut] = []
        for prompt, seed in zip(prompts, seeds):
            rng = random.Random(
                int(hashlib.sha256(f"{seed}|{prompt}".encode()).hexdigest()[:12], 16)
            )
            code = "python program" in prompt
            ans = rng.choice(self.answer_pool)
            # vary the phrasing so the semantic-similarity filter actually
            # produces both a "kept" and a "different" group
            v = rng.randrange(3)
            if code:
                opener = ["a = 7", "count = 7", "items = 7"][v]
                mid = ["b = 72", "size = 72", "length = 72"][v]
                text = (f"{opener}\n{mid}\ntotal = 7 * 72\n"
                        f"result = {ans}\nprint(result)\n")
            else:
                opener = ["First I set up the problem.",
                          "Let me restate what is given.",
                          "Starting from the quantities involved."][v]
                mid = ["Then I compute the intermediate value.",
                       "Next the running total is needed.",
                       "Now multiply the two quantities."][v]
                text = (f"{opener} {mid} That gives something to work with. "
                        f"So the answer is \\boxed{{{ans}}}.")
            out.append(GenOut(text=text, finish_reason="stop", n_tokens=32))
        return out


class _StopOnStrings:
    """Halt a sequence once a stop string is fully generated.

    transformers' built-in `stop_strings=` removes the match from the output;
    we need it kept (vLLM's include_stop_str_in_output=True), so the truncation
    is done by the caller instead.  Only a tail window of each sequence is
    decoded per step, so the cost does not grow with sequence length.
    """

    def __init__(self, tokenizer, stops: list[str], prompt_len: int, margin: int = 8):
        self.tokenizer = tokenizer
        self.stops = stops
        self.prompt_len = prompt_len
        longest = max((len(tokenizer(s, add_special_tokens=False)["input_ids"])
                       for s in stops), default=1)
        self.window = longest + margin

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        gen = input_ids[:, self.prompt_len:]
        tail = gen[:, -self.window:] if gen.shape[1] > self.window else gen
        texts = self.tokenizer.batch_decode(tail, skip_special_tokens=False)
        return torch.tensor(
            [any(s in t for s in self.stops) for t in texts],
            dtype=torch.bool, device=input_ids.device,
        )


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------
class HFEngine(Engine):
    """Plain transformers generation. Slower than vLLM but far fewer moving
    parts -- it needs only a torch build matching the driver, no separately
    compiled CUDA kernels.

    One difference from VLLMEngine that matters for reproducibility: HF cannot
    seed individual sequences within a batch, so the RNG is seeded once per
    batch from the first sequence's seed.  Re-running with the same batch size
    and the same input order reproduces the output; changing `batch_size` does
    not.  The batch seed is therefore recorded on every GenOut.
    """

    def __init__(
        self,
        model: str,
        dtype: str = "bfloat16",
        device_map: str | None = None,   # None -> "auto" on GPU, "cpu" otherwise
        batch_size: int = 32,
        seed: int = 0,
        attn_implementation: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.batch_size = batch_size
        self.seed = seed
        self.torch = torch
        if device_map is None:
            # "auto" needs accelerate and hangs on CPU-only hosts, which makes
            # the pipeline impossible to smoke-test off-GPU
            device_map = "auto" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # decoder-only batching requires left padding, else the generated
        # continuation starts after the pads and the prompt is misaligned
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "dtype": getattr(torch, dtype),
            "device_map": device_map,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()

    def generate(self, prompts, temperature, max_tokens, seeds, stop):
        torch = self.torch
        assert len(prompts) == len(seeds)
        results: list[GenOut] = []

        for start in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[start: start + self.batch_size])
            chunk_seeds = list(seeds[start: start + self.batch_size])
            batch_seed = int(chunk_seeds[0])
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)

            enc = self.tokenizer(chunk, return_tensors="pt", padding=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature and temperature > 0.0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)
            if stop:
                # NB: transformers' own `stop_strings=` STRIPS the matched
                # string from the output, whereas vLLM is configured here with
                # include_stop_str_in_output=True.  That difference is not
                # cosmetic: a stop string that marks the end of a structured
                # block (a closing tag, say) must survive into the output or the
                # parser sees a truncated block.  So we halt on a custom
                # criterion and truncate to just after the match ourselves.
                from transformers import StoppingCriteriaList

                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [_StopOnStrings(self.tokenizer, list(stop), prompt_len)]
                )

            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)

            new_tokens = out[:, prompt_len:]
            for row in new_tokens:
                ids = row.tolist()
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                # count real tokens, ignoring right-padding on finished rows
                n_real = len(ids)
                pad_id = self.tokenizer.pad_token_id
                while n_real > 0 and ids[n_real - 1] == pad_id:
                    n_real -= 1
                hit_length = n_real >= max_tokens
                # truncate to just after the earliest stop string, keeping it
                hit_stop = False
                if stop:
                    ends = [text.index(t) + len(t) for t in stop if t in text]
                    if ends:
                        text = text[: min(ends)]
                        hit_stop = True
                text, saw_ender = strip_enders(text)
                reason = "length" if (hit_length and not (hit_stop or saw_ender)) else "stop"
                results.append(GenOut(text=text, finish_reason=reason,
                                      n_tokens=n_real))
        return results
