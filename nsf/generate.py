"""Generation backends.

vLLM is the production path and runs on the GPU box.  The HF path exists so the
pipeline can be exercised on a laptop (MPS/CPU) at small n, and the mock path so
the plumbing can be tested with no model at all.

All three take prompts as fully-rendered strings and return raw completions.
Chat templating happens in nsf.prompts and nowhere else -- vLLM's own .chat()
helper is deliberately not used, because it would apply a template we do not
control and silently desynchronise generation from the activation pass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence


@dataclass
class Rollout:
    item_id: str
    condition: str
    dataset: str
    sample_idx: int
    prompt_text: str  # exact string fed to the model; the activation pass re-uses it
    completion: str
    finish_reason: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    n: int = 1
    seed: int = 0


class Backend(Protocol):
    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[str]]:
        """Return, per prompt, a list of `cfg.n` completion strings."""


class VLLMBackend:
    def __init__(
        self,
        model: str,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = 4096,
    ) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[str]]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            n=cfg.n,
            seed=cfg.seed,
        )
        outs = self.llm.generate(list(prompts), params)
        # vLLM may return results out of submission order; sort by request id.
        outs = sorted(outs, key=lambda o: int(o.request_id))
        return [[c.text for c in o.outputs] for o in outs]


class HFBackend:
    """Small-n fallback.  Correct, not fast."""

    def __init__(self, model: str, device: str = "auto", dtype: str = "float16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype)
        ).to(device).eval()

    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[str]]:
        import torch

        results: list[list[str]] = []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            n_in = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=cfg.max_tokens,
                    do_sample=cfg.temperature > 0,
                    temperature=cfg.temperature if cfg.temperature > 0 else None,
                    top_p=cfg.top_p if cfg.temperature > 0 else None,
                    num_return_sequences=cfg.n,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            results.append(
                [self.tokenizer.decode(seq[n_in:], skip_special_tokens=True) for seq in out]
            )
        return results


class MockBackend:
    """Deterministic fake completions for plumbing tests."""

    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[str]]:
        out = []
        for i, p in enumerate(prompts):
            if "Python program" in p:
                body = f"x = {i % 9 + 1}\nprint(f'Answer: {{x}}')"
            else:
                body = f"Step one.\nStep two.\nAnswer: {i % 9 + 1}"
            out.append([body] * cfg.n)
        return out


def build_backend(kind: str, model: str, **kw) -> Backend:
    if kind == "vllm":
        return VLLMBackend(model, **kw)
    if kind == "hf":
        return HFBackend(model, **kw)
    if kind == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend {kind!r}")


def write_rollouts(rollouts: list[Rollout], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rollouts:
            fh.write(r.to_json() + "\n")
    return path


def read_rollouts(path: str | Path) -> list[Rollout]:
    with Path(path).open() as fh:
        return [Rollout(**json.loads(line)) for line in fh if line.strip()]
