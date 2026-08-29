"""Experiment configuration. One dataclass, serialised into every output dir."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Config:
    # --- model -------------------------------------------------------------
    # Chosen for the 1-2x A100 budget and because Qwen2.5-Instruct has a native
    # Hermes-style tool-calling template.  Any Hermes-format model can be
    # swapped in without touching the parser; for Llama 3.1 set
    # tool_format="llama31".
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    tool_format: str = "hermes"          # "hermes" | "llama31"
    backend: str = "vllm"                # "vllm" | "mock"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    dtype: str = "bfloat16"

    # --- task --------------------------------------------------------------
    dataset: str = "chain_sum"           # "chain_sum" | "products"
    n_prompts: int = 50                  # per difficulty level
    levels: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    # Overrides `levels` when set, e.g. "4x3,4x4,5x4,5x5" (terms x digits).
    custom_levels: str | None = None

    # --- sampling ----------------------------------------------------------
    free_n_samples: int = 8
    free_temperature: float = 0.7
    forced_n_samples: int = 1
    forced_temperature: float = 0.0
    max_new_tokens: int = 768
    max_tool_rounds: int = 4

    # --- bookkeeping -------------------------------------------------------
    seed: int = 1234
    out_dir: str = "runs/phase0"
    # Gate: the free-choice tool-call rate must land inside this band at some
    # difficulty level, otherwise Phase 0 fails and the project stops.
    gate_low: float = 0.25
    gate_high: float = 0.75
    malformed_budget: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(Path(path).read_text()))
