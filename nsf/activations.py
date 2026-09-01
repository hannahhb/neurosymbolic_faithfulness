"""Teacher-forced activation harvesting.

Generation happens under vLLM; probing needs hidden states, which vLLM does not
expose.  So we replay each rollout through HF transformers in a single forward
pass over prompt_ids + completion_ids and read the residual stream off the
positions we care about.

The prompt string is taken verbatim from the stored rollout, never re-rendered.
Any drift between the generation-time and probe-time prompt silently shifts
every token index and corrupts the readout positions -- the single easiest way
to get a beautiful, meaningless result out of this pipeline.

Readout positions
-----------------
`prompt_end` is the last prompt token: the residual stream there is what
produces the model's first generated token, before any reasoning exists.  It is
the headline position.  Decodability of the final answer *there* is evidence the
answer was fixed before the reasoning was written.

The completion deciles are context, not evidence.  Late in a CoT the
intermediate results are literally present in the context window, so high
decodability at 80% is the reasoning working as intended, not unfaithfulness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROMPT_END = "prompt_end"


@dataclass
class HarvestSpec:
    """Which positions and layers to keep."""

    n_deciles: int = 10          # completion readout points, evenly spaced
    layer_stride: int = 1        # 2 halves storage
    include_prompt_end: bool = True

    def position_names(self) -> list[str]:
        names = [PROMPT_END] if self.include_prompt_end else []
        names += [f"gen_{int(100 * (i + 1) / self.n_deciles):03d}" for i in range(self.n_deciles)]
        return names


def readout_indices(n_prompt: int, n_completion: int, spec: HarvestSpec) -> list[int]:
    """Absolute token indices for each readout position.

    Completion decile k is the last token of the first k/n of the completion, so
    the final decile is the completion's last token.  Clamped into the prompt
    when a completion is shorter than the number of deciles, which makes short
    completions degenerate rather than crash -- filter those out downstream.
    """
    idx = [n_prompt - 1] if spec.include_prompt_end else []
    for k in range(1, spec.n_deciles + 1):
        off = max(1, int(round(n_completion * k / spec.n_deciles)))
        idx.append(min(n_prompt + off - 1, n_prompt + n_completion - 1))
    return idx


class Harvester:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "float16",
        max_len: int = 4096,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # output_hidden_states is requested per forward call, not here: passing
        # it to from_pretrained lands it in generation_config and transformers
        # warns about an invalid generation flag.
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype)
        ).to(device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden = self.model.config.hidden_size

    def encode(self, prompt: str, completion: str) -> tuple[list[int], int, int]:
        """Tokenise prompt and completion separately, then concatenate.

        Separate tokenisation is what lets us know the boundary exactly.  It can
        differ from joint tokenisation if the seam merges into one token; we
        check for that and report it rather than let it pass unnoticed.
        """
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        return p_ids + c_ids, len(p_ids), len(c_ids)

    def seam_is_clean(self, prompt: str, completion: str) -> bool:
        split, _, _ = self.encode(prompt, completion)
        joint = self.tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        return split == joint

    def harvest_one(
        self, prompt: str, completion: str, spec: HarvestSpec
    ) -> Optional[np.ndarray]:
        """Return [n_positions, n_kept_layers, hidden] float16, or None if unusable."""
        import torch

        ids, n_p, n_c = self.encode(prompt, completion)
        if n_c < 1:
            return None
        if len(ids) > self.max_len:
            # Truncating would move prompt_end or drop the completion tail; both
            # corrupt the readout, so drop the rollout instead.
            return None

        idx = readout_indices(n_p, n_c, spec)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True)

        # hidden_states[0] is the embedding output; [L] is the residual stream
        # after block L.  Position t encodes everything up to and including t.
        # Index the readout positions per layer *before* stacking: stacking the
        # full [T, L, H] tensor first would allocate hundreds of MB per rollout
        # only to throw almost all of it away.
        layers = list(range(0, self.n_layers + 1, spec.layer_stride))
        pos = torch.tensor(idx, device=self.device)
        picked = torch.stack(
            [out.hidden_states[l][0].index_select(0, pos) for l in layers], dim=1
        )  # [P, L, H]
        picked = picked.to(torch.float16).cpu().numpy()
        del out
        return picked

    def kept_layers(self, spec: HarvestSpec) -> list[int]:
        return list(range(0, self.n_layers + 1, spec.layer_stride))


def save(
    path: str | Path,
    acts: np.ndarray,
    row_ids: Sequence[str],
    positions: Sequence[str],
    layers: Sequence[int],
    meta: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), acts)
    sidecar = {
        "row_ids": list(row_ids),
        "positions": list(positions),
        "layers": [int(l) for l in layers],
        "shape": list(acts.shape),
        **meta,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return path.with_suffix(".npy")


def load(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path).with_suffix("")
    acts = np.load(path.with_suffix(".npy"), mmap_mode="r")
    meta = json.loads(path.with_suffix(".json").read_text())
    return acts, meta
