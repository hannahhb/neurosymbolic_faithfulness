"""Activation harvesting via TransformerLens.

Generation happens under vLLM; probing needs internals, which vLLM does not
expose.  So we replay each rollout through a HookedTransformer in a single
teacher-forced forward pass and read the hook points we care about.

Why TransformerLens rather than HF `output_hidden_states`: HF gives one tensor
per layer -- the residual stream, and nothing else.  TransformerLens exposes the
stream *and* the two things that write to it, so the probe can distinguish
"the answer is present at layer 12" from "attention wrote it there".  That is
the difference between describing the phenomenon and locating it.

Components
----------
resid_pre    residual stream entering block L
resid_post   residual stream leaving block L  (== resid_pre of L+1)
attn_out     what attention wrote into the stream at block L
mlp_out      what the MLP wrote into the stream at block L

resid_pre + attn_out + mlp_out == resid_post holds to fp16 rounding; that
identity is the cheapest check that hook names and position indices line up.

Verified against HF `output_hidden_states` on Qwen2.5-0.5B: `resid_post` at
layer L is numerically identical to HF `hidden_states[L+1]` for every layer
EXCEPT the last.  HF applies the final RMSNorm before appending its last
hidden state, so HF's `hidden_states[-1]` is post-`ln_final` while ours is the
raw residual stream.  Ours is the more standard interp object and is consistent
across conditions; just do not expect the final layer to match a previous
HF-based harvest.

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

# hook name templates, keyed by the short component name used on the CLI
COMPONENTS = {
    "resid_pre": "blocks.{l}.hook_resid_pre",
    "resid_post": "blocks.{l}.hook_resid_post",
    "attn_out": "blocks.{l}.hook_attn_out",
    "mlp_out": "blocks.{l}.hook_mlp_out",
}


@dataclass
class HarvestSpec:
    """Which positions, layers and components to keep."""

    n_deciles: int = 10          # completion readout points, evenly spaced
    layer_stride: int = 1        # 2 halves storage
    include_prompt_end: bool = True
    components: tuple[str, ...] = ("resid_post",)

    def __post_init__(self) -> None:
        bad = set(self.components) - set(COMPONENTS)
        if bad:
            raise ValueError(f"unknown component(s) {sorted(bad)}; choose from {sorted(COMPONENTS)}")

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
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
    ) -> None:
        """Load a HookedTransformer.

        The three weight-processing flags are TransformerLens defaults and are
        surfaced here because they change the numbers a probe sees.  Folding
        LayerNorm and centering writing weights make the residual stream
        mean-zero along d_model, which removes the large common component that
        otherwise dominates raw activations (distinct prompts correlate at ~0.93
        without it).  That helps a linear probe rather than hurting it, and it is
        applied identically to both conditions -- but if you want activations
        numerically comparable to raw HF `output_hidden_states`, turn all three
        off.
        """
        import torch
        from transformer_lens import HookedTransformer

        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.max_len = max_len
        self.torch = torch
        self.model = HookedTransformer.from_pretrained(
            model,
            device=device,
            dtype=getattr(torch, dtype),
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.n_layers = self.model.cfg.n_layers
        self.hidden = self.model.cfg.d_model

    # --- tokenisation -----------------------------------------------------

    def encode(self, prompt: str, completion: str) -> tuple[list[int], int, int]:
        """Tokenise prompt and completion separately, then concatenate.

        Deliberately uses the raw tokenizer rather than `model.to_tokens`.
        `to_tokens` defaults to prepend_bos=True, and Qwen's chat template
        carries no BOS -- prepending one would shift every readout index by one
        token and silently probe the wrong positions.  Passing explicit ids
        sidesteps that entirely.

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

    def kept_layers(self, spec: HarvestSpec) -> list[int]:
        return list(range(0, self.n_layers, spec.layer_stride))

    def hook_names(self, spec: HarvestSpec) -> dict[str, list[str]]:
        layers = self.kept_layers(spec)
        return {c: [COMPONENTS[c].format(l=l) for l in layers] for c in spec.components}

    # --- the forward pass -------------------------------------------------

    def harvest_one(
        self, prompt: str, completion: str, spec: HarvestSpec
    ) -> Optional[dict[str, np.ndarray]]:
        """Return {component: [n_positions, n_kept_layers, hidden] float16}.

        Uses run_with_hooks rather than run_with_cache: the hooks slice to the
        readout positions inside the forward pass, so we never materialise the
        full [n_tokens, n_layers, d_model] cache.  On a 7B with a 1300-token
        sequence that is the difference between ~10 MB and ~800 MB per rollout.

        Returns None if the rollout is unusable.
        """
        torch = self.torch
        ids, n_p, n_c = self.encode(prompt, completion)
        if n_c < 1:
            return None
        if len(ids) > self.max_len:
            # Truncating would move prompt_end or drop the completion tail; both
            # corrupt the readout, so drop the rollout instead.
            return None

        idx = readout_indices(n_p, n_c, spec)
        pos = torch.tensor(idx, device=self.device)
        tokens = torch.tensor([ids], device=self.device)

        names = self.hook_names(spec)
        store: dict[str, torch.Tensor] = {}

        def make_hook(key: str):
            def hook(act, hook):  # act: [batch, pos, d_model]
                store[key] = act[0].index_select(0, pos).detach().to(torch.float16).cpu()
            return hook

        fwd_hooks = [
            (hook_name, make_hook(f"{comp}:{li}"))
            for comp, hook_list in names.items()
            for li, hook_name in enumerate(hook_list)
        ]

        with torch.no_grad():
            self.model.run_with_hooks(tokens, return_type=None, fwd_hooks=fwd_hooks)

        out: dict[str, np.ndarray] = {}
        for comp, hook_list in names.items():
            layers = [store[f"{comp}:{li}"] for li in range(len(hook_list))]
            out[comp] = torch.stack(layers, dim=1).numpy()  # [P, L, H]
        store.clear()
        return out


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
