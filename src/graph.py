"""Module-level graph walk for HuggingFace GPT-2 and generic Sequential nets.

A "path" here is an ordered chain of leaf modules, e.g. one GPT-2 MLP block's
c_fc -> (gelu) -> c_proj. The walker does not execute anything; it only
enumerates candidate paths and flags whether a nonlinearity sits inside them,
so the collapse stages know what they are allowed to touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class Path:
    """An ordered chain of leaf modules forming one collapse candidate."""

    name: str
    module_names: list  # dotted names, forward order
    modules: list  # module objects, forward order
    has_nonlinearity: bool
    kind: str  # "mlp" | "linear_chain" | ...
    notes: str = ""


def linear_weight_bias(mod):
    """Return (W, b) in standard (out_features, in_features) layout.

    Handles torch.nn.Linear and HuggingFace transformers Conv1D (whose weight
    is stored transposed as (in, out)).
    """
    if isinstance(mod, nn.Linear):
        return mod.weight.detach().clone(), mod.bias.detach().clone()
    # transformers Conv1D: weight shape (in, out) == (nx, nf)
    w = mod.weight.detach()
    if w.shape[0] != mod.bias.shape[0] and w.shape[1] == mod.bias.shape[0]:
        w = w.t().contiguous()
    return w.clone(), mod.bias.detach().clone()


def is_linear_leaf(mod) -> bool:
    if isinstance(mod, nn.Linear):
        return True
    # Duck-type transformers Conv1D without importing transformers here.
    return type(mod).__name__ == "Conv1D" and hasattr(mod, "weight")


def _child(obj, part):
    # ModuleList / Sequential support integer indexing; plain modules use
    # attribute access. Try indexing first for numeric parts.
    if part.isdigit():
        try:
            return obj[int(part)]
        except (TypeError, IndexError, KeyError):
            pass
    return getattr(obj, part)


def get_submodule(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        obj = _child(obj, part)
    return obj


def set_submodule(model, dotted: str, new_mod):
    parts = dotted.split(".")
    parent = model
    for part in parts[:-1]:
        parent = _child(parent, part)
    last = parts[-1]
    if last.isdigit():
        try:
            parent[int(last)] = new_mod
            return
        except (TypeError, IndexError):
            pass
    setattr(parent, last, new_mod)


def find_linear_chains(seq: nn.Sequential):
    """Group consecutive nn.Linear layers of a Sequential into foldable chains.

    A chain is foldable only if *no* nonlinearity sits between its layers.
    Returns a list of Path objects with kind="linear_chain".
    """
    paths = []
    current = []
    current_names = []
    nonlinear = {nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU,
                 nn.ELU, nn.Softmax, nn.LayerNorm}

    def flush():
        if len(current) >= 2:
            paths.append(Path(
                name="chain:" + "+".join(current_names),
                module_names=list(current_names),
                modules=list(current),
                has_nonlinearity=False,
                kind="linear_chain",
            ))
        current.clear()
        current_names.clear()

    for name, mod in seq.named_children():
        if isinstance(mod, nn.Linear):
            current.append(mod)
            current_names.append(name)
        elif type(mod) in nonlinear:
            flush()  # nonlinearity breaks the chain; it is not foldable
        else:
            flush()  # unknown module: be conservative, break the chain
    flush()
    return paths


def enumerate_gpt2_paths(model):
    """Enumerate collapse candidates for a HuggingFace GPT-2 model.

    Returns a dict:
      - "exact_candidates": chains of Linear/Conv1D with no nonlinearity
        between them. For stock GPT-2 this is EMPTY (honest finding: every
        linear layer is separated by an activation, attention softmax, or
        residual add), which is why the exact folder has nothing to do here.
      - "mlp_paths": one Path per transformer block for the approximate
        folder: c_fc -> gelu -> c_proj  ==>  single Linear.
    """
    exact_candidates = []
    mlp_paths = []

    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        mlp = block.mlp
        c_fc = mlp.c_fc
        c_proj = mlp.c_proj
        assert is_linear_leaf(c_fc) and is_linear_leaf(c_proj), (
            f"block {i}: unexpected MLP submodule types "
            f"{type(c_fc).__name__}, {type(c_proj).__name__}"
        )
        mlp_paths.append(Path(
            name=f"h.{i}.mlp",
            module_names=[f"transformer.h.{i}.mlp.c_fc",
                          f"transformer.h.{i}.mlp.c_proj"],
            modules=[c_fc, c_proj],
            has_nonlinearity=True,  # GELU sits between c_fc and c_proj
            kind="mlp",
            notes="c_fc -> gelu -> c_proj; approximate collapse target",
        ))

        # Attention: c_attn is linear, but its output goes through the
        # (nonlinear) attention operator before c_proj, so no exact chain.
        # Residual adds also break exact foldability everywhere.

    return {"exact_candidates": exact_candidates, "mlp_paths": mlp_paths}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def param_breakdown_gpt2(model):
    """Return per-part parameter counts for a GPT-2 model (for reporting)."""
    parts = {}
    for name, mod in model.named_modules():
        if is_linear_leaf(mod):
            n = sum(p.numel() for p in mod.parameters())
            parts[name] = parts.get(name, 0) + n
    mlp_total = sum(n for k, n in parts.items() if ".mlp." in k)
    attn_total = sum(n for k, n in parts.items() if ".attn." in k)
    return {"mlp_total": mlp_total, "attn_total": attn_total,
            "linear_total": sum(parts.values()),
            "model_total": count_parameters(model)}


def enumerate_llama_like_paths(model, layers_attr="model.layers",
                               mlp_attr="mlp", layer_prefix="layers"):
    """Enumerate collapse candidates for Llama-style models
    (SmolLM2, Qwen2.5, Llama, Mistral, ...).

    Architecture: RMSNorm + attention (q/k/v/o projs around the attention
    operator) + gated SwiGLU MLP (gate_proj / up_proj / down_proj).

    Returns the same dict shape as enumerate_gpt2_paths:
      - "exact_candidates": always EMPTY for these architectures (honest
        finding: no two Linear layers sit adjacent without a nonlinearity,
        norm, or residual between them).
      - "mlp_paths": one Path per layer; the approximate folder mimics the
        whole gated MLP sub-path (gate_proj/up_proj/down_proj + SiLU gating)
        with a single Linear.
    """
    layers = get_submodule(model, layers_attr)
    exact_candidates = []
    mlp_paths = []
    for i, block in enumerate(layers):
        mlp = getattr(block, mlp_attr)
        proj_names = [n for n, _ in mlp.named_children()]
        mlp_paths.append(Path(
            name=f"{layer_prefix}.{i}.mlp",
            module_names=[f"{layers_attr}.{i}.{mlp_attr}.{p}"
                          for p in proj_names],
            modules=[mlp],  # single module; exact folder refuses (nonlinear)
            has_nonlinearity=True,
            kind="mlp",
            notes=(f"gated SwiGLU MLP ({', '.join(proj_names)}); "
                   "approximate collapse target"),
        ))
    return {"exact_candidates": exact_candidates, "mlp_paths": mlp_paths}
