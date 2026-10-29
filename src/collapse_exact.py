"""Exact affine folding of consecutive Linear layers.

If x_{k+1} = W_k x_k + b_k with NO nonlinearity between the layers, the whole
chain composes into a single affine map:

    W* = W_{n-1} ... W_1
    b* = b_{n-1} + W_{n-1} b_{n-2} + W_{n-1} W_{n-2} b_{n-3} + ...

This is bit-for-bit exact (up to floating point roundoff). Any path that
contains a nonlinearity is REFUSED -- the function returns (None, reason)
instead of silently producing a wrong answer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .graph import Path, linear_weight_bias


def fold_affine_chain(weights_biases):
    """Fold [(W1,b1), (W2,b2), ...] in forward order into (W*, b*).

    W_k has shape (out_k, in_k), b_k has shape (out_k,).
    """
    if not weights_biases:
        raise ValueError("empty chain")
    W_star, b_star = weights_biases[0][0].clone(), weights_biases[0][1].clone()
    for W, b in weights_biases[1:]:
        if W.shape[1] != W_star.shape[0]:
            raise ValueError(
                f"shape mismatch: W {tuple(W.shape)} cannot follow "
                f"W* {tuple(W_star.shape)}"
            )
        b_star = W @ b_star + b
        W_star = W @ W_star
    return W_star, b_star


def try_collapse_exact(path: Path):
    """Try to exactly collapse a Path.

    Returns (W_star, b_star, None) on success, or (None, None, reason) if the
    path cannot be folded exactly (e.g. it contains a nonlinearity).
    """
    if path.has_nonlinearity:
        return None, None, (
            f"path '{path.name}' contains a nonlinearity between its layers; "
            "exact affine folding is mathematically invalid -- refusing"
        )
    if len(path.modules) < 2:
        return None, None, (
            f"path '{path.name}' has fewer than 2 linear layers; nothing to fold"
        )
    try:
        wbs = [linear_weight_bias(m) for m in path.modules]
    except Exception as exc:  # noqa: BLE001 - report, don't crash the loop
        return None, None, f"path '{path.name}': cannot read weights ({exc})"
    try:
        W_star, b_star = fold_affine_chain(wbs)
    except ValueError as exc:
        return None, None, f"path '{path.name}': {exc}"
    return W_star, b_star, None


def make_folded_linear(W_star: torch.Tensor, b_star: torch.Tensor,
                       dtype=None) -> nn.Linear:
    """Build an nn.Linear equivalent to the folded chain."""
    out_f, in_f = W_star.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        lin.weight.copy_(W_star)
        lin.bias.copy_(b_star)
    if dtype is not None:
        lin = lin.to(dtype)
    return lin
