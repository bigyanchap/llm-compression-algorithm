"""Verification: the losslessness / quality checks behind commit-or-rollback."""

from __future__ import annotations

import math

import torch


def _model_output(model, input_ids):
    out = model(input_ids)
    return out.logits if hasattr(out, "logits") else out


@torch.no_grad()
def max_abs_diff(model_a, model_b, input_ids: torch.Tensor) -> float:
    """Max absolute difference of model outputs on given inputs.

    Works with HuggingFace models (returns .logits) and plain nn.Modules
    (returns a tensor).
    """
    model_a.eval()
    model_b.eval()
    out_a = _model_output(model_a, input_ids)
    out_b = _model_output(model_b, input_ids)
    return float((out_a - out_b).abs().max())


@torch.no_grad()
def mean_abs_diff(model_a, model_b, input_ids: torch.Tensor) -> float:
    model_a.eval()
    model_b.eval()
    out_a = _model_output(model_a, input_ids)
    out_b = _model_output(model_b, input_ids)
    return float((out_a - out_b).abs().mean())


@torch.no_grad()
def perplexity(model, token_ids: torch.Tensor, max_len: int = 1024,
               max_chunks: int | None = None):
    """Causal-LM perplexity over non-overlapping chunks of token_ids.

    token_ids: 1-D tensor of token ids. Returns (ppl, n_tokens_used).
    """
    model.eval()
    ids = token_ids
    n_chunks = ids.numel() // max_len
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    total_nll = 0.0
    total_tokens = 0
    for c in range(n_chunks):
        chunk = ids[c * max_len:(c + 1) * max_len].unsqueeze(0)
        out = model(chunk, labels=chunk)
        # out.loss is mean NLL over the chunk's tokens
        n_tok = chunk.numel()
        total_nll += float(out.loss) * n_tok
        total_tokens += n_tok
    ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
    return ppl, total_tokens


@torch.no_grad()
def block_output_drift(orig_module, new_module, X: torch.Tensor) -> float:
    """Max abs drift of a single submodule's output on calibration inputs."""
    y0 = orig_module(X)
    y1 = new_module(X)
    return float((y0 - y1).abs().max())
