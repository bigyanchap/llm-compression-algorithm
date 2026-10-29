"""Variation 1: residual block-delta collapse.

The original prototype tried to collapse node-chains *inside* a transformer
block and found zero candidates on real LLMs: every linear layer sits behind
a norm, an activation, an attention operator, or a residual add. This
variation redefines what a "path" is.

A transformer block computes ``x_{k+1} = x_k + Block_k(x_k)`` on the
residual stream -- the block's contribution is a *delta*. Instead of
collapsing chains inside a block, collapse *across* blocks: replace an
adjacent pair ``(Block_k, Block_{k+1})`` with ONE block ``Block'`` such that
``Block'(h_k) ~= h_{k+2}``, where ``h_k`` is the residual stream entering
layer ``k``.

This is distillation-flavored, and we say so honestly: it relaxes the
original "collapse into one x1w1" to "collapse into the cheapest verified
equivalent". The commit/rollback discipline is unchanged -- a distilled
pair is spliced in tentatively and kept only if validation perplexity stays
within tau of the pristine baseline.

Protocol notes:
  * Calibration boundary states (h_0..h_n) are captured in ONE pass over the
    pristine model (documented limitation: sequential commits shift
    downstream inputs; a production system re-caches per iteration).
  * The student is a deep copy of Block_k (good init: it already computes
    half the teacher mapping), trained with Adam on MSE against the
    teacher's h_{k+2}.
  * Rotary embeddings must be passed explicitly when running a decoder
    layer standalone (transformers >= 5 computes them in LlamaModel).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _block_out(out):
    if isinstance(out, tuple):
        return out[0]
    return getattr(out, "last_hidden_state", out)


def cache_boundary_states(model, calib_ids: torch.Tensor,
                          n_tokens: int = 4096, seq_len: int = 512):
    """Capture residual-stream states at every layer boundary.

    Runs the frozen model over the first ``n_tokens`` of ``calib_ids`` as
    non-overlapping ``seq_len`` sequences. Returns ``(bounds, offsets)``:

      * bounds: list of n_layers+1 tensors, each (n_seq, seq_len, hidden),
        where bounds[k] is the hidden state *entering* layer k (h_0..h_n).
      * offsets: true token offset of each sequence (for RoPE consistency).
    """
    model.eval()
    layers = model.model.layers
    n = len(layers)
    n_seq = n_tokens // seq_len
    assert calib_ids.numel() >= n_seq * seq_len, "not enough calibration tokens"

    captured: dict[int, torch.Tensor] = {}
    last_out: dict[str, torch.Tensor] = {}

    def make_hook(k: int):
        def hook(mod, inp, out):
            captured[k] = inp[0].detach().clone()
        return hook

    def last_hook(mod, inp, out):
        last_out["h"] = _block_out(out).detach().clone()

    handles = [layers[k].register_forward_hook(make_hook(k)) for k in range(n)]
    h_last = layers[n - 1].register_forward_hook(last_hook)
    try:
        per_seq = []
        offsets = []
        with torch.no_grad():
            for s in range(n_seq):
                seq = calib_ids[s * seq_len:(s + 1) * seq_len].unsqueeze(0)
                captured.clear()
                last_out.clear()
                model(seq)
                per_seq.append(
                    [captured[k].squeeze(0) for k in range(n)]
                    + [last_out["h"].squeeze(0)]
                )
                offsets.append(s * seq_len)
    finally:
        for hdl in handles:
            hdl.remove()
        h_last.remove()
    bounds = [
        torch.stack([per_seq[s][k] for s in range(n_seq)], dim=0)
        for k in range(n + 1)
    ]
    return bounds, offsets


def train_student(student: nn.Module, rotary_emb, X: torch.Tensor,
                  Y: torch.Tensor, offsets: list[int], steps: int = 300,
                  lr: float = 1e-4, batch_seqs: int = 2, seed: int = 0,
                  log_every: int | None = None):
    """Distill: train ``student`` so student(X_seq) ~= Y_seq (MSE).

    X, Y: (n_seq, seq_len, hidden). ``offsets`` gives each sequence's true
    token offset so RoPE angles match the teacher's. Returns
    (student, history) with per-step MSE.
    """
    g = torch.Generator().manual_seed(seed)
    seq_len = X.shape[1]
    n_seq = X.shape[0]
    for p in student.parameters():
        p.requires_grad_(True)  # the parent model is frozen; the student trains
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    history: list[float] = []
    off_t = torch.tensor(offsets)
    for step in range(steps):
        idx = torch.randint(0, n_seq, (batch_seqs,), generator=g)
        xb = X[idx]
        yb = Y[idx]
        pos = (torch.arange(seq_len).unsqueeze(0).expand(batch_seqs, -1)
               + off_t[idx].unsqueeze(1))
        pe = rotary_emb(xb, pos)
        opt.zero_grad()
        pred = _block_out(student(xb, position_ids=pos,
                                  position_embeddings=pe))
        loss = torch.nn.functional.mse_loss(pred, yb)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        history.append(lv)
        if log_every and (step + 1) % log_every == 0:
            print(f"      step {step + 1}/{steps} mse {lv:.6f}", flush=True)
    student.eval()
    return student, history


def distill_pair(model, bounds, offsets, k: int, steps: int = 300,
                 lr: float = 1e-4, seed: int = 0):
    """Distill pair (k, k+1): student ~= Block_{k+1}(Block_k(.)) end-to-end.

    Returns (student_block, {"mse_init": ..., "mse_final": ...}).
    """
    student = copy.deepcopy(model.model.layers[k])
    X, Y = bounds[k], bounds[k + 2]
    with torch.no_grad():
        student.eval()
        pos0 = torch.arange(X.shape[1]).unsqueeze(0) + offsets[0]
        pe0 = model.model.rotary_emb(X[:1], pos0)
        pred0 = _block_out(student(X[:1], position_ids=pos0,
                                   position_embeddings=pe0))
        mse_init = float(torch.nn.functional.mse_loss(pred0, Y[:1]))
    student, hist = train_student(
        student, model.model.rotary_emb, X, Y, offsets,
        steps=steps, lr=lr, seed=seed)
    return student, {"mse_init": mse_init, "mse_final": hist[-1],
                     "steps": steps, "lr": lr}


def splice_pair(layers: nn.ModuleList, k: int, student: nn.Module) -> nn.ModuleList:
    """New ModuleList with layers[k], layers[k+1] replaced by ``student``."""
    return nn.ModuleList(list(layers[:k]) + [student] + list(layers[k + 2:]))


def drop_layer(layers: nn.ModuleList, k: int) -> nn.ModuleList:
    """New ModuleList with layers[k] removed (naive depth-reduction baseline)."""
    return nn.ModuleList(list(layers[:k]) + list(layers[k + 1:]))


def install_layers(model, new_layers: nn.ModuleList):
    """Swap the model's decoder stack; update config for cleanliness.

    Re-indexes attention layers (0..n-1): after splicing or dropping, the
    surviving blocks keep their *original* layer_idx values, which breaks
    the KV-cache (indexed by position). The new stack is a new model, so
    positions are renumbered.
    """
    reindex_layers(new_layers)
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model


def reindex_layers(layers: nn.ModuleList) -> nn.ModuleList:
    """Renumber each block's attention layer_idx to its stack position."""
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "layer_idx"):
            attn.layer_idx = i
    return layers


def should_commit(ppl_after: float, ppl_baseline: float,
                  tau: float = 0.05) -> bool:
    """Commit iff perplexity stays within ``tau`` (relative) of baseline."""
    return ppl_after <= ppl_baseline * (1.0 + tau)
