"""Approximate collapse of a nonlinear sub-path into one Linear layer.

Target use case: one GPT-2 MLP block,  x -> c_fc -> gelu -> c_proj -> y,
replaced by a single Linear y ~= W x + b.

Two stages:
  1. Least-squares fit: choose (W, b) minimizing ||X W^T + b - Y||^2 over
     calibration activations (X = sub-path inputs, Y = sub-path outputs).
     Solved in float64 for numerical stability.
  2. REPAIR-style variance calibration: per-output-channel affine rescale so
     the fitted layer's output mean/variance match the ORIGINAL sub-path's
     mean/variance on the calibration data. This counters the internal
     covariate shift that naive fitting introduces, without any retraining.

This is intentionally NOT the report's epsilon-FDE quotient lumping -- that
is listed as future work. Least squares + variance calibration is the
simplest honest baseline for "approximate collapse", and its quality limits
are measured (not assumed) in the experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CollapsedMLP(nn.Module):
    """Drop-in replacement for a GPT-2 MLP block: one Linear layer."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.lin = linear

    def forward(self, x):
        return self.lin(x)


def fit_linear_least_squares(X: torch.Tensor, Y: torch.Tensor):
    """Solve min ||X @ W^T + b - Y||^2. Returns (W, b), float32.

    X: (N, d_in), Y: (N, d_out). Solves the augmented system [X, 1] @ S = Y.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    A = torch.cat([X.to(torch.float64),
                   torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)
    Y64 = Y.to(torch.float64)
    sol, *_ = torch.linalg.lstsq(A, Y64)  # (d_in + 1, d_out)
    W = sol[:-1, :].t().to(torch.float32)  # (d_out, d_in)
    b = sol[-1, :].to(torch.float32)  # (d_out,)
    return W, b


def repair_calibrate(W: torch.Tensor, b: torch.Tensor,
                     X: torch.Tensor, Y_orig: torch.Tensor,
                     eps: float = 1e-6):
    """Per-channel affine rescale so fitted outputs match Y_orig's statistics.

    For each output channel j:
        W_cal[j] = s_j * W[j],   b_cal[j] = s_j * (b[j] - mu_hat[j]) + mu[j]
    with s_j = std(Y_orig[:,j]) / (std(Y_hat[:,j]) + eps).
    After calibration, channel j of the output has mean mu[j] and std sd[j]
    on the calibration data, matching the original sub-path.
    """
    with torch.no_grad():
        Y_hat = X @ W.t() + b
        mu = Y_orig.mean(dim=0)
        sd = Y_orig.std(dim=0)
        mu_hat = Y_hat.mean(dim=0)
        sd_hat = Y_hat.std(dim=0)
        scale = sd / (sd_hat + eps)
        W_cal = W * scale.unsqueeze(1)
        b_cal = scale * (b - mu_hat) + mu
    return W_cal, b_cal


def collapse_mlp_block(X: torch.Tensor, Y_orig: torch.Tensor,
                       dtype=torch.float32):
    """Fit + calibrate a single Linear mimicking an MLP sub-path.

    Returns (CollapsedMLP, diagnostics dict).
    """
    W, b = fit_linear_least_squares(X, Y_orig)
    with torch.no_grad():
        pre_mse = float(((X @ W.t() + b - Y_orig) ** 2).mean())
    Wc, bc = repair_calibrate(W, b, X, Y_orig)
    with torch.no_grad():
        post_mse = float(((X @ Wc.t() + bc - Y_orig) ** 2).mean())
        out_std_orig = float(Y_orig.std())
        out_std_cal = float((X @ Wc.t() + bc).std())
    lin = nn.Linear(X.shape[1], Y_orig.shape[1], bias=True).to(dtype)
    with torch.no_grad():
        lin.weight.copy_(Wc)
        lin.bias.copy_(bc)
    diag = {
        "mse_before_calibration": pre_mse,
        "mse_after_calibration": post_mse,
        "out_std_original": out_std_orig,
        "out_std_calibrated": out_std_cal,
        "calibration_rows": X.shape[0],
    }
    return CollapsedMLP(lin), diag
