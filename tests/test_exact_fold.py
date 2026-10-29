"""Unit tests for the path-collapse prototype.

Run:  python -m unittest discover -s tests -v   (from the repo root)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.graph import Path, linear_weight_bias, find_linear_chains
from src.collapse_exact import fold_affine_chain, try_collapse_exact
from src.collapse_approx import (fit_linear_least_squares, repair_calibrate,
                                 collapse_mlp_block)

torch.manual_seed(0)


def rand_affine(out_f, in_f):
    W = torch.randn(out_f, in_f)
    b = torch.randn(out_f)
    return W, b


class TestExactFold(unittest.TestCase):
    def test_two_layer_fold_matches_sequential(self):
        W1, b1 = rand_affine(16, 8)
        W2, b2 = rand_affine(4, 16)
        Ws, bs = fold_affine_chain([(W1, b1), (W2, b2)])
        x = torch.randn(32, 8)
        expected = (x @ W1.t() + b1) @ W2.t() + b2
        got = x @ Ws.t() + bs
        self.assertTrue(torch.allclose(got, expected, atol=1e-6),
                        f"max diff {(got - expected).abs().max()}")

    def test_three_layer_fold_matches_sequential(self):
        W1, b1 = rand_affine(16, 8)
        W2, b2 = rand_affine(16, 16)
        W3, b3 = rand_affine(4, 16)
        Ws, bs = fold_affine_chain([(W1, b1), (W2, b2), (W3, b3)])
        x = torch.randn(32, 8)
        h = x @ W1.t() + b1
        h = h @ W2.t() + b2
        expected = h @ W3.t() + b3
        got = x @ Ws.t() + bs
        # 3 chained matmuls: allow float32 roundoff (still exact algebraically)
        self.assertTrue(torch.allclose(got, expected, atol=1e-4),
                        f"max diff {(got - expected).abs().max()}")

    def test_fold_formula_spot_check(self):
        # Hand-computed: W1=I, b1=[1,1]; W2=[[2,0],[0,3]], b2=[0,0]
        W1 = torch.eye(2)
        b1 = torch.ones(2)
        W2 = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        b2 = torch.zeros(2)
        Ws, bs = fold_affine_chain([(W1, b1), (W2, b2)])
        self.assertTrue(torch.allclose(Ws, W2, atol=1e-7))
        self.assertTrue(torch.allclose(bs, torch.tensor([2.0, 3.0]), atol=1e-7))

    def test_refuses_path_with_gelu(self):
        lin = nn.Linear(8, 8)
        path = Path(name="bad", module_names=["a", "b"], modules=[lin, lin],
                    has_nonlinearity=True, kind="linear_chain",
                    notes="gelu between layers")
        W, b, reason = try_collapse_exact(path)
        self.assertIsNone(W)
        self.assertIsNone(b)
        self.assertIsInstance(reason, str)
        self.assertIn("nonlinearity", reason.lower())

    def test_linear_weight_bias_nn_linear(self):
        lin = nn.Linear(6, 4)  # in=6, out=4 -> W is (out, in) = (4, 6)
        W, b = linear_weight_bias(lin)
        self.assertEqual(tuple(W.shape), (4, 6))
        self.assertEqual(tuple(b.shape), (4,))
        x = torch.randn(5, 6)
        self.assertTrue(torch.allclose(x @ W.t() + b, lin(x), atol=1e-6))

    def test_find_linear_chains_breaks_on_activation(self):
        seq = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8),
                            nn.GELU(), nn.Linear(8, 2))
        paths = find_linear_chains(seq)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].module_names, ["0", "1"])
        self.assertFalse(paths[0].has_nonlinearity)

    def test_find_linear_chains_none_when_isolated(self):
        seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.assertEqual(find_linear_chains(seq), [])


class TestApproxCollapse(unittest.TestCase):
    def test_least_squares_recovers_true_linear_map(self):
        torch.manual_seed(1)
        N, d = 2000, 16
        W_true = torch.randn(8, d)
        b_true = torch.randn(8)
        X = torch.randn(N, d)
        Y = X @ W_true.t() + b_true
        W, b = fit_linear_least_squares(X, Y)
        self.assertTrue(torch.allclose(W, W_true, atol=1e-3),
                        f"max W err {(W - W_true).abs().max()}")
        self.assertTrue(torch.allclose(b, b_true, atol=1e-3))

    def test_variance_calibration_restores_stats(self):
        torch.manual_seed(2)
        N, d = 4000, 32
        X = torch.randn(N, d)
        Y_orig = torch.tanh(X @ torch.randn(16, d).t()) * 3.0 + 1.0
        W = torch.randn(16, d) * 0.05  # deliberately bad fit
        b = torch.zeros(16)
        Wc, bc = repair_calibrate(W, b, X, Y_orig)
        Y_cal = X @ Wc.t() + bc
        self.assertTrue(torch.allclose(Y_cal.mean(0), Y_orig.mean(0), atol=1e-4))
        self.assertTrue(torch.allclose(Y_cal.std(0), Y_orig.std(0), atol=1e-4))

    def test_collapse_mlp_block_end_to_end(self):
        torch.manual_seed(3)
        N, d = 3000, 64
        X = torch.randn(N, d)
        mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        with torch.no_grad():
            Y = mlp(X)
        collapsed, diag = collapse_mlp_block(X, Y)
        with torch.no_grad():
            Yc = collapsed(X)
        # calibrated stats must match the original sub-path's stats
        self.assertTrue(torch.allclose(Yc.mean(0), Y.mean(0), atol=1e-3))
        self.assertTrue(torch.allclose(Yc.std(0), Y.std(0), atol=1e-3))
        self.assertIn("mse_before_calibration", diag)
        self.assertIn("mse_after_calibration", diag)


if __name__ == "__main__":
    unittest.main()
