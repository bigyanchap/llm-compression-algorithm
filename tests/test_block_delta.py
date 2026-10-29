"""Unit tests for Variation 1 (residual block-delta collapse).

Covers: splicing keeps the model runnable with correct param counts,
distillation actually reduces the imitation loss on synthetic data, and
the commit/rollback gate commits within tau and rolls back beyond it.
"""

import copy
import unittest

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import block_delta as BD


def make_toy(n_layers=4, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64,
                      num_hidden_layers=n_layers, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=128)
    return LlamaForCausalLM(cfg).eval()


class TestSplice(unittest.TestCase):
    def test_splice_pair_param_count_and_forward(self):
        toy = make_toy(4)
        layers = list(toy.model.layers)
        per_block = BD.count_parameters(layers[0])
        before = BD.count_parameters(toy)
        student = copy.deepcopy(layers[1])
        new_layers = BD.splice_pair(nn.ModuleList(layers), 1, student)
        self.assertEqual(len(new_layers), 3)
        # originals untouched by the splice
        self.assertEqual(len(layers), 4)
        BD.install_layers(toy, new_layers)
        self.assertEqual(toy.config.num_hidden_layers, 3)
        self.assertEqual(BD.count_parameters(toy), before - per_block)
        ids = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            logits = toy(ids).logits
        self.assertEqual(tuple(logits.shape), (1, 8, 128))

    def test_drop_layer(self):
        toy = make_toy(4)
        layers = list(toy.model.layers)
        per_block = BD.count_parameters(layers[2])
        before = BD.count_parameters(toy)
        BD.install_layers(toy, BD.drop_layer(nn.ModuleList(layers), 2))
        self.assertEqual(len(toy.model.layers), 3)
        self.assertEqual(BD.count_parameters(toy), before - per_block)
        ids = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            logits = toy(ids).logits
        self.assertEqual(tuple(logits.shape), (1, 8, 128))

    def test_splice_at_edges(self):
        toy = make_toy(4)
        layers = nn.ModuleList(list(toy.model.layers))
        first = BD.splice_pair(layers, 0, copy.deepcopy(layers[0]))
        last = BD.splice_pair(layers, 2, copy.deepcopy(layers[2]))
        self.assertEqual(len(first), 3)
        self.assertEqual(len(last), 3)


class TestDistill(unittest.TestCase):
    def test_distill_loss_decreases(self):
        torch.manual_seed(1)
        toy = make_toy(2)
        rot = toy.model.rotary_emb
        l0, l1 = toy.model.layers[0], toy.model.layers[1]
        # synthetic teacher: two stacked blocks; distill into one
        X = torch.randn(4, 16, 32)
        offs = [0, 16, 32, 48]
        with torch.no_grad():
            l0.eval(); l1.eval()
            pos = torch.arange(16).unsqueeze(0).expand(4, -1) + \
                torch.tensor(offs).unsqueeze(1)
            pe = rot(X, pos)
            Y = BD._block_out(l1(BD._block_out(l0(X, position_ids=pos,
                                                   position_embeddings=pe)),
                                 position_ids=pos, position_embeddings=pe))
        student = copy.deepcopy(l0)
        student, hist = BD.train_student(student, rot, X, Y, offs,
                                         steps=40, lr=1e-3, batch_seqs=2,
                                         seed=0)
        self.assertLess(hist[-1], hist[0],
                        f"loss did not decrease: {hist[0]:.4f} -> {hist[-1]:.4f}")

    def test_distill_pair_reports_mse(self):
        torch.manual_seed(2)
        toy = make_toy(2)
        X = torch.randn(2, 8, 32)
        bounds = [X, torch.randn(2, 8, 32), torch.randn(2, 8, 32)]
        student, diag = BD.distill_pair(toy, bounds, [0, 8], k=0,
                                        steps=5, lr=1e-3, seed=0)
        self.assertIn("mse_init", diag)
        self.assertIn("mse_final", diag)
        self.assertGreaterEqual(diag["mse_init"], 0.0)


class TestGate(unittest.TestCase):
    def test_commit_within_tau(self):
        self.assertTrue(BD.should_commit(10.0, 10.76))          # improved
        self.assertTrue(BD.should_commit(10.76, 10.76))         # identical
        self.assertTrue(BD.should_commit(10.76 * 1.05, 10.76))  # boundary

    def test_rollback_beyond_tau(self):
        self.assertFalse(BD.should_commit(10.76 * 1.0501, 10.76))
        self.assertFalse(BD.should_commit(11.5, 10.76))         # +6.9%
        self.assertFalse(BD.should_commit(29118.0, 10.76))      # destroyed

    def test_custom_tau(self):
        self.assertTrue(BD.should_commit(11.0, 10.0, tau=0.10))
        self.assertFalse(BD.should_commit(11.0, 10.0, tau=0.05))


if __name__ == "__main__":
    unittest.main()


class TestCheckpoints(unittest.TestCase):
    """Checkpoint save/load + resume-order logic for the long experiment."""

    @staticmethod
    def _load_bdc():
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "experiments",
                            "block_delta_collapse.py")
        spec = importlib.util.spec_from_file_location(
            "block_delta_collapse", os.path.abspath(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_ckpt_key_roundtrip(self):
        bdc = self._load_bdc()
        for a, b in [(0, 1), (6, 7), (30, 31)]:
            self.assertEqual(bdc.parse_ckpt_key(bdc.ckpt_key(a, b)), (a, b))

    def test_json_roundtrip(self):
        import tempfile
        bdc = self._load_bdc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json")
            self.assertEqual(bdc.load_json(p, "dflt"), "dflt")
            obj = {"a": [1, 2], "b": {"c": 1.5}}
            bdc.save_json(p, obj)
            self.assertEqual(bdc.load_json(p, None), obj)

    def test_student_ckpt_roundtrip(self):
        import tempfile
        bdc = self._load_bdc()
        toy = make_toy(4)
        student = copy.deepcopy(toy.model.layers[1])
        with tempfile.TemporaryDirectory() as d:
            old = bdc.CKPT_DIR
            bdc.CKPT_DIR = d
            try:
                bdc.save_student_ckpt(2, 3, student)
                # pristine layer passed explicitly: the live stack may shift
                back = bdc.load_student_ckpt(toy.model.layers[2], 2, 3)
            finally:
                bdc.CKPT_DIR = old
        for p0, p1 in zip(student.parameters(), back.parameters()):
            self.assertTrue(torch.equal(p0, p1))
        self.assertFalse(back.training)
        self.assertFalse(any(p.requires_grad for p in back.parameters()))

    def test_load_uses_pristine_layer_not_live_stack(self):
        # after commits shift the live stack, loading must still deepcopy
        # the PRISTINE block k (passed explicitly), not model.model.layers[k]
        import tempfile
        bdc = self._load_bdc()
        toy = make_toy(4)
        pristine1 = copy.deepcopy(toy.model.layers[1])
        with tempfile.TemporaryDirectory() as d:
            old = bdc.CKPT_DIR
            bdc.CKPT_DIR = d
            try:
                bdc.save_student_ckpt(1, 2, pristine1)
                # simulate a shifted live stack: swap in a different block
                toy.model.layers[1] = copy.deepcopy(toy.model.layers[3])
                back = bdc.load_student_ckpt(pristine1, 1, 2)
            finally:
                bdc.CKPT_DIR = old
        for p0, p1 in zip(pristine1.parameters(), back.parameters()):
            self.assertTrue(torch.equal(p0, p1))

    def test_sweep_order(self):
        bdc = self._load_bdc()
        all_pairs = [(i, i + 1) for i in range(0, 10, 2)]
        pilot = [{"pair": [2, 3], "delta_ppl_pct": 1.0},
                 {"pair": [6, 7], "delta_ppl_pct": 0.5},
                 {"pair": [0, 1], "delta_ppl_pct": 3.0},
                 {"pair": [4, 5], "delta_ppl_pct": 2.0}]
        diag = {(i, i + 1): {"mse_final": float(10 - i)}
                for i in range(0, 10, 2)}
        order = bdc.sweep_order(pilot, all_pairs, diag)
        # pilot pairs first, most collapsible first; rest by mse_final
        self.assertEqual(order[:4], [(6, 7), (2, 3), (4, 5), (0, 1)])
        self.assertEqual(order[4:], [(8, 9)])
        self.assertEqual(sorted(order), sorted(all_pairs))

    def test_replay_committed(self):
        bdc = self._load_bdc()
        toy = make_toy(6)
        orig = list(toy.model.layers)
        s01 = copy.deepcopy(orig[0])
        s45 = copy.deepcopy(orig[4])
        students = {(0, 1): s01, (4, 5): s45}
        get_student = lambda a, b: students[(a, b)]  # noqa: E731
        bdc.replay_committed(toy, get_student, [(0, 1), (4, 5)])
        layers = list(toy.model.layers)
        self.assertEqual(len(layers), 4)
        self.assertIs(layers[0], s01)      # pair (0,1) -> position 0
        self.assertIs(layers[1], orig[2])  # untouched middle blocks shift up
        self.assertIs(layers[2], orig[3])
        self.assertIs(layers[3], s45)      # pair (4,5) -> position 3
        self.assertEqual(toy.config.num_hidden_layers, 4)
        # untouched originals still pristine
        self.assertEqual(len(orig), 6)
