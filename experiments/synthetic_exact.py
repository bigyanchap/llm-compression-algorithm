"""Synthetic demo: exact collapse of a genuine linear chain.

Builds a small feedforward net that is PURELY linear in one span
(Linear -> Linear -> Linear, no activations, with biases), folds it with the
exact folder, and asserts the outputs match to within 1e-6 on random inputs.

Also exercises the orchestrator's commit path and the refusal path
(a chain broken by GELU must be refused, never folded).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.graph import find_linear_chains, count_parameters, Path
from src.collapse_exact import try_collapse_exact
from src.orchestrator import Orchestrator

torch.manual_seed(42)


class ChainNet(nn.Module):
    """y = L3(L2(L1(x))) with L1,L2,L3 linear (foldable), then a head."""

    def __init__(self):
        super().__init__()
        self.chain = nn.Sequential(
            nn.Linear(8, 16), nn.Linear(16, 16), nn.Linear(16, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.chain(x))


class BrokenNet(nn.Module):
    """Same shape, but GELU breaks the chain: must be refused."""

    def __init__(self):
        super().__init__()
        self.chain = nn.Sequential(
            nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.chain(x)


def main():
    print("== synthetic exact-collapse demo ==")
    model = ChainNet().eval()
    paths = find_linear_chains(model.chain)
    print(f"foldable chains found: {len(paths)} "
          f"(modules: {[p.module_names for p in paths]})")
    assert len(paths) == 1, "expected exactly one foldable chain"

    # Wrap the chain in a tiny model whose forward == chain output, so the
    # orchestrator's full-model logit check is meaningful.
    class ChainOnly(nn.Module):
        def __init__(self, chain):
            super().__init__()
            self.chain = chain

        def forward(self, x):
            return self.chain(x)

    probe = ChainOnly(model.chain).eval()
    calib = torch.randn(64, 8)

    # Fix dotted names: paths were found under model.chain; re-root them so
    # the orchestrator can address them on `probe`.
    path = paths[0]
    path.module_names = [f"chain.{n}" for n in path.module_names]

    orch = Orchestrator(probe, exact_tol=1e-5)
    before_params = orch.initial_params
    orch.run_exact([path], calib_input_ids=calib)

    rec = orch.records[0]
    print(f"decision: {'COMMITTED' if rec.committed else 'ROLLED BACK'}")
    print(f"reason: {rec.reason}")
    assert rec.committed, "exact fold of a pure linear chain must commit"

    x = torch.randn(256, 8)
    with torch.no_grad():
        y_orig = model.chain(x)
        y_new = orch.model.chain(x)
    max_diff = float((y_orig - y_new).abs().max())
    print(f"max abs diff on 256 random inputs: {max_diff:.3e}")
    assert max_diff < 1e-6, f"NOT lossless: {max_diff}"

    after_params = orch.summary()["final_params"]
    print(f"params: {before_params} -> {after_params} "
          f"(-{before_params - after_params})")
    print("lossless: True")

    # Refusal check: GELU-broken chain must never fold.
    broken_paths = find_linear_chains(BrokenNet().chain)
    assert broken_paths == [], "gelu-broken chain must yield no candidates"
    fake = Path(name="gelu-chain", module_names=["0", "2"],
                modules=[nn.Linear(8, 16), nn.Linear(16, 4)],
                has_nonlinearity=True, kind="linear_chain")
    W, b, reason = try_collapse_exact(fake)
    assert W is None and reason, "nonlinear path must be refused"
    print(f"refusal check passed: {reason[:80]}...")
    print("ALL SYNTHETIC CHECKS PASSED")


if __name__ == "__main__":
    main()
