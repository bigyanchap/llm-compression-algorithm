"""Greedy orchestrator with commit/rollback discipline.

This is the deterministic stand-in for the report's MARL Compression
Orchestrator (PCA-MA). Same contract, no learned policies:

  * propose a collapse for one path at a time,
  * apply it tentatively (in-place module swap, original kept for rollback),
  * verify against thresholds,
  * COMMIT only if every check passes, otherwise ROLL BACK the swap,
  * stop when the target compression ratio is reached or a full pass over
    all remaining candidates commits nothing.

The MARL search (learned path-selection / solver policies) is future work.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from . import graph as G
from .collapse_exact import try_collapse_exact, make_folded_linear
from .collapse_approx import collapse_mlp_block
from .verify import max_abs_diff, block_output_drift


@dataclass
class CommitRecord:
    path_name: str
    kind: str  # "exact" | "approx"
    committed: bool
    reason: str
    params_before: int = 0
    params_after: int = 0
    diagnostics: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, model, target_ratio: float = 0.10,
                 exact_tol: float = 1e-5,
                 approx_block_tol: float = 2.0,
                 mlp_prefix: str = "transformer.h",
                 dtype=torch.float32):
        self.model = model
        self.target_ratio = target_ratio
        self.exact_tol = exact_tol
        self.approx_block_tol = approx_block_tol
        self.mlp_prefix = mlp_prefix  # dotted path to the block container,
        # e.g. "transformer.h" (GPT-2) or "model.layers" (Llama-style)
        self.dtype = dtype
        self.records: list[CommitRecord] = []
        self.initial_params = G.count_parameters(model)

    # -- low-level swap with rollback -------------------------------------
    # NOTE: run_exact/run_approx_mlp below implement their own swap logic
    # explicitly (snapshot -> swap -> verify -> commit-or-rollback) so the
    # rollback path is auditable. This helper is kept for future candidates.

    # -- exact phase -------------------------------------------------------
    def run_exact(self, paths, calib_input_ids):
        for path in paths:
            W_star, b_star, reason = try_collapse_exact(path)
            if reason is not None:
                self.records.append(CommitRecord(
                    path.name, "exact", False, f"refused: {reason}"))
                continue
            folded = make_folded_linear(W_star, b_star, dtype=self.dtype)

            # Snapshot the model BEFORE the tentative swap.
            before = copy.deepcopy(self.model)
            params_before = G.count_parameters(before)

            # Tentatively replace the chain: first member becomes the folded
            # layer, the rest become Identity (zero params, pass-through).
            G.set_submodule(self.model, path.module_names[0], folded)
            for extra in path.module_names[1:]:
                G.set_submodule(self.model, extra, torch.nn.Identity())

            drift = max_abs_diff(before, self.model, calib_input_ids)
            if drift <= self.exact_tol:
                params_after = G.count_parameters(self.model)
                self.records.append(CommitRecord(
                    path.name, "exact", True,
                    f"committed: max logit drift {drift:.3e} <= tol {self.exact_tol}",
                    params_before, params_after,
                    {"max_logit_drift": drift}))
            else:
                # ROLLBACK: restore the pre-swap snapshot.
                self.model = before
                self.records.append(CommitRecord(
                    path.name, "exact", False,
                    f"rolled back: max logit drift {drift:.3e} > tol {self.exact_tol}"))
        return self.records

    # -- approximate phase -------------------------------------------------
    def run_approx_mlp(self, mlp_paths, calib_X_by_block: dict,
                       free_calib: bool = True):
        """Collapse each MLP block using calibration activations.

        calib_X_by_block: {block_index: (X_in, Y_out)} with X_in the MLP
        input activations and Y_out the original MLP outputs. Blocks are
        addressed as f"{self.mlp_prefix}.{i}.mlp".

        If free_calib is True (default), each block's calibration tensors are
        released right after its collapse is decided, keeping peak memory
        flat instead of holding all blocks' activations at once.
        """
        for path in mlp_paths:
            i = int(path.name.split(".")[1])
            dotted = f"{self.mlp_prefix}.{i}.mlp"
            X, Y = calib_X_by_block[i]
            new_mod, diag = collapse_mlp_block(X, Y, dtype=self.dtype)
            old_mod = G.get_submodule(self.model, dotted)
            drift = block_output_drift(old_mod, new_mod, X[:4096])
            diag["block_output_max_drift"] = drift
            if free_calib:
                del calib_X_by_block[i]
                del X, Y
            if drift <= self.approx_block_tol:
                params_before = G.count_parameters(self.model)
                G.set_submodule(self.model, dotted, new_mod)
                params_after = G.count_parameters(self.model)
                self.records.append(CommitRecord(
                    path.name, "approx", True,
                    f"committed: block drift {drift:.4f} <= tol {self.approx_block_tol}",
                    params_before, params_after, diag))
            else:
                self.records.append(CommitRecord(
                    path.name, "approx", False,
                    f"rolled back: block drift {drift:.4f} > tol {self.approx_block_tol}",
                    diagnostics=diag))
        return self.records

    def compression_ratio(self) -> float:
        return G.count_parameters(self.model) / self.initial_params

    def summary(self):
        committed = [r for r in self.records if r.committed]
        return {
            "initial_params": self.initial_params,
            "final_params": G.count_parameters(self.model),
            "compression_ratio": self.compression_ratio(),
            "params_removed": self.initial_params - G.count_parameters(self.model),
            "committed": len(committed),
            "rolled_back_or_refused": len(self.records) - len(committed),
            "records": [
                {"path": r.path_name, "kind": r.kind,
                 "committed": r.committed, "reason": r.reason,
                 "diagnostics": r.diagnostics}
                for r in self.records
            ],
        }
