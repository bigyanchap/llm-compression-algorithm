# Revising the Path-Collapse Algorithm — what the prototype taught us

These revisions are grounded in measured results from the prototype
(`README.md`, `experiments/results_llm.json`, `experiments/results_llm_forced.json`)
on SmolLM2-360M-Instruct, not in speculation.

## 1. Make the approximate solver outlier-robust (highest priority)

**Evidence:** 4 of 32 MLP blocks showed catastrophic drift (932–1980) while the
other 28 stayed under ~21. Per-block least-squares MSE grows ~300× from early
to late layers (0.06 → 19.4). This signature — a few blocks exploding — is
classic activation-outlier behavior, and ordinary least squares is
outlier-fragile by construction.

**Revision:** fit with a Huber loss (or hard clip / winsorize the top 0.1% of
activation magnitudes) before solving, and normalize per-channel before the
least-squares step. Expected effect: the 4 outlier blocks stop vetoing the
whole idea; early blocks likely collapse cleanly.

## 2. Replace the fixed drift threshold with Δ-perplexity gating

**Evidence:** with tol=2.0, all 32 blocks rolled back (0% reduction); with
tol=3000, all 32 committed (−57% params, perplexity 10.76 → 29,118 — model
destroyed). A single global max-abs threshold is a blunt instrument: it can't
distinguish "harmless everywhere" from "catastrophic somewhere."

**Revision:** adopt the report's own accuracy-drop indicator — commit a collapse
only if Δ-perplexity (or Δ-accuracy on a validation slice) stays within τ.
Per-token max drift is a proxy; task loss is the ground truth. This also
enables per-block adaptive thresholds: early layers, which fit 300× better,
get collapsed aggressively while late sensitive layers are left alone.

## 3. Few-step distillation fine-tune of fitted layers

**Evidence:** closed-form least squares cannot mimic a gated SwiGLU block
within any strict bound — the forced run proves the solver, not the
threshold, is the bottleneck.

**Revision:** after the least-squares init, run a few hundred gradient steps
distilling the original block's outputs on calibration data (frozen rest of
model). Still orders of magnitude cheaper than full retraining, and a
universal approximator where closed-form is not. This is the most likely
single change to make approximate collapse viable on real LLMs.

## 4. Make paths residual-aware

**Evidence:** the exact phase found **0** foldable chains on both GPT-2-style
and Llama-style architectures. The path model `x_{k+1} = σ(Wx+b)` pretends
residual connections don't exist; in modern LLMs every linear layer sits
between a norm, an activation, an attention op, or a residual add.

**Revision:** extend the graph model to include residual adds as first-class
edges, so candidate paths can span them (e.g., norm→linear→residual as one
approximate unit, or a whole MLP+residual branch collapsed jointly). Without
this, the "exact" half of the algorithm will always report 0% on real LLMs.

## 5. Profile data-dependent linearity before searching

**Revision:** on calibration data, measure per-layer "linearity index" —
fraction of pre-activations where GELU/SiLU operates in its near-linear
regime. Paths scoring high collapse *almost exactly* with plain algebra,
no ε-FDE needed. This turns the search from blind enumeration into a
prioritized hunt, and it's cheap: one calibration forward pass.

## 6. Implement ε-FDE quotient lumping as the real solver

The prototype's least-squares folder was always a placeholder baseline.
The report's proposed Approximate Forward Differential Equivalence —
lumping functionally symmetric paths into a quotient system with an O(ε)
trajectory guarantee — is the principled replacement and should be
benchmarked directly against this prototype's numbers (10.76 → 29,118 is
the baseline to beat).

## 7. Re-calibrate sequentially, not once

**Evidence/limitation:** the prototype calibrates variance statistics once,
on the pristine model. After each committed collapse, downstream inputs
shift, so later fits solve against stale statistics.

**Revision:** after every commit, refresh calibration activations for
downstream blocks before proposing the next collapse (the report's
orchestrator loop implies this; the prototype skipped it for speed).

## 8. Stack collapse with quantization (orthogonal gains)

Path collapse changes *structure* (fewer, dense matrices); quantization
changes *precision*. They multiply: collapse the MLP blocks, then quantize
the dense survivors to 4-bit. Either alone is a 2–4× story; together they
are the realistic route to the 1/10th target on LLMs.

## 9. Treat attention separately from MLPs

Q/K/V/O projections cannot fold through softmax — don't waste search budget
there. MLP blocks hold ~2/3 of LLM parameters; concentrate path collapse on
them and handle attention with complementary methods (head pruning,
low-rank factorization).

## 10. Upgrade greedy search to beam search before reaching for MARL

The full 4-agent Dec-POMDP is heavy machinery. A beam search over path
candidates scored by the report's joint reward
(α·ΔParams + β·ΔFLOPs − λ·ΔLoss − Ω·1[ΔAcc>τ]) captures most of the benefit
with a fraction of the complexity — and gives MARL a strong baseline to
beat if it's ever implemented.
