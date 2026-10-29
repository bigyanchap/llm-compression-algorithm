# Path Collapse Algorithm — Research Prototype

A working prototype of the **iterative neural path collapse** compression idea:
walk a neural network's computation graph, find paths of nodes/weights
`x1 w1 → x2 w2 → … → xn wn`, and collapse each path into a single
node-weight pair — **but only if the net outcome is unchanged** (exact
folding) or stays within a verified tolerance (approximate folding).
Repeat until the model reaches 1/10th size or no further collapse passes
verification.

Based on the deep-research report *Algorithmic Framework for Iterative Neural
Path Collapse and Aggregation via Multi-Agent Optimization* (see
`/tmp/gemini_report.txt` on the author's machine).

## Honest scope statement

This is a **research prototype, not the full system from the report**:

- ✅ Exact affine folding of consecutive linear layers (bit-for-bit, verified).
- ✅ Approximate collapse of LLM MLP blocks (GPT-2's `c_fc → gelu → c_proj`;
  SmolLM2/Qwen's gated SwiGLU `gate_proj / up_proj / down_proj`) into one
  Linear via least-squares fitting on calibration activations +
  REPAIR-style variance calibration (no retraining).
- ✅ Greedy orchestrator with **commit/rollback discipline** (per-path
  verification; rollback on failure) — the deterministic stand-in for the
  report's Compression Orchestrator agent.
- ❌ **No MARL.** The report's 4-agent reinforcement-learning search
  (Path Search / Equivalent Solver / Verification / Orchestrator agents
  trained as a Dec-POMDP) is **not implemented** — path candidates here are
  enumerated deterministically. MARL is documented as future work below.
- ❌ **No ε-FDE quotient lumping.** The approximate folder is least-squares +
  variance calibration, a simpler baseline. Its quality limits are measured
  honestly in the experiments.
- ❌ **No compiler lowering** (TVM/TensorRT fusion passes).

## Layout

```
src/
  graph.py            module-level graph walk; GPT-2 path enumeration
  collapse_exact.py   exact affine folding (refuses nonlinear paths)
  collapse_approx.py  least-squares fit + REPAIR-style variance calibration
  verify.py           max-diff checks, perplexity
  orchestrator.py     greedy commit/rollback loop
experiments/
  synthetic_exact.py  tiny linear-chain net: proves lossless exact folding
  llm_collapse.py     LLM on CPU (default SmolLM2-360M-Instruct): exact phase
                      + approximate MLP collapse; writes results_llm.json
  results_llm.json    machine-readable results (written by llm_collapse.py)
tests/
  test_exact_fold.py  unit tests (fold math, refusal, calibration)
data/
  wiki.valid.tokens   WikiText-2 validation text (ppl eval + calibration)
```

## Install & run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers

# unit tests (no model download needed)
python -m unittest discover -s tests -v

# synthetic exact-fold demo (fast, CPU)
python experiments/synthetic_exact.py

# LLM experiment (downloads SmolLM2-360M-Instruct ~720MB on first run; CPU)
# NOTE: if huggingface.co weight downloads stall, the weights can be fetched
# from the ModelScope mirror instead:
#   https://www.modelscope.cn/models/HuggingFaceTB/SmolLM2-360M-Instruct
# and --model-id pointed at the local directory.
python experiments/llm_collapse.py \
    --wikitext data/wiki.valid.tokens \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
    --calib-chunks 8 --eval-chunks 16 --approx-tol 2.0
```

Perplexity is computed over non-overlapping 1024-token chunks of WikiText-2
validation text (pytorch/examples mirror). Calibration activations come from
a disjoint earlier slice of the same file.

## Results

### Unit tests — 10/10 pass
`python -m unittest discover -s tests` — folding math (2-layer to 1e-6,
3-layer to 1e-4 float32 roundoff), hand-computed formula spot check, refusal
of a GELU-containing path, `nn.Linear` weight layout, chain enumeration
breaking on activations, least-squares recovery of a true linear map,
variance calibration restoring per-channel mean/std, end-to-end MLP-block
collapse on synthetic data.

### Synthetic exact-fold demo — lossless
`python experiments/synthetic_exact.py` — a 3-layer pure-linear chain
(`Linear(8,16) → Linear(16,16) → Linear(16,4)`, with biases) folds into one
layer: **max abs diff 0.0 on 256 random inputs**, orchestrator COMMITTED
(drift 1.19e-07 ≤ tol 1e-5), params 484 → 36. A GELU-broken chain is refused,
never folded.

### SmolLM2-360M-Instruct on WikiText-2 — strict verification (tol = 2.0)
`python experiments/llm_collapse.py --model-id models/SmolLM2-360M-Instruct`
(CPU, fp32; WikiText-2 validation, non-overlapping 1024-token chunks):

| metric | before | after |
|---|---|---|
| params | 361,821,120 | 361,821,120 (**0.00%** reduction) |
| perplexity (16,384 tokens) | 10.76 | 10.76 |
| max logit drift (sample prompts) | — | 0.0000 |

- **Exact phase:** 0 foldable chains — every Linear is separated by a norm,
  activation, attention operator, or residual add. Nothing forced.
- **Approx phase:** 32 MLP blocks proposed, **32 rolled back, 0 committed**.
  Per-block max abs drift ranged 8.6 – 1980.4 (tol 2.0); least-squares MSE
  ranged 0.06 (early blocks) to 19.4 (late blocks), with blocks 3/13/30/31
  showing catastrophic drift (932 – 1980) consistent with massive activation
  outliers. The verification bar did its job: a single linear layer cannot
  mimic a gated SwiGLU MLP block within a strict per-token bound, so the
  orchestrator correctly refused every collapse rather than silently
  degrading the model.

### Diagnostic: forced commit (tol = 3000, all 32 blocks collapse)
To measure the quality cliff the threshold protects against
(`experiments/results_llm_forced.json`):

| metric | before | after |
|---|---|---|
| params | 361,821,120 | 155,413,440 (**57.05%** reduction, ratio 0.43) |
| perplexity (16,384 tokens) | 10.76 | **29,117.82** |
| max logit drift | — | 26.47 / 20.96 |

Collapsing every gated SwiGLU MLP into one linear layer removes 57% of
parameters — but perplexity explodes from 10.76 to ~29,000. The model is
destroyed. This is exactly why the strict run's verification bar refused
all 32: the commit/rollback discipline is load-bearing, not decorative.
A better *solver* (not a looser threshold) is what would make approximate
collapse viable — see Future work.

## What worked / what didn't (prototype assessment)

- **Exact folding: works, provably.** On a synthetic 3-layer linear chain the
  folded layer reproduces the original outputs to within floating-point
  roundoff (max diff 0.0 on 256 random inputs), and the folder correctly
  *refuses* any path containing a nonlinearity.
- **Exact phase on real LLMs: nothing to fold, as predicted.** Both GPT-2
  and Llama-style architectures (SmolLM2, Qwen2.5) have zero
  consecutive-linear chains — every Linear is separated by a norm,
  activation, attention operator, or residual add. The orchestrator reports
  this instead of forcing a collapse.
- **Approximate phase (SmolLM2):** results above. Least-squares + variance
  calibration is a weak approximator for a full gated-SwiGLU MLP block; the
  forced-commit diagnostic proves it: −57% params, perplexity 10.76 → 29,118.
  This is the honest baseline against which a future ε-FDE solver should be
  compared. Notably, per-block MSE grows ~300× from early to late layers
  (0.06 → 19.4), and 4 blocks show catastrophic drift consistent with
  massive activation outliers — both point at concrete solver improvements
  (outlier-robust fitting, per-block adaptive thresholds, Δ-perplexity-gated
  greedy commits).

## Variation 1: residual block-delta collapse (distillation-flavored)

**The rethink.** The original prototype hunted collapsible node-chains
*inside* transformer blocks and found zero on real LLMs — every linear
layer sits behind a norm, an activation, an attention op, or a residual
add. Variation 1 redefines what a "path" is: a transformer block computes
`x_{k+1} = x_k + Block_k(x_k)`, so the block's contribution is a *delta*
on the residual stream. Collapse *across* blocks instead: distill an
adjacent pair `(Block_k, Block_{k+1})` into ONE block `Block'` — a deep
copy of `Block_k` trained (300 Adam steps, MSE) so `Block'(h_k) ~= h_{k+2}`
on cached pristine boundary states — splice it in tentatively, and keep
the commit/rollback discipline with a Δ-perplexity gate (τ = 5% relative
vs the pristine baseline).

Honest note: this relaxes the original "collapse into one x1w1" to
"collapse into the cheapest verified equivalent". It is
distillation-flavored, and we say so. What it keeps from the original
idea: the search → collapse → verify-or-rollback loop, and the rule that
nothing is kept unless validation says it's safe.

**Protocol** (`experiments/block_delta_collapse.py`, `src/block_delta.py`):
distill all 16 non-overlapping adjacent pairs → pilot on 4 spread pairs
with a drop-one-block (no distillation) baseline → greedy sweep in
collapsibility order (pilot Δ-ppl, then distillation MSE), re-measuring
perplexity after each commit, gate always against the pristine baseline.
Checkpointed per pair under `experiments/checkpoints_block_delta/`, so
interrupted runs resume exactly. 24/24 unit tests pass
(`tests/test_block_delta.py`, incl. checkpoint/resume logic).

**Results** — SmolLM2-360M-Instruct, WikiText-2 validation, 16,384 tokens
(CPU fp32; `experiments/results_block_delta.json`):

| metric | before | after |
|---|---|---|
| depth | 32 blocks | 31 blocks |
| params | 361,821,120 | 351,988,800 (**−2.72%**) |
| perplexity | 10.68 | 11.09 (+3.89%, within τ = 5%) |

- **Committed: 1 pair — (14,15). Rolled back: 15 of 16.**
- Pilot (Δ-perplexity vs pristine 10.68):

| pair | distilled splice | drop-one-block baseline |
|---|---|---|
| (6,7) | 11.22 (+5.07%) | 11.38 (+6.63%) |
| (14,15) | 11.09 (+3.89%) | 11.48 (+7.52%) |
| (22,23) | 12.59 (+17.97%) | 13.17 (+23.37%) |
| (28,29) | 12.14 (+13.73%) | 13.26 (+24.24%) |

- Distillation MSE (init → final, 300 steps): early pairs fit well
  ((0,1): 1.26 → 0.32; (4,5): 1.30 → 0.62), mid pairs moderately
  ((14,15): 3.33 → 1.38; (20,21): 8.86 → 3.34), late pairs poorly —
  final MSE grows ~2500× from early to late layers
  ((26,27): 67.5 → 20.0; (28,29): 140 → 38.3; (30,31): 1573 → 813),
  consistent with massive late-layer activation outliers. Pair (2,3) is
  pathological (1290 → 21.9; splicing it explodes perplexity +78.9%).

**Honest assessment.** Variation 1 moves the needle where the original
got 0%: **0% → 2.72% verified compression with the model intact** — a
real result, not a forced one. Distillation consistently beats the
drop-only baseline by 0.2–1.1 perplexity points, proving the student
learns something beyond deletion. But it is a long way from 1/10th:
15 of 16 pairs cannot be absorbed into a single block within a 5%
perplexity budget — one block's delta simply cannot cover two blocks'
deltas in a 32-layer model. The failure pattern is informative: early
pairs distill cleanly, late pairs drown in outliers, and one early pair
(2,3) is catastrophically sensitive. The next variations to try:
outlier-robust distillation (Huber/clipping), distilling into a
*slightly larger* student (e.g. wider MLP, still cheaper than two
blocks), and per-pair adaptive τ.

## Future work

1. **MARL search (PCA-MA):** replace deterministic enumeration with learned
   Path Search / Solver / Verification policies and the joint reward from
   the report.
2. **ε-FDE quotient lumping** as the approximate solver, replacing
   least-squares.
3. **Per-iteration re-calibration** (current prototype calibrates once on the
   pristine model; sequential commits shift downstream inputs).
4. **Compiler lowering:** export the collapsed graph to ONNX/TVM Relax and
   measure real latency, not just parameter counts.
5. **Better verification:** replace the fixed per-block drift threshold with
   the report's accuracy-drop indicator (ΔAcc > τ) on a validation set.
