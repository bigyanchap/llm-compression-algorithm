"""LLM path-collapse experiment (default: SmolLM2-360M-Instruct).

Pipeline:
  1. Load an instruction-tuned causal LM on CPU, eval mode. Record baseline
     perplexity on a held-out slice of WikiText-2 validation text.
  2. EXACT phase: enumerate foldable linear chains. Llama-style architectures
     (SmolLM2, Qwen2.5) are expected to have ZERO -- every Linear sits behind
     a norm, activation, attention operator, or residual add. Reported
     honestly, whatever the count is.
  3. APPROXIMATE phase: for each MLP block, capture calibration activations
     (MLP input/output on separate calibration text), fit a single Linear by
     least squares, apply REPAIR-style variance calibration, and let the
     greedy orchestrator COMMIT or ROLL BACK each block against a documented
     per-block drift threshold.
  4. Report: params before/after, % reduction, perplexity before/after on
     the eval slice, max logit drift on sample prompts vs the pristine model.

Calibration activations are captured in ONE pass over the pristine model
(documented limitation: sequential commits shift downstream inputs, which a
production system would re-calibrate per iteration).

Usage:
  python experiments/llm_collapse.py \
      --wikitext data/wiki.valid.tokens \
      --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
      --calib-chunks 8 --eval-chunks 16 --approx-tol 2.0
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src import graph as G
from src.orchestrator import Orchestrator
from src.verify import perplexity


def load_wikitext_valid(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return "\n".join(ln for ln in lines if ln.strip())


def detect_arch(model):
    """Return (arch_name, mlp_prefix, enumerator)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return ("gpt2", "transformer.h", G.enumerate_gpt2_paths)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return ("llama-like", "model.layers",
                lambda m: G.enumerate_llama_like_paths(m))
    raise RuntimeError("unrecognized model architecture: "
                       + type(model).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--fallback-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--calib-chunks", type=int, default=8)
    ap.add_argument("--eval-chunks", type=int, default=16)
    ap.add_argument("--chunk-len", type=int, default=1024)
    ap.add_argument("--approx-tol", type=float, default=2.0)
    ap.add_argument("--exact-tol", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results_llm.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/7] loading {args.model_id} ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.float32).eval()
    except Exception as exc:
        print(f"      primary model failed ({type(exc).__name__}); "
              f"falling back to {args.fallback_model_id}", flush=True)
        args.model_id = args.fallback_model_id
        tok = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params_before = G.count_parameters(model)
    arch, mlp_prefix, enumerator = detect_arch(model)
    n_blocks = len(G.get_submodule(model, mlp_prefix))
    print(f"      arch: {arch}, blocks: {n_blocks}, params: {params_before:,}")

    print("[2/7] tokenizing WikiText-2 validation text ...", flush=True)
    text = load_wikitext_valid(args.wikitext)
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
    print(f"      vocab: {tok.vocab_size}, total tokens: {ids.numel():,}")
    L = args.chunk_len
    if ids.numel() < (args.calib_chunks + args.eval_chunks) * L:
        raise RuntimeError("not enough tokens in wikitext file for the "
                           "requested chunk counts")
    calib_ids = ids[:args.calib_chunks * L]
    eval_ids = ids[args.calib_chunks * L:(args.calib_chunks + args.eval_chunks) * L]
    print(f"      calib tokens: {calib_ids.numel():,}, "
          f"eval tokens: {eval_ids.numel():,}")

    print("[3/7] baseline perplexity (pristine model) ...", flush=True)
    t = time.time()
    ppl_before, n_tok = perplexity(model, eval_ids, max_len=L)
    print(f"      perplexity: {ppl_before:.2f} over {n_tok:,} tokens "
          f"({time.time() - t:.1f}s)")

    print("[4/7] enumerating paths ...", flush=True)
    paths = enumerator(model)
    print(f"      exact fold candidates: {len(paths['exact_candidates'])}")
    print(f"      mlp approx candidates: {len(paths['mlp_paths'])}")
    if paths["mlp_paths"]:
        print(f"      e.g. {paths['mlp_paths'][0].name}: "
              f"{paths['mlp_paths'][0].notes}")

    orch = Orchestrator(model, target_ratio=0.10,
                        exact_tol=args.exact_tol,
                        approx_block_tol=args.approx_tol,
                        mlp_prefix=mlp_prefix)

    print("[5/7] EXACT phase ...", flush=True)
    if paths["exact_candidates"]:
        orch.run_exact(paths["exact_candidates"],
                       calib_input_ids=calib_ids[:L].unsqueeze(0))
    else:
        print("      no exactly-foldable linear chains "
              "(every Linear is separated by a norm, activation, attention "
              "operator, or residual add) -- nothing to do, as expected.")
    model = orch.model

    print("[6/7] capturing MLP calibration activations ...", flush=True)
    X_list = [[] for _ in range(n_blocks)]
    Y_list = [[] for _ in range(n_blocks)]

    def make_hook(i):
        def hook(mod, inp, out):
            X_list[i].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
            Y_list[i].append(out.detach().reshape(-1, out.shape[-1]))
        return hook

    layers = G.get_submodule(model, mlp_prefix)
    handles = [layers[i].mlp.register_forward_hook(make_hook(i))
               for i in range(n_blocks)]
    t = time.time()
    with torch.no_grad():
        for c in range(args.calib_chunks):
            chunk = calib_ids[c * L:(c + 1) * L].unsqueeze(0)
            model(chunk)
            if (c + 1) % 2 == 0 or c + 1 == args.calib_chunks:
                print(f"      calib chunk {c + 1}/{args.calib_chunks} "
                      f"({time.time() - t:.1f}s)", flush=True)
    for h in handles:
        h.remove()
    calib = {}
    for i in range(n_blocks):
        X = torch.cat(X_list[i], dim=0)
        Y = torch.cat(Y_list[i], dim=0)
        calib[i] = (X, Y)
        # Release the per-chunk lists immediately: without this, the lists
        # AND the concatenated tensors coexist (2x memory) and OOM the box.
        X_list[i].clear()
        Y_list[i].clear()
    print(f"      captured: {n_blocks} blocks, "
          f"{tuple(calib[0][0].shape)} activations each")
    del X_list, Y_list

    print("      capturing pristine prompt logits (for final drift check) ...",
          flush=True)
    prompts = ["The meaning of life is",
               "In a shocking finding, scientists discovered"]
    pristine_logits = {}
    with torch.no_grad():
        for pr in prompts:
            ids_p = torch.tensor(tok(pr)["input_ids"]).unsqueeze(0)
            pristine_logits[pr] = model(ids_p).logits.detach()
    # NOTE: no full-model deepcopy -- the pristine logits above are all the
    # "before" state we need, keeping memory flat on small machines.

    print("[7/7] APPROXIMATE phase (greedy commit/rollback) ...", flush=True)
    t = time.time()
    orch.run_approx_mlp(paths["mlp_paths"], calib)
    print(f"      done in {time.time() - t:.1f}s")
    model = orch.model
    summ = orch.summary()
    for r in summ["records"]:
        if r["kind"] == "approx":
            d = r["diagnostics"]
            print(f"      {r['path']}: "
                  f"{'COMMITTED' if r['committed'] else 'rolled back'} -- "
                  f"{r['reason']} "
                  f"(mse pre-cal {d.get('mse_before_calibration', float('nan')):.4f}, "
                  f"post-cal {d.get('mse_after_calibration', float('nan')):.4f})")

    params_after = summ["final_params"]
    reduction_pct = 100.0 * (params_before - params_after) / params_before
    print(f"      params: {params_before:,} -> {params_after:,} "
          f"({reduction_pct:.2f}% reduction, ratio {summ['compression_ratio']:.4f})")

    print("[eval] perplexity after collapse ...", flush=True)
    t = time.time()
    ppl_after, _ = perplexity(model, eval_ids, max_len=L)
    print(f"      perplexity: {ppl_after:.2f} ({time.time() - t:.1f}s)")

    print("[eval] max logit drift on sample prompts ...", flush=True)
    drifts = {}
    with torch.no_grad():
        for pr in prompts:
            ids_p = torch.tensor(tok(pr)["input_ids"]).unsqueeze(0)
            new_logits = model(ids_p).logits
            d = float((pristine_logits[pr] - new_logits).abs().max())
            drifts[pr] = d
            print(f"      {pr!r}: max abs logit drift {d:.4f}")

    import math

    def _safe(x):
        return round(x, 4) if math.isfinite(x) else None

    result = {
        "model_id": args.model_id,
        "arch": arch,
        "n_blocks": n_blocks,
        "params_before": params_before,
        "params_after": params_after,
        "reduction_pct": round(reduction_pct, 2),
        "compression_ratio": round(summ["compression_ratio"], 4),
        "perplexity_before": _safe(ppl_before),
        "perplexity_after": _safe(ppl_after),
        "eval_tokens": n_tok,
        "calib_tokens": int(calib_ids.numel()),
        "exact_candidates": len(paths["exact_candidates"]),
        "approx_committed": sum(1 for r in summ["records"]
                                if r["kind"] == "approx" and r["committed"]),
        "approx_rolled_back": sum(1 for r in summ["records"]
                                  if r["kind"] == "approx" and not r["committed"]),
        "approx_tol": args.approx_tol,
        "prompt_drifts": {k: _safe(v) for k, v in drifts.items()},
        "records": summ["records"],
        "elapsed_s": round(time.time() - t0, 1),
        "notes": ("perplexity: non-overlapping 1024-token chunks, WikiText-2 "
                  "validation text (pytorch/examples mirror); calibration "
                  "captured in one pristine pass; approx commit threshold: "
                  f"per-block max abs output drift <= {args.approx_tol}"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"results written to {args.out}")
    print(f"TOTAL elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
