"""Variation 1: residual block-delta collapse (SmolLM2-360M-Instruct).

A transformer block computes x_{k+1} = x_k + Block_k(x_k) on the residual
stream. This experiment collapses ACROSS blocks instead of inside them:
each adjacent pair (Block_k, Block_{k+1}) is distilled into ONE block
(a deep copy of Block_k trained so Block'(h_k) ~= h_{k+2}), then spliced in
tentatively. The orchestrator COMMITS the splice only if validation
perplexity stays within tau (5% relative) of the pristine baseline,
otherwise it ROLLS BACK. This is distillation-flavored -- an honest
relaxation of "collapse into one x1w1" to "collapse into the cheapest
verified equivalent".

Protocol:
  PILOT: distill + evaluate 4 spread pairs (6,7), (14,15), (22,23), (28,29);
         also measure plain drop-one-block (no distillation) as baseline.
  SWEEP: distill the remaining pairs, then commit greedily -- pilot pairs
         first in pilot-measured collapsibility order (lowest d-ppl first),
         then the rest by distillation MSE -- re-measuring perplexity after
         each commit. Gate is always against the PRISTINE baseline.

Checkpointing: distilled students, distill diagnostics, pilot records and
sweep records are saved under experiments/checkpoints_block_delta/ as they
complete, so an interrupted run resumes without redoing finished work.
Everything is deterministic (fixed seeds), so resumed state is identical
to an uninterrupted run.

Writes experiments/results_block_delta.json.
"""

import argparse
import copy
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src import block_delta as BD
from src.verify import perplexity


PILOT_PAIRS = [(6, 7), (14, 15), (22, 23), (28, 29)]
TAU = 0.05
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "checkpoints_block_delta")


# ---------------------------------------------------------------------------
# checkpoint helpers (module-level so unit tests can exercise them)
# ---------------------------------------------------------------------------

def ckpt_key(a: int, b: int) -> str:
    return f"{a},{b}"


def parse_ckpt_key(key: str) -> tuple[int, int]:
    a, b = key.split(",")
    return int(a), int(b)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def student_ckpt_path(a: int, b: int) -> str:
    return os.path.join(CKPT_DIR, f"student_{a}_{b}.pt")


def save_student_ckpt(a: int, b: int, student: nn.Module) -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(student.state_dict(), student_ckpt_path(a, b))


def load_student_ckpt(pristine_layer: nn.Module, a: int, b: int) -> nn.Module:
    """Fresh deep copy of the PRISTINE Block_k with checkpointed weights.

    Takes the pristine layer module (not the live model): after commits the
    live stack is shifted, so model.model.layers[a] may be a different block.
    The student is a module, so it can be spliced into any stack position.
    """
    student = copy.deepcopy(pristine_layer)
    sd = torch.load(student_ckpt_path(a, b), map_location="cpu",
                    weights_only=True)
    student.load_state_dict(sd)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student


def sweep_order(pilot: list, all_pairs: list,
                distill_diag: dict) -> list:
    """Deterministic greedy order: pilot pairs by measured collapsibility
    (lowest delta_ppl_pct first), then the rest by distillation MSE."""
    pilot_pairs = [tuple(r["pair"]) for r in pilot]
    pilot_order = sorted(
        pilot_pairs,
        key=lambda pr: next(r["delta_ppl_pct"] for r in pilot
                            if tuple(r["pair"]) == pr))
    rest = [pr for pr in all_pairs if pr not in pilot_pairs]
    rest.sort(key=lambda pr: distill_diag[pr]["mse_final"])
    return pilot_order + rest


def replay_committed(model, get_student, committed: list) -> None:
    """Re-apply committed splices (in commit order) to a pristine stack.

    Reproduces exactly the stack state the sweep had after those commits:
    pairs are non-overlapping, so the positional shift of pair (a,b) is the
    number of already-committed pairs fully below it. Students are loaded
    on demand and released -- only the installed ones stay resident.
    """
    done: list = []
    for (a, b) in committed:
        shift = sum(1 for (x, y) in done if y < a)
        pa = a - shift
        cur = list(model.model.layers)
        BD.install_layers(
            model, nn.ModuleList(cur[:pa] + [get_student(a, b)]
                                 + cur[pa + 2:]))
        done.append((a, b))


def load_wikitext_valid(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return "\n".join(ln for ln in lines if ln.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext", default="data/wiki.valid.tokens")
    ap.add_argument("--model-id", default="models/SmolLM2-360M-Instruct")
    ap.add_argument("--calib-tokens", type=int, default=4096)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--chunk-len", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--distill-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results_block_delta.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[1/6] loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_blocks = len(model.model.layers)
    params_before = BD.count_parameters(model)
    per_block = BD.count_parameters(model.model.layers[0])
    print(f"      blocks: {n_blocks}, params: {params_before:,}, "
          f"per-block: {per_block:,}", flush=True)
    orig_layers = list(model.model.layers)  # pristine, never mutated

    def set_stack(modules):
        BD.install_layers(model, nn.ModuleList(list(modules)))

    def reset_pristine():
        set_stack(orig_layers)

    print("[2/6] tokenizing ...", flush=True)
    text = load_wikitext_valid(args.wikitext)
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
    need = args.calib_tokens + args.eval_tokens
    assert ids.numel() >= need, "not enough wikitext tokens"
    calib_ids = ids[:args.calib_tokens]
    eval_ids = ids[args.calib_tokens:args.calib_tokens + args.eval_tokens]
    print(f"      calib: {calib_ids.numel():,}, eval: {eval_ids.numel():,}",
          flush=True)

    print("[3/6] baseline perplexity (pristine) ...", flush=True)
    t = time.time()
    ppl_pristine, n_tok = perplexity(model, eval_ids, max_len=args.chunk_len)
    print(f"      ppl: {ppl_pristine:.2f} over {n_tok:,} tokens "
          f"({time.time() - t:.1f}s)", flush=True)

    all_pairs = [(i, i + 1) for i in range(0, n_blocks, 2)]
    assert len(all_pairs) == n_blocks // 2

    # ---- distillation, with resume ----
    diag_path = os.path.join(CKPT_DIR, "distill_diag.json")
    ckpt_diag = load_json(diag_path, {})
    distill_diag: dict = {}
    for k, v in ckpt_diag.items():
        distill_diag[parse_ckpt_key(k)] = v
    missing = [p for p in all_pairs if p not in distill_diag]

    if missing:
        print(f"[4/6] caching boundary states h_0..h_N "
              f"({len(missing)} pairs still to distill) ...", flush=True)
        t = time.time()
        bounds, offsets = BD.cache_boundary_states(
            model, calib_ids, n_tokens=args.calib_tokens,
            seq_len=args.seq_len)
        print(f"      {len(bounds)} boundaries x {tuple(bounds[0].shape)} "
              f"({time.time() - t:.1f}s)", flush=True)
        print(f"[5/6] distilling {len(missing)} pairs "
              f"({args.distill_steps} steps each) ...", flush=True)
        t = time.time()
        for idx, (a, b) in enumerate(missing):
            st, dg = BD.distill_pair(model, bounds, offsets, a,
                                     steps=args.distill_steps, lr=args.lr,
                                     seed=args.seed + a)
            save_student_ckpt(a, b, st)
            del st  # keep memory flat; students reload on demand
            distill_diag[(a, b)] = dg
            ckpt_diag[ckpt_key(a, b)] = dg
            save_json(diag_path, ckpt_diag)
            print(f"      pair ({a},{b}) [{idx + 1}/{len(missing)} new, "
                  f"{len(distill_diag)}/{len(all_pairs)} total]: "
                  f"mse {dg['mse_init']:.4f} -> {dg['mse_final']:.6f} "
                  f"({time.time() - t:.1f}s total)", flush=True)
        del bounds  # free ~0.5GB of cached activations; students are trained
    else:
        print("[4/6]+[5/6] all 16 pairs already distilled -- "
              "checkpoints on disk ...", flush=True)

    def get_student(a: int, b: int) -> nn.Module:
        # pristine layer: the live stack may be shifted by commits
        return load_student_ckpt(orig_layers[a], a, b)

    print(f"      {len(distill_diag)}/{len(all_pairs)} students checkpointed",
          flush=True)

    def eval_current():
        p, _ = perplexity(model, eval_ids, max_len=args.chunk_len)
        return p

    def delta(p):
        return 100.0 * (p - ppl_pristine) / ppl_pristine

    # ---- PILOT (with resume; checkpointed per pair) ----
    pilot_path = os.path.join(CKPT_DIR, "pilot.json")
    pilot = load_json(pilot_path, None)
    if pilot is None:
        pilot = []
    done_pilot = {tuple(r["pair"]) for r in pilot}
    todo_pilot = [pr for pr in PILOT_PAIRS if pr not in done_pilot]
    if todo_pilot:
        print(f"[6a/6] PILOT: {len(todo_pilot)} pairs to evaluate "
              f"({len(done_pilot)} from checkpoint) ...", flush=True)
        reset_pristine()
        for (a, b) in todo_pilot:
            # distilled splice into pristine stack
            st = get_student(a, b)
            set_stack(orig_layers[:a] + [st] + orig_layers[b + 1:])
            del st
            t = time.time()
            p_splice = eval_current()
            # naive baseline: drop block b, no distillation
            set_stack(orig_layers[:b] + orig_layers[b + 1:])
            p_drop = eval_current()
            reset_pristine()
            rec = {"pair": [a, b],
                   "mse_init": distill_diag[(a, b)]["mse_init"],
                   "mse_final": distill_diag[(a, b)]["mse_final"],
                   "ppl_spliced": p_splice, "delta_ppl_pct": delta(p_splice),
                   "ppl_drop_baseline": p_drop,
                   "delta_ppl_drop_pct": delta(p_drop)}
            pilot.append(rec)
            save_json(pilot_path, pilot)
            print(f"      pair ({a},{b}): distilled ppl {p_splice:.2f} "
                  f"({delta(p_splice):+.2f}%) | drop-only ppl {p_drop:.2f} "
                  f"({delta(p_drop):+.2f}%)", flush=True)
    else:
        print("[6a/6] PILOT: all 4 pairs loaded from checkpoint", flush=True)

    # ---- SWEEP (with resume) ----
    print("[6b/6] SWEEP: greedy commit/rollback ...", flush=True)
    order = sweep_order(pilot, all_pairs, distill_diag)
    print("      order: " + ", ".join(f"({a},{b})" for a, b in order),
          flush=True)

    sweep_path = os.path.join(CKPT_DIR, "sweep.json")
    sweep_ckpt = load_json(sweep_path, None)
    if sweep_ckpt is None:
        committed: list = []
        sweep_records: list = []
    else:
        committed = [tuple(p) for p in sweep_ckpt["committed"]]
        sweep_records = sweep_ckpt["records"]
        print(f"      resuming: {len(committed)} committed, "
              f"{len(sweep_records)} pairs already evaluated", flush=True)

    reset_pristine()
    if committed:
        replay_committed(model, get_student, committed)

    for (a, b) in order[len(sweep_records):]:
        def shift(i):
            return sum(1 for (x, y) in committed if y < i)
        pa, pb = a - shift(a), b - shift(b)
        assert pb == pa + 1, f"pair ({a},{b}) not adjacent after shifts"
        cur_layers = list(model.model.layers)
        st = get_student(a, b)
        set_stack(cur_layers[:pa] + [st] + cur_layers[pb + 1:])
        del st
        t = time.time()
        p_after = eval_current()
        ok = BD.should_commit(p_after, ppl_pristine, tau=args.tau)
        if ok:
            committed.append((a, b))
            reason = (f"COMMITTED: ppl {p_after:.2f} "
                      f"({delta(p_after):+.2f}% vs pristine) <= +{args.tau * 100:.0f}%")
        else:
            # ROLLBACK: restore pre-splice stack
            set_stack(cur_layers)
            reason = (f"rolled back: ppl {p_after:.2f} "
                      f"({delta(p_after):+.2f}% vs pristine) > +{args.tau * 100:.0f}%")
        sweep_records.append({
            "pair": [a, b], "committed": ok, "reason": reason,
            "ppl_measured": round(p_after, 4),
            "delta_vs_pristine_pct": round(delta(p_after), 3),
            "mse_final": distill_diag[(a, b)]["mse_final"],
            "depth_after_eval": len(model.model.layers),
        })
        save_json(sweep_path,
                  {"committed": [list(p) for p in committed],
                   "records": sweep_records})
        print(f"      pair ({a},{b}): {reason} [{time.time() - t:.1f}s]",
              flush=True)

    # Final perplexity re-measured on the finished stack (robust to resume).
    cur_ppl = eval_current()
    depth_after = len(model.model.layers)
    params_after = BD.count_parameters(model)
    reduction_pct = 100.0 * (params_before - params_after) / params_before

    def _safe(x):
        return round(x, 4) if math.isfinite(x) else None

    result = {
        "variation": "block-delta (distillation-flavored)",
        "model_id": args.model_id,
        "n_blocks_before": n_blocks,
        "depth_after": depth_after,
        "params_before": params_before,
        "params_after": params_after,
        "reduction_pct": round(reduction_pct, 2),
        "compression_ratio": round(params_after / params_before, 4),
        "perplexity_before": _safe(ppl_pristine),
        "perplexity_after": _safe(cur_ppl),
        "eval_tokens": n_tok,
        "tau_relative": args.tau,
        "committed_pairs": [list(p) for p in committed],
        "n_committed": len(committed),
        "n_rolled_back": len(sweep_records) - len(committed),
        "pilot": [{**r,
                   "ppl_spliced": _safe(r["ppl_spliced"]),
                   "ppl_drop_baseline": _safe(r["ppl_drop_baseline"])}
                  for r in pilot],
        "sweep": sweep_records,
        "distill_diag": {ckpt_key(a, b): distill_diag[(a, b)]
                         for (a, b) in all_pairs},
        "elapsed_s": round(time.time() - t0, 1),
        "notes": ("gate: commit iff ppl <= pristine_ppl * (1+tau), tau=0.05; "
                  "distillation caches captured in one pristine pass "
                  "(documented limitation); drop baseline removes the "
                  "second block of the pair with no distillation; "
                  "checkpoints under experiments/checkpoints_block_delta/."),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"results written to {args.out}")
    print(f"FINAL: depth {n_blocks} -> {depth_after}, "
          f"params {params_before:,} -> {params_after:,} "
          f"({reduction_pct:.2f}%), ppl {ppl_pristine:.2f} -> {cur_ppl:.2f}")
    print(f"TOTAL elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
