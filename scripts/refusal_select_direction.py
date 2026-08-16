#!/usr/bin/env python3
"""Chunk E — select_direction: sweep candidate (pos, layer) directions and pick the
single best, reproducing Arditi et al.'s selection. Validates by landing on the
model's published (layer, pos).

THIS SCRIPT LOADS A MODEL. Run it on the Lambda GPU box, not in the CPU/planning
env (see docs/06-refusal-generation.md). Heavy: ~n_pos x n_layers scored sweeps
over the val sets (ablation on harmful_val, actadd on harmless_val, KL on harmless_val).

Faithful to pipeline/submodules/select_direction.py @9d852fa:
  - candidates := the paper's mean_diffs.pt grid by default (isolates SELECTION from
    extraction fidelity); pass --use-extracted to select over our own grid instead.
  - per (pos, layer): ablation_refusal = mean refusal_score(harmful_val | ablate dir
    everywhere); steering_refusal = mean refusal_score(harmless_val | actadd raw dir
    at that layer, coeff=1.0); kl = mean KL(baseline_harmless || ablated_harmless).
  - filter_fn: drop NaN; drop layer >= int(n_layer*(1-0.2)); drop kl > 0.1;
    drop steering < 0.0. Then pick the MINIMUM ablation refusal score.

NOTE ON THE METRIC: selection uses the paper's LOGIT-based refusal_score
(refusal_extract.get_refusal_scores), NOT the substring judge (judge.is_refusal).
The substring judge is a downstream jailbreak-EVAL metric (a later chunk), not part
of selection. See docs/06-refusal-generation.md.
"""

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _last_logits(loaded, instructions, fwd_hooks=(), batch_size=32):
    """Last-token logits (float64, on CPU) under optional TransformerLens hooks."""
    import torch
    from src.bias_steer.refusal_extract import format_refusal_prompt

    model, tok = loaded.model, loaded.tokenizer
    prev = getattr(tok, "padding_side", None)
    tok.padding_side = "left"
    chunks = []
    try:
        for i in range(0, len(instructions), batch_size):
            strs = [format_refusal_prompt(loaded.spec.name, s) for s in instructions[i:i + batch_size]]
            enc = tok(strs, return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(model.cfg.device)
            attn = enc["attention_mask"].to(model.cfg.device)
            with torch.no_grad():
                ctx = model.hooks(list(fwd_hooks)) if fwd_hooks else _nullctx()
                with ctx:
                    logits = model(ids, attention_mask=attn)
            chunks.append(logits[:, -1, :].to(torch.float64).cpu())
    finally:
        if prev is not None:
            tok.padding_side = prev
    return __import__("torch").cat(chunks, dim=0)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _refusal_from_logits(logits, toks):
    import torch
    probs = logits.softmax(dim=-1)
    refusal_p = probs[:, toks].sum(dim=-1)
    return (torch.log(refusal_p + 1e-8) - torch.log(1.0 - refusal_p + 1e-8))


def _kl(baseline_logits, intervention_logits, epsilon=1e-6):
    """Mean KL(baseline || intervention) at the last position (float64)."""
    import torch
    pa = baseline_logits.softmax(dim=-1)
    pb = intervention_logits.softmax(dim=-1)
    kl = (pa * (torch.log(pa + epsilon) - torch.log(pb + epsilon))).sum(dim=-1)
    return kl.mean().item()


def _filter_out(refusal_score, steering_score, kl_div_score, layer, n_layer,
                kl_threshold=0.1, induce_refusal_threshold=0.0, prune_layer_percentage=0.2):
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-extracted", action="store_true",
                    help="select over our freshly-extracted grid instead of their mean_diffs.pt")
    args = ap.parse_args(argv)

    import torch
    from src.bias_steer.registry import MODELS
    from src.bias_steer import models, refusal, steering
    from src.bias_steer import refusal_extract as rx

    model_key = rx._resolve_model_key(args.model)
    print(f"loading {model_key} ...")
    loaded = models.load_model(MODELS[model_key])
    toks = rx.refusal_toks(model_key)

    sampled = rx.load_and_sample_repro(n_val=args.n_val)
    harmful_val, harmless_val = sampled["harmful_val"], sampled["harmless_val"]

    if args.use_extracted:
        n_pos = rx.n_positions(loaded)
        candidates, _ = rx.run_extraction(
            model_key, sampled["harmful_train"], sampled["harmless_train"],
            n_pos=n_pos, batch_size=args.batch_size)
        candidates = candidates.to(torch.float64)
    else:
        candidates = refusal.artifacts.load_pt_tensor(
            refusal.artifact_dir(refusal.MODEL_TO_RUN_DIR[model_key])
            / "generate_directions" / "mean_diffs.pt").to(torch.float64)
    n_pos, n_layer, d_model = candidates.shape
    print(f"candidates grid {tuple(candidates.shape)}  (source: "
          f"{'extracted' if args.use_extracted else 'their mean_diffs.pt'})")

    baseline_harmless = _last_logits(loaded, harmless_val, (), args.batch_size)

    rows = []
    for pi in range(n_pos):
        source_pos = pi - n_pos
        for layer in range(n_layer):
            direction = candidates[pi, layer].to(loaded.model.cfg.dtype)
            abl = steering.apply_directional_ablation(loaded.model, direction)
            add = steering.apply_actadd_single(loaded.model, direction, coeff=1.0, layer=layer)

            abl_harmful = _last_logits(loaded, harmful_val, abl, args.batch_size)
            refusal_score = _refusal_from_logits(abl_harmful, toks).mean().item()

            add_harmless = _last_logits(loaded, harmless_val, add, args.batch_size)
            steering_score = _refusal_from_logits(add_harmless, toks).mean().item()

            abl_harmless = _last_logits(loaded, harmless_val, abl, args.batch_size)
            kl = _kl(baseline_harmless, abl_harmless)

            rows.append({"pos": source_pos, "layer": layer, "refusal": refusal_score,
                         "steering": steering_score, "kl": kl})
        print(f"  pos {source_pos:+d} done")

    kept = [r for r in rows
            if not _filter_out(r["refusal"], r["steering"], r["kl"], r["layer"], n_layer)]
    assert kept, "all candidates filtered out"
    best = sorted(kept, key=lambda r: r["refusal"])[0]  # minimum ablation refusal

    meta = refusal.load_refusal_direction(model_key)
    ok = (best["pos"], best["layer"]) == (meta.pos, meta.layer)
    print(f"\nselected: position={best['pos']}, layer={best['layer']}  "
          f"(refusal={best['refusal']:.4f}, steering={best['steering']:.4f}, kl={best['kl']:.4f})")
    print(f"published: position={meta.pos}, layer={meta.layer}")
    print("MATCH" if ok else "MISMATCH — investigate (metric ties, filtering, dtype)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
