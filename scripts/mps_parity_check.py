#!/usr/bin/env python3
"""Gate: prove MPS numerics against CPU before any Apple-Silicon run counts.

TransformerLens warns that the MPS backend can produce silently incorrect
results (TransformerLens issue #1178: an IOI logit difference INVERTS on MPS
under some PyTorch versions, with no error raised). This project has already
lost a year to one silent numerical bug, so no run captured on MPS is evidence
until this gate passes on the same machine, torch version, and model family.

    TRANSFORMERLENS_ALLOW_MPS=1 python3 -m scripts.mps_parity_check \
        --model qwen-1.8b --n 12 --out runs/_mps_parity_qwen-1.8b.json

What it does, using the SAME capture path as the runs
(`bbq_score.capture_prompt_residuals`):

  1. capture residuals for n real BBQ prompts on CPU float32 (reference)
  2. capture the same prompts on MPS float32   -> backend error, isolated
  3. capture the same prompts on MPS float16   -> backend + dtype, what runs use
  4. compare per-layer cosine of matched residual vectors, and the cosine of a
     small split contrast direction built from each capture

PASS iff the MPS fp32 vs CPU fp32 direction cosine is >= 0.999 at the median
layer and no matched residual pair falls below 0.995. fp16-vs-fp32 numbers are
reported for context (dtype noise is expected and acceptable; a backend
inversion is not).

Exit 0 = safe to run tonight. Anything else = capture on this machine is not
evidence; fall back to CPU capture or a CUDA box.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def capture(model_key, device, dtype_name, prompts, system_prompt):
    import torch
    from src.bias_steer import bbq_score as bs
    from src.bias_steer import models as M
    from src.bias_steer.registry import MODELS
    from transformer_lens import HookedTransformer

    spec = MODELS[model_key]
    dtype = {"float32": torch.float32, "float16": torch.float16}[dtype_name]
    extra = {"revision": spec.revision} if spec.revision else {}
    model = HookedTransformer.from_pretrained_no_processing(
        spec.hf_id, device=device, dtype=dtype,
        default_padding_side="left", **extra)
    model.eval()
    loaded = M.LoadedModel(model=model, tokenizer=model.tokenizer,
                           spec=spec, device=device)
    out = bs.capture_prompt_residuals(loaded, prompts, system_prompt,
                                      batch_size=4)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return out


def per_layer_cos(a, b):
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def summarize(ref, test):
    """Residual-level and direction-level agreement between two captures."""
    # residual level: cosine per (item, layer)
    c = per_layer_cos(ref, test)                     # (n, L)
    # direction level: contrast first half vs second half of items, per capture
    h = ref.shape[0] // 2
    d_ref = ref[:h].mean(axis=0) - ref[h:].mean(axis=0)
    d_test = test[:h].mean(axis=0) - test[h:].mean(axis=0)
    dc = per_layer_cos(d_ref, d_test)                # (L,)
    fin = c[np.isfinite(c)]
    dfin = dc[np.isfinite(dc)]
    return {
        "resid_cos_min": float(fin.min()),
        "resid_cos_median": float(np.median(fin)),
        "direction_cos_median": float(np.median(dfin)),
        "direction_cos_min": float(dfin.min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
    import torch
    if not torch.backends.mps.is_available():
        print("MPS not available; nothing to gate.")
        return 1

    from src.bias_steer import bbq_score as bs
    from src.bias_steer.config import DEFAULT_SYS

    items = bs.load_scoreable("Religion", "ambig", args.n, 0)
    prompts = [bs.bare_prompt(e) for e, _ in items]
    print(f"{len(prompts)} prompts, model {args.model}")

    print("capture 1/3: CPU float32 (reference; slow) ...", flush=True)
    cpu32 = capture(args.model, "cpu", "float32", prompts, DEFAULT_SYS)
    print("capture 2/3: MPS float32 ...", flush=True)
    mps32 = capture(args.model, "mps", "float32", prompts, DEFAULT_SYS)
    print("capture 3/3: MPS float16 (what the runs use) ...", flush=True)
    mps16 = capture(args.model, "mps", "float16", prompts, DEFAULT_SYS)

    backend = summarize(cpu32, mps32)
    dtype = summarize(mps32, mps16)
    end_to_end = summarize(cpu32, mps16)

    ok = (backend["direction_cos_median"] >= 0.999
          and backend["resid_cos_min"] >= 0.995)

    report = {
        "model": args.model, "n_prompts": len(prompts),
        "torch": torch.__version__,
        "backend_mps32_vs_cpu32": backend,
        "dtype_mps16_vs_mps32": dtype,
        "end_to_end_mps16_vs_cpu32": end_to_end,
        "pass": bool(ok),
        "criterion": "backend direction_cos_median >= 0.999 and resid_cos_min >= 0.995",
    }
    print(json.dumps(report, indent=2))
    out = Path(args.out or f"runs/_mps_parity_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}\n{'PASS - MPS capture counts as evidence on this stack'
          if ok else 'FAIL - do NOT use MPS captures as evidence'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
