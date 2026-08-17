#!/usr/bin/env python3
"""Overnight validation. Read-only. Expands no scope.

Checks that (a) every committed run is what it claims, (b) the shape guard would
actually reject the 2025 archive it exists to protect against, and (c) the
ablation operator behaves on CPU.
"""
import glob, json, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

report = {"runs": [], "archive_vectors": [], "operator": {}, "verdict": {}}

# --- (a) run provenance -----------------------------------------------------
from safetensors.torch import load_file
for d in sorted(glob.glob("runs/*/")):
    rec = {"run": os.path.basename(d.rstrip("/"))}
    for n in ("results.csv", "summary.md", "manifest.json", "steering_vector.safetensors"):
        rec[n] = os.path.isfile(d + n) and os.path.getsize(d + n) > 0
    if rec.get("steering_vector.safetensors"):
        try:
            v = load_file(d + "steering_vector.safetensors")["vector"]
            rec["shape"] = list(v.shape); rec["ndim"] = v.ndim
        except Exception as e:
            rec["shape_error"] = str(e)
    if rec.get("results.csv"):
        rec["rows"] = sum(1 for _ in open(d + "results.csv")) - 1
    rec["complete"] = all(rec.get(n) for n in
        ("results.csv", "summary.md", "manifest.json", "steering_vector.safetensors"))
    report["runs"].append(rec)

# --- (b) does the guard actually catch the 2025 archive? --------------------
import torch
from src.bias_steer import steering
class _Cfg: n_layers, d_model = 24, 2048
class _M: cfg = _Cfg()
caught = missed = 0
cands = sorted(glob.glob("experiments/past_logs/past_vecs/official_refusal_vecs/*.pt"))
cands += sorted(glob.glob("experiments/best_vecs/*steer_vec.pkl"))[:40]
for p in cands:
    try:
        obj = (torch.load(p, map_location="cpu", weights_only=False)
               if p.endswith(".pt") else pickle.load(open(p, "rb")))
    except Exception:
        continue
    t = obj if torch.is_tensor(obj) else (next(iter(obj.values()), None) if isinstance(obj, dict) else None)
    if not torch.is_tensor(t):
        continue
    entry = {"path": p, "shape": list(t.shape), "ndim": t.ndim}
    try:
        steering.apply_resid_pre_add(_M(), t, coeff=1.0)
        entry["guard"] = "ACCEPTED"
        if t.ndim == 1:
            missed += 1; entry["guard"] = "MISSED-1D"
    except steering.SteeringShapeError:
        entry["guard"] = "REJECTED"; caught += 1
    report["archive_vectors"].append(entry)

# --- (c) operator smoke on CPU ---------------------------------------------
try:
    v = torch.ones(24, 2048)
    hooks = steering.apply_resid_pre_add(_M(), v, coeff=2.0)
    x = torch.zeros(1, 3, 2048)
    out = hooks[0][1](x, hook=None)
    report["operator"] = {"hooks": len(hooks),
                          "per_layer_delta": float(out[0, 0, 0]),
                          "expected": 2.0 / 24,
                          "ok": abs(float(out[0, 0, 0]) - 2.0 / 24) < 1e-6}
except Exception as e:
    report["operator"] = {"error": str(e)}

report["verdict"] = {
    "runs_total": len(report["runs"]),
    "runs_complete": sum(1 for r in report["runs"] if r["complete"]),
    "runs_all_2d": all(r.get("ndim", 2) == 2 for r in report["runs"] if "ndim" in r),
    "archive_checked": len(report["archive_vectors"]),
    "archive_1d_rejected_by_guard": caught,
    "archive_1d_MISSED_by_guard": missed,
}
os.makedirs("runs/_validation", exist_ok=True)
with open("runs/_validation/overnight.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report["verdict"], indent=2))
