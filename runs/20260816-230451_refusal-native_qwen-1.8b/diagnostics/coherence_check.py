"""Coherence spot-check for the Phase-2 ablation arm (docs/needed-experiments.md §0.3).

`experiment_refusal` records verdicts, not text, so a "refusal collapsed to 0.00"
result cannot be distinguished from model degeneration by the run artifacts alone.
This prints the actual completions for a handful of harmful prompts under
(a) no intervention and (b) ablation of OUR native direction at a chosen layer.

    python coherence_check.py --layer 15 --n 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")

from src.bias_steer import models, refusal, refusal_extract, steering  # noqa: E402
from src.bias_steer.datasets import load_refusal_eval  # noqa: E402
from src.bias_steer.config import DatasetSpec  # noqa: E402
from src.bias_steer.registry import MODELS  # noqa: E402

VECTOR = ("/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering/"
          "runs/20260816-230451_refusal-native_qwen-1.8b/steering_vector.safetensors")

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=96)
args = ap.parse_args()

loaded = models.load_model(MODELS["qwen-1.8b"])
rd = refusal.load_native_direction(VECTOR, layer=args.layer, model_key="qwen-1.8b")
template = refusal_extract.REFUSAL_TEMPLATES["qwen-1.8b"].template

spec = DatasetSpec(name="refusal_eval")
spec.harm, spec.source_model = "harmful", "qwen-1_8b-chat"
prompts = [e.prompt for e in load_refusal_eval(spec)[:args.n]]

base = models.generate(loaded, prompts, args.max_tokens, "", template=template)
hooks = steering.apply_directional_ablation(loaded.model, rd.direction)
abl = models.generate_with_hooks(loaded, prompts, hooks, args.max_tokens, "", template=template)

print(f"\n### ablation of OUR native direction, layer {args.layer}, |r|={rd.direction.norm():.3f}\n")
for p, b, a in zip(prompts, base, abl):
    print(f"PROMPT: {p}")
    print(f"  [baseline ] {b.strip()[:300]}")
    print(f"  [ablation ] {a.strip()[:300]}")
    print()
