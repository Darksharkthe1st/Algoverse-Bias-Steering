"""Coherence check for the dose-matched act-add arms at layer 15 (§0.3).

act-add(+) raising harmless refusal 0.01 -> 0.59 cannot be faked by degeneration
(gibberish matches no refusal prefix), but the arm still has to be fluent to count.
Prints completions + the same degeneracy metrics used for the ablation arms.

    python actadd_coherence.py --coeff 2.009 --layer 15
"""

import argparse
import sys

sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")

from src.bias_steer import models, refusal, refusal_extract, steering  # noqa: E402
from src.bias_steer.config import DatasetSpec  # noqa: E402
from src.bias_steer.datasets import load_refusal_eval  # noqa: E402
from src.bias_steer.judge import is_refusal  # noqa: E402
from src.bias_steer.registry import MODELS  # noqa: E402

VECTOR = ("/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering/"
          "runs/20260816-230451_refusal-native_qwen-1.8b/steering_vector.safetensors")

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--coeff", type=float, default=2.009)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-tokens", type=int, default=96)
ap.add_argument("--show", type=int, default=6)
args = ap.parse_args()

loaded = models.load_model(MODELS["qwen-1.8b"])
tok = loaded.tokenizer
template = refusal_extract.REFUSAL_TEMPLATES["qwen-1.8b"].template
rd = refusal.load_native_direction(VECTOR, layer=args.layer, model_key="qwen-1.8b")

spec = DatasetSpec(name="refusal_eval")
spec.harm, spec.source_model = "harmless", "qwen-1_8b-chat"
prompts = [e.prompt for e in load_refusal_eval(spec)[:args.n]]


def distinct_ratio(text):
    ids = tok(text).input_ids
    return len(set(ids)) / len(ids) if ids else 0.0


def max_ngram_share(text, n=4):
    ids = tok(text).input_ids
    if len(ids) < n:
        return 0.0
    grams = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        grams[g] = grams.get(g, 0) + 1
    return max(grams.values()) * n / len(ids)


def degenerate(text):
    return max_ngram_share(text) > 0.5 or distinct_ratio(text) < 0.25


out = []
for i in range(0, len(prompts), 16):
    b = prompts[i:i + 16]
    hooks = steering.apply_actadd_single(loaded.model, rd.direction, coeff=args.coeff, layer=args.layer)
    out += models.generate_with_hooks(loaded, b, hooks, args.max_tokens, "", template=template)

print(f"\n### harmless + act-add(+{args.coeff}) at layer {args.layer} "
      f"(dose {args.coeff * float(rd.direction.norm()):.2f})\n")
print(f"refusal    {sum(is_refusal(c) for c in out) / len(out):.2f}")
print(f"degenerate {sum(degenerate(c) for c in out) / len(out):.2f}")
print(f"distinct   {sum(distinct_ratio(c) for c in out) / len(out):.2f}\n")
for p, c in list(zip(prompts, out))[:args.show]:
    print(f"PROMPT: {p}")
    print(f"  [{'refusal ' if is_refusal(c) else 'comply  '}] {c.strip()[:260]}\n")
