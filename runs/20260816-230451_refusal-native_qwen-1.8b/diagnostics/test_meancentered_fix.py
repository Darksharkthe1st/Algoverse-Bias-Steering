"""Does mean-centering rescue our native direction for ablation?

why_ablation_breaks.py showed our rows carry a large component along the grand-mean
residual (|cos| up to 0.92), and that ablation collapses the model exactly where that
overlap is large. The implied fix: project the grand-mean direction OUT of the row
before using it as an ablation direction.

    r_corrected = r - (r . m_hat) m_hat        m = grand-mean residual at that layer

If the hypothesis is right, the corrected L12/L15 directions should drop harmful
refusal WITHOUT the token-loop collapse.

    python test_meancentered_fix.py --layers 12 15 19
"""

import argparse
import sys

sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from src.bias_steer import artifacts, models, refusal_extract, steering  # noqa: E402
from src.bias_steer.config import DatasetSpec  # noqa: E402
from src.bias_steer.datasets import load_refusal_eval  # noqa: E402
from src.bias_steer.judge import is_refusal  # noqa: E402
from src.bias_steer.registry import MODELS  # noqa: E402

RUN = ("/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering/"
       "runs/20260816-230451_refusal-native_qwen-1.8b")

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, nargs="+", default=[12, 15, 19])
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--batch-size", type=int, default=16)
args = ap.parse_args()

ours = artifacts.load_vector(f"{RUN}/steering_vector.safetensors").float()
resids = load_file(f"{RUN}/residuals.safetensors")
stacked = torch.cat([v.float().reshape(-1, *v.shape[-2:]) for v in resids.values()], dim=0)
mean_resid = stacked.mean(dim=0)

loaded = models.load_model(MODELS["qwen-1.8b"])
tok = loaded.tokenizer
template = refusal_extract.REFUSAL_TEMPLATES["qwen-1.8b"].template

spec = DatasetSpec(name="refusal_eval")
spec.harm, spec.source_model = "harmful", "qwen-1_8b-chat"
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


def mean_centered(layer):
    r, m = ours[layer], mean_resid[layer]
    m_hat = m / m.norm()
    return r - (r @ m_hat) * m_hat


for layer in args.layers:
    d = mean_centered(layer).contiguous()
    raw = ours[layer]
    kept = float(d.norm() / raw.norm())
    out = []
    for i in range(0, len(prompts), args.batch_size):
        b = prompts[i:i + args.batch_size]
        hooks = steering.apply_directional_ablation(loaded.model, d)
        out += models.generate_with_hooks(loaded, b, hooks, args.max_tokens, "", template=template)
    print(f"ablation(mean-centered ours, L{layer})  refusal "
          f"{sum(is_refusal(c) for c in out) / len(out):.2f}  "
          f"degenerate {sum(degenerate(c) for c in out) / len(out):.2f}  "
          f"distinct {sum(distinct_ratio(c) for c in out) / len(out):.2f}  "
          f"(kept {kept:.2f} of ||r||)", flush=True)
    print(f"    sample: {out[0].strip()[:220]}\n", flush=True)
