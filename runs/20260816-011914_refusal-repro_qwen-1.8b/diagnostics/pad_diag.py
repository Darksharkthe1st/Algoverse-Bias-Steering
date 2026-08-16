"""Is the harmful/baseline gap caused by batched left-padding?

Generate the SAME 40 harmful prompts at batch_size=16 and batch_size=1 and
compare refusal rates. Paper: 0.70. Batched run measured 0.38.
"""
import sys; sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")
from src.bias_steer import models, judge
from src.bias_steer.registry import MODELS
from src.bias_steer.experiment_refusal import _default_load_eval

TMPL = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
N = 40

loaded = models.load_model(MODELS["qwen-1.8b"])
ex = _default_load_eval("qwen-1_8b-chat", "harmful")[:N]
prompts = [e.prompt for e in ex]

def rate(bs):
    out = []
    for i in range(0, len(prompts), bs):
        out += models.generate(loaded, prompts[i:i+bs], 128, "", template=TMPL)
    r = sum(judge.is_refusal(t) for t in out) / len(out)
    return r, out

r16, o16 = rate(16)
r1,  o1  = rate(1)
print(f"\nn={N}  batch=16 refusal {r16:.3f}   batch=1 refusal {r1:.3f}   (paper 0.700)")
flips = [(p, a, b) for p, a, b in zip(prompts, o16, o1) if judge.is_refusal(a) != judge.is_refusal(b)]
print(f"flips between batch sizes: {len(flips)}/{N}")
for p, a, b in flips[:3]:
    print(f"\n--- {p[:70]}")
    print(f"  bs16 [{'REFUSE' if judge.is_refusal(a) else 'COMPLY'}]: {a[:100]!r}")
    print(f"  bs1  [{'REFUSE' if judge.is_refusal(b) else 'COMPLY'}]: {b[:100]!r}")
