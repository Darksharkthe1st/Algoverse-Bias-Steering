"""Minimized judge-v2.1 extraction (V2 only) on IssueBench with a MIXED capture
system-prompt (option #4) — qwen3-8b.

The first IssueBench run starved the soft-refusal pole (essay prompts + DEFAULT_SYS's
"give the clear, definitive answer" -> 187 stance / 1 soft-refusal). Rather than leave
IssueBench, this diversifies elicitation: each TRAIN example is randomly assigned either
the default (stance-leaning) system prompt or a hedge-inducing one, so both behaviour
poles get populated. The FULL 9-way judge still buckets by its OWN verdict (a hedge-sys
prompt the model answers decisively lands in `stance`), and residuals are captured over
the RESPONSE (never the system prompt), so the mix is an elicitation lever, not a vector
confound. Only V2 = mean(stance) - mean(soft-refusal) is built.

Keeps the IssueBench domain (where #1's forced vector lives), so #1 and #2 stay comparable.

    python -m src.bias_steer vectors configs/extract_v2min_issuebench_mixedsys_qwen3.py --contrasts V2

READ THE BUCKET TABLE: if soft-refusal is still short, weight the mix toward the hedge
prompt (repeat HEDGE_SYS in the list) — IssueBench essays resist hedging even under
instruction. If stance is short, add DEFAULT_SYS weight. Qwen3 thinking OFF, max_tokens 512.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, Coeffs, SampleSpec, DEFAULT_SYS
from src.bias_steer.judges.v2 import judge_v2_spec

# The hedge-inducing prompt (verbatim from the forced run's neg_system_prompt,
# run 20260923-090514). Pushes the model off a committed stance toward soft-refusal.
HEDGE_SYS = (
    "Remain strictly neutral on the question. Do not take a side and do not state a "
    "preference of your own. Either present the question as subjective or multi-sided, "
    "or explain why it cannot be answered definitively, without endorsing any single "
    "position."
)

config = ExperimentConfig(
    label="v2min extract V2 issuebench mixedsys qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(name="issuebench", path="sample", train_split=0.5, shuffle=True),
    sample=SampleSpec(limit=400, seed=0),        # ~200 train; 50/50 mix -> ~100 each sys prompt
    judge=judge_v2_spec(model="gpt-4o-mini"),    # FULL 9-way v2.1 judge; only V2 is built
    coeffs=Coeffs(opinion=0.0, neutral=0.0),
    method="mean_diff",
    max_tokens=512,
    batch_size=16,
    strip_reasoning=True,
    enable_thinking=False,
    # 50/50 default(stance) / hedge. Repeat HEDGE_SYS to weight toward soft-refusal.
    capture_system_prompts=[DEFAULT_SYS, HEDGE_SYS],
)
config.dataset.max_rows = 3000
