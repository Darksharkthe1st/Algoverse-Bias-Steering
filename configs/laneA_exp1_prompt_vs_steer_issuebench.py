"""Lane A Exp-1 — prompt-vs-steer on IssueBench, the PIVOTAL comparison.

`docs/HANDOFF_lane_a.md` §Exp-1. The question: does a fitted opinion direction beat
simply *asking* the model to take a side, on prompts a system prompt does not
saturate? The toy forced-choice comparison prompts cannot answer it — the prompt arm
hit 100% in BOTH directions there (run `20260923-030712_prompt-baseline-opinion_qwen3-8b`),
a ceiling regime. IssueBench (Röttger et al., 2025; arXiv:2502.08395) is realistic
writing-assistance requests on contested issues, where the prompt arm has room to
lose. This config differs from `configs/prompt_baseline_issuebench.py` in four ways:

1. **The forced-contrast vector**, not the natural-contrast one. Natural contrast
   buckets residuals by the judge's verdict on default-prompt generations, so on
   IssueBench (whose prompts embed a stance: "X being a bad thing") it confounds
   prompt framing with the behaviour. Forced contrast generates each prompt twice,
   under the pos/neg system prompts, and buckets BY CONSTRUCTION — paired on the
   same prompt, so content cancels in the mean difference.
2. **`max_tokens=512`**, not 256. CLAUDE.md-level guardrail (HANDOFF §0.3): a
   truncated completion makes every metric compute on garbage.
3. **n=200 held-out items** from the 636k `sample` split, not the 150-prompt debug
   split — a bigger n on the same prompt distribution the vector was fitted on.
4. **The exclusion list** (see below), which the earlier config had no need for.

## What is excluded from the eval set, and why

`datasets/laneA/issuebench_exclusions.json`, built by
`scripts/build_heldout_exclusions.py`, which explains both rules. In short:

- **the vector's own 200 fit items** (`exclude_ids`) — the applied vector was fitted
  on the TRAIN split of `20260923-090514_...forced-contrast`, which drew from this
  same split, so without this the comparison would be scored partly on the fit set;
- **11 ShareGPT scraping-artifact templates** (`exclude`) — the handoff asks for the
  `"1 / 1…Share Prompt"` filter "before reporting"; doing it before *generating*
  also spends no GPU on prompts we would discard.

## Seeds, and why there is only one here

The handoff asks for ≥3 generation seeds. This pipeline generates GREEDILY
(`models.generate`: `do_sample=False`; `generate_with_hooks`: `temperature=0`), so a
generation seed is not a source of variance — three seeds would return byte-identical
completions and a fake three-fold CI. The honest variance estimate at k=1 is the
**item bootstrap**, which `metrics.beat_rate` already computes and which
`scripts/bootstrap_ci.py` recomputes from the judged CSV. `sample.seed` below picks
*which* items, not how they decode. Raised as a decision in
`runs/<id>/OPEN_QUESTIONS.md` rather than silently resolved.

## Judge

The pipeline's binary `neutrality` judge runs inline, because the 3-arm contrast and
`beat_rate` machinery is built on a 2-label judge (`_contrast` takes
`judge.labels[1]` as the positive pole — with v2.1's 9 labels that would silently
become "incoherent"). The **reportable** numbers come from re-judging the persisted
completions under pinned judge v2.1 offline (`scripts/rejudge_v21.py`), per
CLAUDE.md §4: one judge version per table, and never mixed.

    python -m src.bias_steer run configs/laneA_exp1_prompt_vs_steer_issuebench.py

Needs OPENAI_API_KEY (inline judge) and ~1 A100 for Qwen3-8B.
"""

import json

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec,
)
from src.utils import get_repo_root

# get_repo_root(), not a __file__-relative path: the same resolution the dataset
# loaders use, so the config still finds its data when it is imported from
# somewhere other than configs/ (a pilot copy, a scratch dir).
_REPO = get_repo_root()

# The exclusion artifact is DATA, not a constant restated here: one owner
# (scripts/build_heldout_exclusions.py writes it, this reads it), and it lands in
# the run manifest via SampleSpec, so the run records exactly what it held out.
_EXCLUSIONS = json.loads(
    (_REPO / "datasets" / "laneA" / "issuebench_exclusions.json").read_text()
)

# The forced-contrast opinion vector. Shape is asserted against (n_layers, d_model)
# by `_load_provided_vector` -> `steering.assert_steering_shape` before a single
# token is generated (CLAUDE.md §6 — a 1-D tensor broadcasts a scalar DC offset,
# the retracted 2025 bug). Verified here: (36, 4096) fp16, md5 dab6cecca362db8378ebd31b20f34ae0.
VECTOR = ("runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b"
          "/steering_vector.safetensors")

config = ExperimentConfig(
    label="laneA_exp1_prompt_vs_steer_issuebench",
    models=["qwen3-8b"],                          # frozen submission model
    dataset=DatasetSpec(
        name="issuebench",
        path="sample",                            # the 636k split the vector was fitted on
        # With a supplied vector nothing is extracted, so TRAIN and TEST are both
        # evaluated (`_run_one`: eval_examples = train + test). train_split only has
        # to be a legal fraction; every one of the `limit` items below is scored.
        train_split=0.5,
    ),
    judge=JudgeSpec(name="neutrality"),           # binary, inline; v2.1 re-judge is offline
    coeffs=Coeffs(opinion=8.0, neutral=8.0),      # headline dose; Exp-3 sets the reportable one
    sample=SampleSpec(
        limit=200,
        seed=0,
        exclude_ids=tuple(_EXCLUSIONS["heldout"]["exclude_ids"]),
        exclude={"template_id": _EXCLUSIONS["artifacts"]["exclude_template_ids"]},
    ),
    max_tokens=512,
    batch_size=32,
)
config.enable_thinking = False   # qwen3 defaults to thinking; off = answer directly
config.intervention = "both"     # INITIAL + STEERED_POS/NEG + PROMPT_POS/NEG
config.vector_path = VECTOR
