"""CrowS-Pairs run on qwen-1.8b — completing needed-experiments.md §2.

The archived `farhan-fixed-crows` run crashed in the judge before writing a
transition matrix; its Batched_Gen.csv has headers only. The judge now has retry
with backoff, so the original crash mode is handled.

This is the dataset where the method is most likely to fail (the opinion vector
already failed to transfer to CrowS in the archive), which is exactly what makes
it informative.

Sampling is stratified across the stereotype/anti-stereotype poles so both are
equally represented. n=300 gives 150 train / 150 test, comfortably above the
n>=100 §2 asks for and above the ~4% judge-noise floor measured in
docs/04-parity.md rung 1.

Known limit: this anonymized CSV carries no bias_type, so the per-category
breakdown §2 wants is not derivable here — it needs the full CrowS release.
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="crows qwen-1.8b",
    models=["qwen-1.8b"],
    dataset=DatasetSpec(
        name="crows_q",
        path="datasets/Crows_Pairs/crows_pairs_anonymized.csv",
        train_split=0.5,
    ),
    sample=SampleSpec(per_group=("stereo", 150), seed=0),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=14, neutral=15),
    method="mean_diff",
    max_tokens=128,
    batch_size=32,
)
