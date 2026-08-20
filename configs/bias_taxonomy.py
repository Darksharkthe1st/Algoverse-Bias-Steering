"""Bias-taxonomy run configs — one direction per BBQ category (JZ-1/JZ-2).

Standard single-category run (Religion by default):

    python -m src.bias_steer run configs/bias_taxonomy.py

All categories, which is what the experiment actually needs:

    python scripts/run_bias_taxonomy.py --model qwen-1.8b

Three things here differ from `configs/example_bbq.py` and each is deliberate.

**`judge="bbq_choice"`, not `"neutrality"`.** BBQ is multiple choice and on an
ambiguous item the correct answer is the unknown option, so the label is a string
match rather than a rubric call. No `OPENAI_API_KEY`, no API latency, and no
judge version attached to any number that comes out (`AGENTS.md` §4 retires
judge v1 and leaves v2 unfrozen — this sidesteps that entirely).

**`labels=["unknown", "biased"]`.** `experiment._contrast` reads
`(labels[1], labels[0])` as (positive, negative), so this makes the direction
point from correct abstention toward stereotyped answering:

    direction = mean(resid | biased) - mean(resid | unknown)

Getting the order backwards silently flips the sign of every direction, which
would not error and would not be visible in a cosine magnitude.

**`filter={"context_condition": ["ambig"]}`.** Ambiguous items only. On a
disambiguated item the context resolves who did it, so a confident answer is
CORRECT and naming a group is not evidence of bias. Mixing them in would
contaminate the biased pole with ordinary right answers.
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

BBQ_DIR = "datasets/BBQ_Prompt_Sets"

CATEGORIES = [
    "Religion", "Race_ethnicity", "Gender_identity", "Age", "Nationality",
    "Physical_appearance", "Disability_status", "Sexual_orientation",
    "Race_x_gender", "Race_x_SES",
]

#: Per-model steering coefficients. Only used by the eval half of `run`; the
#: direction itself does not depend on them. qwen-1.8b values are the notebook's
#: (configs/example_bbq.py); others are placeholders until a sweep says otherwise
#: — per-model coefficients never stabilised, which the norm profiles explain
#: (docs/VERIFICATION_2026-08-07.md).
COEFFS = {
    "qwen-1.8b": Coeffs(opinion=14, neutral=15),
    "qwen-7b": Coeffs(opinion=32, neutral=32),
    "qwen-14b": Coeffs(opinion=40, neutral=40),
    "yi-6b": Coeffs(opinion=32, neutral=32),
    "gemma-2b": Coeffs(opinion=5, neutral=5),
    "gemma-7b": Coeffs(opinion=28, neutral=28),
}


def make_config(category: str, model: str = "qwen-1.8b", *,
                limit: int | None = 400, seed: int = 0,
                train_split: float = 1.0, max_tokens: int = 24,
                batch_size: int = 16) -> ExperimentConfig:
    """One category -> one ExperimentConfig.

    `train_split=1.0` by default: for this experiment every item is used to build
    the direction, because the deliverable is the direction itself rather than a
    steered-generation table. Set it below 1.0 to hold out an eval split.

    `max_tokens=24` because we only need to see which option the model names; a
    longer generation costs time and adds nothing the parser reads.
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown BBQ category {category!r}; known: {CATEGORIES}")
    return ExperimentConfig(
        label=f"bias-taxonomy {category.lower()}",
        models=[model],
        dataset=DatasetSpec(
            name="bbq",
            path=f"{BBQ_DIR}/{category}.jsonl",
            train_split=train_split,
        ),
        sample=SampleSpec(
            filter={"context_condition": ["ambig"]},
            limit=limit,
            seed=seed,
        ),
        judge=JudgeSpec(name="bbq_choice", labels=["unknown", "biased"]),
        coeffs=COEFFS.get(model, Coeffs(opinion=16, neutral=16)),
        method="mean_diff",
        max_tokens=max_tokens,
        batch_size=batch_size,
    )


#: Default for the plain CLI path. Religion is the smallest category that
#: resolves at 100% coverage, so it is the fastest honest smoke test.
config = make_config("Religion")
