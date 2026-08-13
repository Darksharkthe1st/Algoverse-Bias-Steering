"""Refusal-direction repro config (Arditi et al., 2024; arXiv:2406.11717).

Applies the paper's PUBLISHED refusal direction (loaded from
third_party/refusal_direction/, fetched by scripts/fetch_refusal_artifacts.py)
and scores refusal by substring match. No steering vector is trained.

Run it with (NO OpenAI key needed — refusal scoring is deterministic):

    python -m src.bias_steer refuse configs/refusal_repro.py

Requires (on the GPU box): torch + transformer_lens, `HF_TOKEN` for the model
download, and the fetched artifacts. Model execution is intended for the Lambda
GPU box; the flow + metrics are validated under a fake backend without a model.

NOTE (Chunk 5 fidelity): the paper formats prompts with the model's chat template
and NO system turn. `system_prompt=""` here still emits an empty system turn via
models.render_prompts; exact no-system-turn fidelity is a follow-up for the real
run, mirroring the extraction-side finding on the generate track.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs

config = ExperimentConfig(
    label="refusal repro",
    models=["qwen-1.8b"],                       # smallest; extend to yi-6b / llama-2-7b (chat_template=True)
    dataset=DatasetSpec(name="refusal_eval"),   # prompts read per-model from the committed completions
    judge=JudgeSpec(name="refusal_substring", labels=["compliance", "refusal"]),
    coeffs=Coeffs(opinion=1.0, neutral=1.0),    # opinion = act-add dose magnitude; neutral unused
    method="ablation",                          # recorded in the manifest; the flow runs all 5 arms
    system_prompt="",
    max_tokens=128,
    batch_size=16,
)
