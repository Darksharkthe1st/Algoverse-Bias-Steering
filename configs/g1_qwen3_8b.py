"""G1 — the model-internal positive control on the frozen submission model.

Contract §12 **A6**; criteria and their justification in `docs/PREREG.md` §7a.

Run on the Lambda box (NO OpenAI key — the refusal judge is deterministic):

    python scripts/fetch_refusal_artifacts.py          # 6 model-independent splits
    python -m src.bias_steer refuse configs/g1_qwen3_8b.py

Why this config and not `configs/refusal_repro.py`: that one applies Arditi's
PUBLISHED direction, loaded from `third_party/refusal_direction/pipeline/runs/
<model>/direction.pt`. He ships that for five models and `Qwen/Qwen3-8B` is not
one of them. G1 here is computed from this model's own activations — see A6 for
why we changed the gate rather than the model.

What this config covers, and what it does not:

* **G1b / G1c** — this run. The five arms (harmful and harmless × baseline,
  ablation, act-add) give ΔP_refuse on held-out `harmful_test` plus the regime
  checks, and the control arms give specificity.
* **G1a** — `src/bias_steer/g1_stability.py`, over the residuals cached during
  extraction. It needs no second forward pass, so run it from the same cached
  activations rather than reloading the model.

The model is pinned to an immutable revision in `MODEL_CATALOG`, not here, so
every config targeting `qwen3-8b` gets the same weights and the manifest records
what was actually loaded (`model_spec.revision`, PREREG §3b).
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs

config = ExperimentConfig(
    label="g1 qwen3-8b",
    models=["qwen3-8b"],                        # Qwen/Qwen3-8B @ b968826d9c46
    dataset=DatasetSpec(name="refusal_eval"),   # the 6 model-independent splits
    judge=JudgeSpec(name="refusal_substring", labels=["compliance", "refusal"]),
    coeffs=Coeffs(opinion=1.0, neutral=1.0),    # act-add dose; neutral unused here
    method="ablation",                          # λ=1 directional ablation
    system_prompt="",
    max_tokens=128,
    batch_size=16,                              # 8B in fp16; drop to 8 if OOM
)
