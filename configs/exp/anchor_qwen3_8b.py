"""Fit a real opinion vector on qwen3-8b — the frozen submission model.

The two generalization-test configs (`configs/apply_opinion_axbench.py`,
`configs/apply_opinion_issuebench.py`) both pin `models=["qwen3-8b"]` and need a
saved `steering_vector.safetensors` with shape (36, 4096) to apply — the shape
guard rejects anything fit on another model (see `docs/PREREG.md` §3b / contract
§12 A4 for why qwen3-8b is the pinned frozen submission model). No such vector
existed anywhere in the repo or its history, so this config extracts one the
normal way: TRAIN split -> generate + judge + bucket -> mean_diff direction ->
TEST split eval.

Same battery (`snapshot`/log_103_comparison_200.json) and method as the other
`anchor_*` configs, sibling to `configs/exp/anchor_qwen7b.py`. Coeffs match the
dose already fixed in the two apply configs (opinion=8.0, neutral=8.0) rather
than the qwen-7b notebook coeffs, so the fitted vector is validated at the same
dose it will actually be applied at downstream.

RE-EXTRACTION (2026-09-03): the first vector built from this config
(runs/20260901-092009_anchor-qwen3-8b_qwen3-8b, on branch fk/qwen3-8b-opinion-vector)
predates `config.enable_thinking` and ran thinking-ON at max_tokens=128 -- the
same silent <think>-truncation bug fixed for the §14 prompt-control baseline
(docs/HANDOFF_prompt_control.md §0.3) may have contaminated its TRAIN-phase
residuals/verdicts. Applying that vector against a clean (enable_thinking=False)
prompt baseline showed the vector losing to prompting outright (steer-only=0 on
both directions) -- but a prior thinking-mode run of the same experiment instead
found complementarity (prompt and steer matched on aggregate but won on different
items), which is the result this project actually expects to replicate. Adding
`enable_thinking = False` here isolates whether that complementarity was real
model behavior or an artifact of truncation-driven per-item noise. Re-run this
config and replace the vector before drawing any conclusion from the head-to-head.

    python -m src.bias_steer run configs/exp/anchor_qwen3_8b.py
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="anchor qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(name="snapshot", path="datasets/Snapshots/log_103_comparison_200.json", train_split=0.5, shuffle=False),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    method="mean_diff",
    max_tokens=128,
    batch_size=16,  # 8B in fp16; drop to 8 if OOM (matches g1_qwen3_8b.py note)
)
config.enable_thinking = False  # see RE-EXTRACTION note above
