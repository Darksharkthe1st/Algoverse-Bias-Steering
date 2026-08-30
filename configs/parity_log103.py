"""Reproduce archived run Log_103 — the parity anchor (docs/04-parity.md).

    python -m src.bias_steer run configs/parity_log103.py

Every field is set to match the archived run rather than to be a good default, so
the resulting vector and transition matrix can be compared against
`experiments/best_vecs/log_103_Qwen1.5-1.8B-Chat_steer_vec.pkl` and the archived
`pre-steering_responses.txt`. Do not "improve" anything here — a change to any
field silently invalidates the comparison. Copy it for real experiments instead.

Two fields exist specifically to make this reproducible and are not what you want
for new work:

- `shuffle=False` — the notebook took a plain `prompts[:int(len*train_split)]`
  slice with no shuffle. Verified to reproduce the archived 100-prompt train set
  exactly, in order. New experiments should leave shuffle on, so the split can't
  correlate with file order.
- `dataset.name="snapshot"` — the 200 prompts exist only inside the archived
  run's pickle; they are NOT `datasets/GPT_Prompts/comparison_questions_200.csv`
  (296 different prompts). Lifted to JSON by `tools/snapshot_from_pickle.py`.
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="parity log103",
    models=["qwen-1.8b"],                    # archived: Qwen/Qwen1.5-1.8B-Chat
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5,                     # archived: 100 train / 100 test
        shuffle=False,                       # archived: unshuffled head-slice
    ),
    sample=SampleSpec(),                     # no sampling — all 200, in order
    judge=JudgeSpec(name="neutrality"),      # archived: gpt-4o-mini, same rubric
    coeffs=Coeffs(opinion=14, neutral=15),   # archived qwen-1.8b coeffs (notebook cell 37)
    method="mean_diff",                      # archived: coeff/n_layers, raw vector, all layers
    max_tokens=128,
    batch_size=32,                           # padding is per-batch; must match
)
