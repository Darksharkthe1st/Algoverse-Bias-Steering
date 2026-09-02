"""Run a coefficient sweep for an already-built steering vector, on the GPU box.

Point it at a run folder produced by `python -m src.bias_steer run <config>`
(one that holds `steering_vector.safetensors` + `manifest.json`). It rebuilds
that run's exact TEST split from the manifest — so the sweep is on held-out data,
never the vector's own build split (the scope guardrail) — then sweeps a small
coefficient grid, judging each with the same judge, and writes a one-row-per-coeff
CSV plus prints the chosen c*.

Usage:
    python scripts/run_coeff_sweep.py runs/<vector_run_id>/

Requires (same as a normal run): torch + transformer_lens, `HF_TOKEN` for the
model download, `OPENAI_API_KEY` for the neutrality judge. Everything below the
CONFIG block is wiring you shouldn't need to touch.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer import artifacts, datasets, models  # noqa: E402
from src.bias_steer.coeff_sweep import sweep_coeff  # noqa: E402
from src.bias_steer.config import from_dict  # noqa: E402
from src.bias_steer.judge import neutrality_judge  # noqa: E402
from src.bias_steer.registry import DATASETS, MODELS  # noqa: E402

# ============================== CONFIG ======================================= #
# The coefficients to try. c=0 is the unsteered baseline; both signs are swept.
COEFF_GRID = (-8, -4, 0, 4, 8)

# Which judge label counts as "steered the way we want". For the neutrality judge
# (labels ["neutral", "opinionated"]) the positive pole is "opinionated".
TARGET_LABEL = "opinionated"

# Labels that mean the model broke rather than took the target behaviour. The
# neutrality judge emits "nonsense" for unparseable output. A coefficient that
# pushes more than MAX_GUARD_FRAC of items here cannot be chosen.
GUARD_LABELS = ("nonsense",)
MAX_GUARD_FRAC = 0.5

MAX_TOKENS = 128
BATCH_SIZE = 16
# ============================================================================ #


def main(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = from_dict(manifest["config"])

    # Rebuild the vector run's TEST split exactly (same loader, sample, seed,
    # shuffle, train_split as src/bias_steer/experiment.run) and sweep only on it.
    examples = DATASETS[cfg.dataset.name](cfg.dataset)
    examples = datasets.sample(examples, cfg.sample)
    if cfg.dataset.shuffle:
        random.Random(cfg.sample.seed).shuffle(examples)
    n_train = int(len(examples) * cfg.dataset.train_split)
    test = examples[n_train:]
    if not test:
        raise SystemExit(f"no held-out examples in {run_dir} (train_split too high?)")

    model_key = manifest["model"]
    print(f"loading {model_key} ({MODELS[model_key].hf_id}) ...")
    loaded = models.load_model(MODELS[model_key])
    vector = artifacts.load_vector(run_dir / "steering_vector.safetensors")

    out_csv = run_dir / "coeff_sweep.csv"
    result = sweep_coeff(
        loaded, vector, test,
        judge=neutrality_judge, judge_spec=cfg.judge,
        target_label=TARGET_LABEL, coeff_grid=COEFF_GRID,
        guard_labels=GUARD_LABELS, max_guard_frac=MAX_GUARD_FRAC,
        max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE, out_csv=out_csv,
    )

    print(f"\nheld-out items: {len(test)}   target label: {TARGET_LABEL!r}")
    print(f"{'coeff':>6} {'target_rate':>12} {'guard_frac':>11}")
    for c in COEFF_GRID:
        mark = "  <- c*" if c == result.c_star else ""
        print(f"{c:>6} {result.rates[c]:>12.3f} {result.guard_fracs[c]:>11.3f}{mark}")
    print(f"\nchosen c* = {result.c_star}   (wrote {out_csv})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/run_coeff_sweep.py runs/<vector_run_id>/")
    main(sys.argv[1])
