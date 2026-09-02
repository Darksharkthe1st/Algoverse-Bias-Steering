"""Run adaptive_add at target in {2, 4, 8} sequentially (same GPU, one model load
per target -- no way around that since each target changes the hook's constant).

Per GPU_RUN_PROMPT.md's calibration instruction: "measure the baseline per-position
projection ... and pick a target from it; consider a small sweep (2, 4, 8)." The
measurement (logs/calibrate_adaptive_add.log) shows the per-layer baseline
projection median spans ~0.01 (layer 0) to ~109 (layer 35) -- about 4 orders of
magnitude -- so no single scalar target sits at a comparable point on every layer;
the sweep is reported with that caveat rather than picking one "calibrated" value
and presenting it as if it were layer-matched.

Each target gets its own label (folder name self-describes the target) via the
existing adaptive_add_qwen3_8b.py config, reusing its dataset/model/judge/vector
untouched -- only `coeffs` and `label` are overridden per iteration.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tqdm import tqdm

from src.bias_steer import experiment
from src.bias_steer.config import Coeffs
from configs.exp.adaptive_add_qwen3_8b import config as base_config

TARGETS = [2.0, 4.0, 8.0]


def _emit_phase(phase, run_id):
    print(f"::bias-steer:phase:{phase}:{run_id}", flush=True)


def main():
    prog = lambda it, **kw: tqdm(list(it), **kw)
    for target in TARGETS:
        cfg = base_config
        cfg.label = f"adaptive-add-target{target:g} qwen3-8b"
        cfg.coeffs = Coeffs(opinion=target, neutral=target)
        print(f"\n=== adaptive_add sweep: target={target} ===", flush=True)
        results = experiment.run(cfg, vector_path=cfg.vector_path, runs_dir="runs",
                                 progress=prog, on_phase=_emit_phase)
        for r in results:
            print(f"\ndone: {r.dir}\n  summary: {r.summary_md}\n  results: {r.results_csv}")


if __name__ == "__main__":
    main()
