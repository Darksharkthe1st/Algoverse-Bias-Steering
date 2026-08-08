"""Command-line entry point (arch roadmap §7.3, §10).

Default: run a single experiment from a config file (what a teammate uses):

    python -m src.bias_steer run path/to/config.py

`--queue` (the batch coordinator) is Phase 4 and not available yet. tqdm is
imported lazily so the module imports without it.
"""

import argparse
from pathlib import Path

from .config import ExperimentConfig


def load_config_file(path) -> ExperimentConfig:
    """Import a Python config file and return its module-level `config`
    (an `ExperimentConfig`)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bias_steer_cfg", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "config", None)
    if not isinstance(cfg, ExperimentConfig):
        raise ValueError(f"{path} must define a module-level `config: ExperimentConfig`")
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bias_steer", description="Run a bias-steering experiment.")
    p.add_argument("command", nargs="?", default="run", choices=["run"],
                   help="what to do (only 'run' in Phase 2)")
    p.add_argument("config", nargs="?", help="path to a Python config file defining `config`")
    p.add_argument("--runs-dir", default="runs", help="where run folders are written")
    p.add_argument("--queue", action="store_true",
                   help="drain _coordinator/route.json across branches (the batch coordinator, §10)")
    return p


def _emit_phase(phase, run_id):
    # stdout sentinel the coordinator parses to commit/push per phase (§10.4).
    print(f"::bias-steer:phase:{phase}:{run_id}", flush=True)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.queue:
        from ..utils import get_repo_root
        route = get_repo_root() / "_coordinator" / "route.json"
        if not route.exists():
            print(f"error: no route file at {route}\n"
                  f"create it to use --queue (see docs/02-architecture-roadmap.md §10.3).")
            return 2
        from .coordinator import Coordinator
        print(f"coordinator: draining {route} ...")
        Coordinator(runs_dir=args.runs_dir).run()
        return 0

    if not args.config:
        print("error: a config file path is required (e.g. `run configs/my_exp.py`)")
        return 2

    cfg = load_config_file(args.config)
    print(f"experiment: {cfg.label}")
    print(f"  models:  {', '.join(cfg.models)}")
    print(f"  dataset: {cfg.dataset.name}  judge: {cfg.judge.name}  method: {cfg.method}")
    print(f"  coeffs:  opinion={cfg.coeffs.opinion} neutral={cfg.coeffs.neutral}\n")

    from tqdm import tqdm  # lazy: only the actual run needs it

    from . import experiment
    results = experiment.run(cfg, runs_dir=args.runs_dir,
                             progress=lambda it, **kw: tqdm(list(it), **kw),
                             on_phase=_emit_phase)
    for r in results:
        print(f"\ndone: {r.dir}\n  summary: {r.summary_md}\n  results: {r.results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
