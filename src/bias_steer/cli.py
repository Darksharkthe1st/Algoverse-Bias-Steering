"""Command-line entry point (arch roadmap §7.3, §10).

Default: run a single experiment from a config file (what a teammate uses):

    python -m src.bias_steer run path/to/config.py

`--queue` (the batch coordinator) is Phase 4 and not available yet. tqdm is
imported lazily so the module imports without it.
"""

import argparse
import os
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


def _load_env() -> None:
    """Load `.env` from the repo root into os.environ (see `.env.example`).

    Secrets are read via `os.getenv` at call time (judge.py) or by libraries
    (huggingface_hub reads HF_TOKEN), so they only have to be in the environment
    before a run starts. Shell-exported values take precedence — we never
    overwrite something already set.

    Deliberately stdlib: a six-line parser beats a dependency, and it keeps the
    "package imports on any machine" invariant intact. Called from `main`, not at
    import time, so importing the CLI stays side-effect free for tests.
    """
    from ..utils import get_repo_root

    path = get_repo_root() / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip one layer of matching quotes, the one dotenv-ism worth supporting.
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _emit_phase(phase, run_id):
    # stdout sentinel the coordinator parses to commit/push per phase (§10.4).
    print(f"::bias-steer:phase:{phase}:{run_id}", flush=True)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    _load_env()

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

    # Preflight: the judge runs in BOTH phases, so a missing key means the run
    # dies after the model load rather than before it. Check it up front.
    if not os.getenv("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is not set, and the judge is needed by both "
              "the train and eval phases.\n"
              "       Put it in .env at the repo root (see .env.example) or export it.")
        return 2

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
