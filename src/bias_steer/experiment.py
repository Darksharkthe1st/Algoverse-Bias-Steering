"""`run(config)` — the pipeline wired end-to-end (arch roadmap §5.2).

A run is a straight function from config to outputs (no ledger, no resume). For
each model in the config:

    load model
    TRAIN split:  generate + capture residuals -> judge -> bucket by verdict
    build steering vector
    TEST split:   initial + steered_pos + steered_neg -> judge -> Results
    metrics -> results.csv + summary.md, append runs/index.csv

The env-dependent operations (model load, generation, tensor saving) are gathered
in `Backend`, whose default is the real implementation. Tests inject a fake so the
whole wiring runs without torch/OpenAI; the numeric correctness of capture/build
lives in the (torch-gated) steering tests.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..utils import get_current_time_str
from . import artifacts, datasets, metrics, models
from .config import ExperimentConfig
from .logs import RunLogger
from .registry import DATASETS, JUDGES, METHODS, MODELS, validate
from .schema import INITIAL, STEERED_NEG, STEERED_POS, Result
from .tracking import append_index, git_sha, index_row, open_run


@dataclass
class Backend:
    """Everything that needs the ML / serialization stack. Default = real; swap for
    a fake to run the wiring without torch/OpenAI."""

    load: Callable = models.load_model
    generate: Callable = models.generate
    generate_with_cache: Callable = models.generate_with_cache
    generate_with_hooks: Callable = models.generate_with_hooks
    save_vector: Callable = artifacts.save_vector
    save_residuals: Callable = artifacts.save_residuals


@dataclass
class RunResult:
    run_id: str
    dir: Path
    results_csv: Path
    summary_md: Path
    counts: dict
    quality: dict


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _contrast(config: ExperimentConfig):
    """(positive_label, negative_label). Default: the judge's two labels, with the
    second as the positive pole (matching the notebook's opinion − neutral)."""
    labels = config.judge.labels
    return labels[1], labels[0]


def run(config: ExperimentConfig, *, backend: Backend | None = None,
        runs_dir="runs", index_path=None, progress=None, on_phase=None) -> list[RunResult]:
    """Run `config` for each of its models; returns one `RunResult` per model.

    `on_phase(phase, run_id)` is called at each persistence boundary ("vector" after
    the steering vector is saved, "eval" after results/index are written). run()
    stays git-agnostic; the coordinator (§10) supplies this callback to commit/push
    per phase. Default None = no-op.
    """
    backend = backend or Backend()
    progress = progress or (lambda it, **kw: it)
    on_phase = on_phase or (lambda phase, run_id: None)
    config.validate()
    validate(config)

    index_path = Path(index_path) if index_path else Path(runs_dir) / "index.csv"

    # Resolve + materialize the dataset once (shared across this config's models).
    examples = DATASETS[config.dataset.name](config.dataset)
    examples = datasets.sample(examples, config.sample)
    if config.dataset.shuffle:
        random.Random(config.sample.seed).shuffle(examples)  # balanced train/test split
    n_train = int(len(examples) * config.dataset.train_split)
    train, test = examples[:n_train], examples[n_train:]

    method = METHODS[config.method]
    judge_fn = JUDGES[config.judge.name]
    contrast = _contrast(config)

    return [
        _run_one(config, model_key, train, test, method, judge_fn, contrast,
                 backend, runs_dir, index_path, progress, on_phase)
        for model_key in config.models
    ]


#: Files a completed run must have written. `logs/run.log` is deliberately NOT
#: here — it is the one file a hollow run still produces, so it is worthless as
#: evidence of success.
REQUIRED_RUN_ARTIFACTS = (
    "results.csv",
    "summary.md",
    "manifest.json",
    "steering_vector.safetensors",
)


class IncompleteRunError(RuntimeError):
    """A run reached the end without writing its evidence."""


def assert_run_artifacts(run_dir, required=REQUIRED_RUN_ARTIFACTS) -> None:
    """Raise unless every required artifact exists and is non-empty.

    Status is derived from evidence, not from control flow. Without this a run
    can log "done", append a `status=done` index row, and leave nothing behind
    that anyone could analyse — which is indistinguishable, downstream, from a
    successful run.
    """
    run_dir = Path(run_dir)
    missing = [
        name for name in required
        if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0
    ]
    if missing:
        raise IncompleteRunError(
            f"run {run_dir.name} reached completion without writing "
            f"{', '.join(missing)}. Not indexing it as done — a run is complete "
            f"when its evidence exists, not when the code reaches the end."
        )


def _run_one(config, model_key, train, test, method, judge_fn, contrast,
             backend, runs_dir, index_path, progress, on_phase) -> RunResult:
    when = get_current_time_str()
    handle = open_run(config, model_key, runs_dir=runs_dir, when=when)
    log = RunLogger(handle.dir)
    spec = MODELS[model_key]
    sys_prompt = config.system_prompt

    log.event(f"loading model {spec.hf_id}")
    loaded = backend.load(spec)
    n_layers = loaded.model.cfg.n_layers

    # --- TRAIN: capture residuals, judge, bucket by verdict --------------------
    resids_by_label: dict = {}
    for batch in progress(list(_batches(train, config.batch_size)), desc=f"{model_key} train"):
        prompts = [e.prompt for e in batch]
        responses, caches = backend.generate_with_cache(
            loaded, prompts, config.max_tokens, sys_prompt,
            capture_names=method.names(n_layers),
        )
        verdicts = judge_fn(responses, batch, config.judge)
        for ex, resp, cache, verdict in zip(batch, responses, caches, verdicts):
            resids_by_label.setdefault(verdict, []).append(method.capture(cache, n_layers))
            log.train(ex, resp, verdict)

    log.event(f"building steering vector (buckets: "
              f"{ {k: len(v) for k, v in resids_by_label.items()} })")
    vector = method.build(resids_by_label, contrast)
    backend.save_vector(handle.dir / "steering_vector.safetensors", vector)
    backend.save_residuals(handle.dir / "residuals.safetensors", resids_by_label)
    on_phase("vector", handle.run_id)  # steering vector persisted -> coordinator commits/pushes

    # --- TEST: initial + steered (both directions), judge each -----------------
    results: list[Result] = []
    for batch in progress(list(_batches(test, config.batch_size)), desc=f"{model_key} eval"):
        prompts = [e.prompt for e in batch]
        initial = backend.generate(loaded, prompts, config.max_tokens, sys_prompt)
        pos_hooks = method.apply(loaded.model, vector, config.coeffs.opinion)
        steered_pos = backend.generate_with_hooks(loaded, prompts, pos_hooks, config.max_tokens, sys_prompt)
        neg_hooks = method.apply(loaded.model, vector, -config.coeffs.neutral)
        steered_neg = backend.generate_with_hooks(loaded, prompts, neg_hooks, config.max_tokens, sys_prompt)

        j_init = judge_fn(initial, batch, config.judge)
        j_pos = judge_fn(steered_pos, batch, config.judge)
        j_neg = judge_fn(steered_neg, batch, config.judge)

        for i, ex in enumerate(batch):
            meta = {"category": ex.metadata.get("category")}
            triple = [
                Result(ex.id, INITIAL, initial[i], j_init[i], dict(meta)),
                Result(ex.id, STEERED_POS, steered_pos[i], j_pos[i], dict(meta)),
                Result(ex.id, STEERED_NEG, steered_neg[i], j_neg[i], dict(meta)),
            ]
            results.extend(triple)
            log.eval(ex, triple)

    # --- metrics + persistence -------------------------------------------------
    rows = metrics.tidy_rows(
        results, run_id=handle.run_id, model=model_key, dataset=config.dataset.name,
        opin_coeff=config.coeffs.opinion, neut_coeff=config.coeffs.neutral,
    )
    results_csv = handle.dir / "results.csv"
    metrics.write_csv(results_csv, rows)

    counts = metrics.condition_verdict_counts(results)
    quality = metrics.steering_quality(results, pos_label=contrast[0], neg_label=contrast[1])

    sha, dirty = git_sha()
    summary_md = handle.dir / "summary.md"
    summary_md.write_text(metrics.render_summary(
        run_id=handle.run_id, label=config.label, model=model_key,
        dataset=config.dataset.name, coeffs=config.coeffs, git=(sha, dirty),
        n_train=len(train), n_test=len(test), counts=counts, quality=quality,
    ))

    # A run is "done" only if its evidence is on disk. Reaching this line is not
    # evidence: the Aug-9 campaign logged "done" for 13 runs and 12 of them were
    # later found holding a 167-byte log and nothing else (the artifacts turned
    # out to be recoverable from phase commits, but nothing here would have
    # noticed either way). Fail loudly rather than index a hollow run.
    assert_run_artifacts(handle.dir)

    row = index_row(config, model_key, handle.run_id, sha, dirty, when, status="done")
    row.update({
        "n_train": len(train), "n_test": len(test),
        "opin_good": quality["opinion"]["good"], "neut_good": quality["neutral"]["good"],
    })
    append_index(index_path, row)
    log.event("done")
    on_phase("eval", handle.run_id)  # results + index persisted -> coordinator commits/pushes

    return RunResult(handle.run_id, handle.dir, results_csv, summary_md, counts, quality)
