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

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..utils import get_current_time_str
from . import artifacts, datasets, metrics, models, steering
from .config import ExperimentConfig
from .logs import RunLogger
from .registry import DATASETS, JUDGES, METHODS, MODELS, validate
from .schema import INITIAL, STEERED_NEG, STEERED_POS, PROMPT_POS, PROMPT_NEG, Result
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
    # Read a saved (n_layers, d_model) vector back off disk — the input to the
    # apply-only path (apply_vector). Appended last so existing Backend(...) calls
    # are unaffected. Tests inject a fake to avoid a real safetensors read.
    load_vector: Callable = artifacts.load_vector


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


def _build_arms(config: ExperimentConfig, method, vector, loaded):
    """The eval arms for this run's `intervention`: `(condition, system_prompt, hooks)`.

    `hooks=None` is a plain generate; a hook list is steered generation. INITIAL is
    always the control (default system prompt, no steering). The steer arms exist
    only when a vector is available (`steer`/`both`); the prompt arms swap in a
    behaviour-inducing system prompt and add no hooks (`prompt`/`both`). This is the
    single place the intervention mode fans out into conditions — see
    needed-experiments §14. Hooks are built once here (torch-free closures over the
    vector) and reused across every batch.
    """
    default_sys = config.system_prompt
    arms = [(INITIAL, default_sys, None)]
    if config.intervention in ("steer", "both"):
        if vector is None:  # defended in _run_one; belt-and-braces against a silent no-op
            raise ValueError(f"intervention={config.intervention!r} needs a steering vector")
        arms.append((STEERED_POS, default_sys,
                     method.apply(loaded.model, vector, config.coeffs.opinion)))
        arms.append((STEERED_NEG, default_sys,
                     method.apply(loaded.model, vector, -config.coeffs.neutral)))
    if config.intervention in ("prompt", "both"):
        arms.append((PROMPT_POS, config.pos_system_prompt, None))
        arms.append((PROMPT_NEG, config.neg_system_prompt, None))
    return arms


def run(config: ExperimentConfig, *, vector_path=None, backend: Backend | None = None,
        runs_dir="runs", index_path=None, progress=None, on_phase=None) -> list[RunResult]:
    """Run `config` for each of its models; returns one `RunResult` per model.

    By default each model extracts a steering vector from the TRAIN split and
    evaluates it on TEST. If `vector_path` (or `config.vector_path`) names a saved
    `steering_vector.safetensors`, extraction is **skipped** and that vector is
    evaluated instead — the same eval path, just fed a pre-existing direction (the
    FK-5 generalization test: apply a direction fit on one battery to another). A
    supplied vector needs no TRAIN split; if one is present anyway it is folded
    into the eval set and a loud warning is logged (not an error).

    `on_phase(phase, run_id)` is called at each persistence boundary ("vector" after
    the steering vector is saved/loaded, "eval" after results/index are written).
    run() stays git-agnostic; the coordinator (§10) supplies this callback to
    commit/push per phase. Default None = no-op.
    """
    backend = backend or Backend()
    progress = progress or (lambda it, **kw: it)
    on_phase = on_phase or (lambda phase, run_id: None)
    config.validate()
    validate(config)

    vector_path = vector_path or getattr(config, "vector_path", None)
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
                 backend, runs_dir, index_path, progress, on_phase,
                 vector_path=vector_path)
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


def _evaluate_and_persist(config, model_key, handle, log, loaded, vector, *,
                          method, judge_fn, contrast, eval_examples, snapshot_examples,
                          n_train, backend, index_path, when, progress, on_phase,
                          phase_desc="eval") -> RunResult:
    """Shared eval + persist tail for BOTH steering paths.

    `run()` builds the vector from a TRAIN split; `apply_vector()` loads it from
    disk. Those are the only steps that differ — from here (apply the vector, run
    the TEST phase, score, write the run folder) the two are byte-for-byte the same
    code, so they share it rather than each keeping a copy (DRY). `eval_examples`
    is what the TEST phase scores; `snapshot_examples` is what examples.csv records
    (run() snapshots train+test; apply snapshots the eval set); `n_train` and
    `phase_desc` only affect the summary/index label and the progress bar.
    """
    # Move the vector on-device ONCE: it is applied at every layer on every forward
    # step of the TEST phase, so a per-hook-fire transfer would recopy it thousands
    # of times. `build`/`load_vector` produce a CPU tensor; one host->device copy
    # here covers the whole phase (#9). A prompt-only run carries no vector.
    if vector is not None:
        vector = vector.to(loaded.device)

    arms = _build_arms(config, method, vector, loaded)

    # --- TEST: run every arm for this intervention, judge each ------------------
    results: list[Result] = []
    for batch in progress(list(_batches(eval_examples, config.batch_size)),
                          desc=f"{model_key} {phase_desc}"):
        prompts = [e.prompt for e in batch]
        # Each arm is a plain generate (hooks None) or a steered one; they differ
        # only in the system prompt and whether hooks are attached.
        per_arm: dict = {}
        for cond, arm_sys, hooks in arms:
            if hooks is None:
                resp = backend.generate(loaded, prompts, config.max_tokens, arm_sys)
            else:
                resp = backend.generate_with_hooks(loaded, prompts, hooks, config.max_tokens, arm_sys)
            per_arm[cond] = (resp, judge_fn(resp, batch, config.judge))

        for i, ex in enumerate(batch):
            meta = {"category": ex.metadata.get("category")}
            row = [
                Result(ex.id, cond, per_arm[cond][0][i], per_arm[cond][1][i], dict(meta))
                for cond, _, _ in arms
            ]
            results.extend(row)
            log.eval(ex, row)

    # --- metrics + persistence -------------------------------------------------
    rows = metrics.tidy_rows(
        results, run_id=handle.run_id, model=model_key, dataset=config.dataset.name,
        opin_coeff=config.coeffs.opinion, neut_coeff=config.coeffs.neutral,
    )
    results_csv = handle.dir / "results.csv"
    metrics.write_csv(results_csv, rows)
    # Snapshot the frozen subset this run used, so the folder holds its own inputs.
    metrics.write_examples_csv(handle.dir / "examples.csv", snapshot_examples,
                               dataset=config.dataset.name)

    counts = metrics.condition_verdict_counts(results)
    pos_label, neg_label = contrast
    has_steer = config.intervention in ("steer", "both")
    has_prompt = config.intervention in ("prompt", "both")

    # Score each present arm-pair with the identical rule; the prompt baseline reuses
    # steering_quality pointed at the PROMPT arms (needed-experiments §14).
    steer_quality = metrics.steering_quality(
        results, pos_label=pos_label, neg_label=neg_label) if has_steer else None
    prompt_quality = metrics.steering_quality(
        results, pos_label=pos_label, neg_label=neg_label,
        pos_cond=PROMPT_POS, neg_cond=PROMPT_NEG) if has_prompt else None

    # The headline question — did the vector beat prompting? — is a per-item paired
    # comparison, and only exists when BOTH arms ran (intervention="both").
    comparisons = None
    if has_steer and has_prompt:
        comparisons = {
            "opinion": metrics.beat_rate(results, target_label=pos_label,
                                         steer_cond=STEERED_POS, prompt_cond=PROMPT_POS),
            "neutral": metrics.beat_rate(results, target_label=neg_label,
                                         steer_cond=STEERED_NEG, prompt_cond=PROMPT_NEG),
        }

    # Index headline: the vector arms when present, else the prompt arms.
    headline = steer_quality or prompt_quality

    sha, dirty = git_sha()
    summary_md = handle.dir / "summary.md"
    # encoding pinned: render_summary emits non-ASCII and the default write_text
    # codec is cp1252 on Windows, where a future non-cp1252 glyph would raise.
    summary_md.write_text(encoding="utf-8", data=metrics.render_summary(
        run_id=handle.run_id, label=config.label, model=model_key,
        dataset=config.dataset.name, coeffs=config.coeffs, git=(sha, dirty),
        n_train=n_train, n_test=len(eval_examples), counts=counts,
        quality=steer_quality, intervention=config.intervention,
        prompt_quality=prompt_quality, comparisons=comparisons,
    ))

    # A run is "done" only if its evidence is on disk. Reaching this line is not
    # evidence (see assert_run_artifacts) — fail loudly rather than index a hollow run.
    # A prompt-only run produces no steering vector, so it is not required of one.
    required = REQUIRED_RUN_ARTIFACTS
    if vector is None:
        required = tuple(a for a in required if a != "steering_vector.safetensors")
    assert_run_artifacts(handle.dir, required)

    row = index_row(config, model_key, handle.run_id, sha, dirty, when, status="done")
    row.update({
        "n_train": n_train, "n_test": len(eval_examples),
        "opin_good": headline["opinion"]["good"], "neut_good": headline["neutral"]["good"],
    })
    append_index(index_path, row)
    log.event("done")
    on_phase("eval", handle.run_id)  # results + index persisted -> coordinator commits/pushes

    return RunResult(handle.run_id, handle.dir, results_csv, summary_md, counts, headline)


def _extract_vector(config, model_key, train, loaded, method, judge_fn, contrast,
                    backend, handle, log, n_layers, d_model, progress):
    """Generate the steering vector from the TRAIN split (the default source).

    TRAIN phase: generate on `train`, judge each response, bucket residuals by the
    verdict, then `method.build` the mean-difference direction. Persists the vector
    and the per-bucket residuals into the run folder. Called by `_run_one` ONLY
    when no vector was supplied — it is the "make a new vector" half of a run.
    """
    sys_prompt = config.system_prompt
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
    backend.save_vector(handle.dir / "steering_vector.safetensors", vector,
                        n_layers=n_layers, d_model=d_model)
    backend.save_residuals(handle.dir / "residuals.safetensors", resids_by_label,
                           n_layers=n_layers, d_model=d_model)
    return vector


def _load_provided_vector(backend, vector_path, handle, log, n_layers, d_model):
    """Load a pre-extracted steering vector from disk, in place of extraction.

    Vouches for its `(n_layers, d_model)` shape (CLAUDE.md §6: a silently 1-D
    vector broadcasts a DC offset instead of steering — the 2025 bug), snapshots it
    into the run folder so the run is self-contained, and records its source.
    """
    log.event(f"loading steering vector from {vector_path}")
    vector = backend.load_vector(vector_path)
    steering.assert_steering_shape(vector, n_layers, d_model)
    backend.save_vector(handle.dir / "steering_vector.safetensors", vector,
                        n_layers=n_layers, d_model=d_model)
    (handle.dir / "applied_vector.json").write_text(json.dumps(
        {"source_vector_path": str(vector_path), "n_layers": n_layers, "d_model": d_model},
        indent=2))
    return vector


def _run_one(config, model_key, train, test, method, judge_fn, contrast,
             backend, runs_dir, index_path, progress, on_phase, vector_path=None) -> RunResult:
    """One model's run: obtain a steering vector, then evaluate + persist it.

    The vector comes from ONE of two sources, and that is the *only* branch — a
    supplied `vector_path` is loaded (extraction skipped); otherwise it is
    generated from the TRAIN split (`_extract_vector`). Everything after — the TEST
    phase, metrics, run folder — is the single shared tail, so a generated vector
    and a provided one are evaluated by identical code.
    """
    when = get_current_time_str()
    handle = open_run(config, model_key, runs_dir=runs_dir, when=when)
    log = RunLogger(handle.dir)
    spec = MODELS[model_key]

    log.event(f"loading model {spec.hf_id}")
    loaded = backend.load(spec)
    n_layers = loaded.model.cfg.n_layers
    d_model = loaded.model.cfg.d_model

    if config.intervention == "prompt":
        # Pure prompt baseline: no steering vector is fit or loaded at all — the
        # intervention is the system prompt (needed-experiments §14). No TRAIN split
        # is needed, so the whole sampled set is evaluated. A supplied vector here is
        # meaningless, so flag it rather than silently ignore it.
        if vector_path:
            log.event(
                f"WARNING: intervention='prompt' — supplied vector ({vector_path}) is "
                "IGNORED; this run steers by system prompt only, not by a direction."
            )
        vector = None
        eval_examples, n_train_label, phase_desc = train + test, 0, "prompt"
    elif vector_path:
        # A vector was supplied: skip extraction. A TRAIN split is not needed to
        # hold out here, so it is folded into the eval set rather than wasted — but
        # a *non-empty* train split usually means "please extract", so if one is
        # present say so loudly (not an error): we are NOT extracting from it.
        if train:
            log.event(
                f"WARNING: steering vector supplied ({vector_path}) — NOT extracting. "
                f"The {len(train)} TRAIN examples will be evaluated, not used to fit a vector."
            )
        vector = _load_provided_vector(backend, vector_path, handle, log, n_layers, d_model)
        eval_examples, n_train_label, phase_desc = train + test, 0, "apply"
    else:
        vector = _extract_vector(config, model_key, train, loaded, method, judge_fn,
                                 contrast, backend, handle, log, n_layers, d_model, progress)
        eval_examples, n_train_label, phase_desc = test, len(train), "eval"

    if vector is not None:
        on_phase("vector", handle.run_id)  # steering vector persisted -> coordinator commits/pushes

    return _evaluate_and_persist(
        config, model_key, handle, log, loaded, vector,
        method=method, judge_fn=judge_fn, contrast=contrast,
        eval_examples=eval_examples, snapshot_examples=train + test, n_train=n_train_label,
        backend=backend, index_path=index_path, when=when, progress=progress,
        on_phase=on_phase, phase_desc=phase_desc,
    )



    return RunResult(handle.run_id, handle.dir, results_csv, summary_md, counts, quality)
