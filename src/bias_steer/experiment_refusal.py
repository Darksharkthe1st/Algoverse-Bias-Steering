"""`run_refusal(config)` — apply the paper's published refusal directions and
score refusal (arXiv:2406.11717).

Sibling to `experiment.run`. Where that pipeline captures residuals, builds a
steering vector, and steers, this one LOADS a pre-computed refusal direction and
runs the paper's two interventions on fixed eval prompts, scoring refusal by
substring match. There is no training and no judge API call.

Per model it produces the paper's five committed arms:

    harmful  (jailbreakbench):  baseline · ablation · act-add(coeff = -mag)
    harmless (alpaca):          baseline · act-add(coeff = +mag)

where `mag = config.coeffs.opinion` (the act-add dose; the raw direction's own
norm scales it further — see steering.apply_actadd_single). Ablation should DROP
harmful refusal; act-add(+) should RAISE harmless refusal.

Env-dependent ops funnel through `RefusalBackend` so the whole flow runs under a
fake backend without torch or a model (see tests/test_refusal.py).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..utils import get_current_time_str
from . import metrics, models, refusal, refusal_compare, refusal_extract, steering
from .config import DatasetSpec, ExperimentConfig
from .datasets import load_refusal_eval
from .logs import RunLogger
from .registry import JUDGES, MODELS, validate
from .schema import Result
from .tracking import append_index, git_sha, index_row, open_run


def _default_load_eval(run_dir: str, harm: str):
    """Eval Examples for one harm type, read from `<run_dir>`'s committed
    completions (byte-identical to the paper's prompts)."""
    spec = DatasetSpec(name="refusal_eval")
    spec.harm = harm
    spec.source_model = run_dir
    return load_refusal_eval(spec)


@dataclass
class RefusalBackend:
    """Everything needing the ML / data stack. Default = real; swap for a fake to
    run the wiring without torch or a model."""

    load: Callable = models.load_model
    generate: Callable = models.generate
    generate_with_hooks: Callable = models.generate_with_hooks
    load_direction: Callable = refusal.load_refusal_direction
    load_eval: Callable = _default_load_eval


@dataclass
class RefusalRunResult:
    run_id: str
    dir: Path
    results_csv: Path
    summary_md: Path
    rates: dict
    comparison: list = None  # our-vs-paper rows (refusal_compare.compare_rates), if fetched


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run_refusal(config: ExperimentConfig, *, backend: RefusalBackend | None = None,
                runs_dir="runs", index_path=None, progress=None, on_phase=None) -> list[RefusalRunResult]:
    """Run the refusal repro for each model in `config`; one result per model."""
    backend = backend or RefusalBackend()
    progress = progress or (lambda it, **kw: it)
    on_phase = on_phase or (lambda phase, run_id: None)
    config.validate()
    validate(config)  # models / dataset / method / judge all registered

    index_path = Path(index_path) if index_path else Path(runs_dir) / "index.csv"
    judge_fn = JUDGES[config.judge.name]

    return [
        _run_one(config, model_key, judge_fn, backend, runs_dir, index_path, progress, on_phase)
        for model_key in config.models
    ]


def _run_arm(config, backend, loaded, judge_fn, examples, harm, condition, coeff,
             hook_builder, sys_prompt, progress, log, template=None) -> list[Result]:
    """Generate + judge one arm. `hook_builder(model) -> fwd_hooks`, or None for
    the un-intervened baseline.

    `template` is the paper's literal prompt template (no system turn); see
    `models.render_prompts`. Passed only when set, so backends that predate the
    parameter (e.g. the test fakes) keep working."""
    kw = {} if template is None else {"template": template}
    out: list[Result] = []
    for batch in progress(list(_batches(examples, config.batch_size)), desc=condition):
        prompts = [e.prompt for e in batch]
        if hook_builder is None:
            responses = backend.generate(loaded, prompts, config.max_tokens, sys_prompt, **kw)
        else:
            hooks = hook_builder(loaded.model)
            responses = backend.generate_with_hooks(loaded, prompts, hooks, config.max_tokens, sys_prompt, **kw)
        verdicts = judge_fn(responses, batch, config.judge)
        for ex, resp, verdict in zip(batch, responses, verdicts):
            out.append(Result(ex.id, condition, resp, verdict,
                              {"harm": harm, "category": ex.metadata.get("category"), "coeff": coeff}))
    refused = sum(1 for r in out if r.verdict == "refusal")
    log.event(f"{condition}: {refused}/{len(out)} refused")
    return out


def _run_one(config, model_key, judge_fn, backend, runs_dir, index_path, progress, on_phase) -> RefusalRunResult:
    when = get_current_time_str()
    handle = open_run(config, model_key, runs_dir=runs_dir, when=when)
    log = RunLogger(handle.dir)
    spec = MODELS[model_key]
    sys_prompt = config.system_prompt
    mag = config.coeffs.opinion  # act-add dose magnitude

    log.event(f"loading model {spec.hf_id}")
    loaded = backend.load(spec)

    rd = backend.load_direction(model_key)
    norm = float(rd.direction.norm())
    log.event(f"loaded refusal direction: layer={rd.layer} pos={rd.pos} |r|={norm:.3f}")

    run_dir = refusal.MODEL_TO_RUN_DIR.get(model_key, model_key)
    harmful = backend.load_eval(run_dir, "harmful")
    harmless = backend.load_eval(run_dir, "harmless")

    def ablation(model):
        return steering.apply_directional_ablation(model, rd.direction)

    def actadd(c):
        return lambda model: steering.apply_actadd_single(model, rd.direction, coeff=c, layer=rd.layer)

    # The paper's five committed arms.
    arms = [
        (harmful,  "harmful",  "harmful/baseline",  0.0,  None),
        (harmful,  "harmful",  "harmful/ablation",  None, ablation),
        (harmful,  "harmful",  "harmful/actadd",   -mag,  actadd(-mag)),
        (harmless, "harmless", "harmless/baseline", 0.0,  None),
        (harmless, "harmless", "harmless/actadd",  +mag,  actadd(+mag)),
    ]

    # The paper formats with the model's chat template and NO system turn. Use its
    # literal template rather than `models.render_prompts`' system+user rendering —
    # for Qwen the latter emits an empty system turn, which cost -0.33 on
    # harmful/baseline (docs/05-refusal-repro.md §3). Falls back to the old
    # rendering for any model without a published template.
    try:
        template = refusal_extract.REFUSAL_TEMPLATES[
            refusal_extract._resolve_model_key(model_key)].template
        log.event(f"prompt template (paper, no system turn): {template!r}")
    except KeyError:
        template = None
        log.event(f"no paper template for {model_key}; falling back to chat template "
                  f"with system_prompt={sys_prompt!r}")

    results: list[Result] = []
    for examples, harm, condition, coeff, hook_builder in arms:
        results += _run_arm(config, backend, loaded, judge_fn, examples, harm,
                            condition, coeff, hook_builder, sys_prompt, progress, log,
                            template=template)

    # --- metrics + persistence ---
    rates = metrics.refusal_rates(results)
    rows = [
        {"run_id": handle.run_id, "model": model_key, "harm": r.metadata.get("harm"),
         "condition": r.condition, "coeff": r.metadata.get("coeff"),
         "example_id": r.example_id, "category": r.metadata.get("category"), "verdict": r.verdict}
        for r in results
    ]
    results_csv = handle.dir / "results.csv"
    metrics.write_rows(results_csv, rows, metrics.REFUSAL_RESULT_COLUMNS)

    # Diff against the paper's committed rates (if the eval files are fetched).
    theirs = refusal_compare.paper_rates(model_key)
    comparison = refusal_compare.compare_rates(rates, theirs)
    if comparison:
        n_ok = sum(r["within_tol"] for r in comparison)
        log.event(f"vs paper: {n_ok}/{len(comparison)} arms within tolerance")

    sha, dirty = git_sha()
    summary_md = handle.dir / "summary.md"
    summary = metrics.render_refusal_summary(
        run_id=handle.run_id, label=config.label, model=model_key, git=(sha, dirty),
        direction={"layer": rd.layer, "pos": rd.pos, "norm": norm}, coeff=mag, rates=rates,
    )
    if comparison:
        summary += "\n" + refusal_compare.render_comparison(comparison)
    summary_md.write_text(summary)

    row = index_row(config, model_key, handle.run_id, sha, dirty, when, status="done")
    row.update({"n_test": len(harmful) + len(harmless)})
    append_index(index_path, row)
    log.event("done")
    on_phase("eval", handle.run_id)

    return RefusalRunResult(handle.run_id, handle.dir, results_csv, summary_md, rates, comparison)
