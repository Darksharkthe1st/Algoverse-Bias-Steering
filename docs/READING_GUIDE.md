# Codebase Reading Guide — progress tracker

A dependency-ordered walk through the `src/bias_steer/` pipeline. Read each file
*after* everything it imports, so understanding builds up. Tick a box (`[x]`) as
you finish each item. Read a cluster, then read its phase test to see the
contracts exercised.

> Focus: the **new package** (`src/bias_steer/`). Legacy code is an optional
> appendix at the bottom.

**Suggested first sitting:** `schema.py` → `config.py` → `registry.py`, then run
`python3 tests/test_phase0.py`. Once `Example`/`Result`, `ExperimentConfig`, and
the registries click, every other module reads as an application of them.

---

## Step 0 — Design docs (the map)

Read these before any code — the package is a direct implementation of them.

- [ ] `docs/00-overview.md` — the four goals (traceability, extensibility, interpretability, configurability)
- [ ] `docs/02-architecture-roadmap.md` — **most important.** §3 contracts, §4 config, §5 file map + data-flow, §7 outputs, §10 coordinator
- [ ] `docs/01-feature-roadmap.md` — skim
- [ ] `docs/03-gpu-bringup.md` — skim
- [ ] `docs/needed-experiments.md` — skim
- [ ] `README.md` + `setup.py` — packaging / entrypoints (2 min)

---

## Cluster A — Foundation (no internal deps)

- [ ] `src/utils.py` (28) — `get_repo_root`, time strings; used everywhere
- [ ] `src/bias_steer/schema.py` (42) — `Example`, `Result` + condition constants; the "currency" every module speaks. Read slowly
- [ ] `src/bias_steer/config.py` (159) — `ExperimentConfig` + sub-specs; *every lever in the system*. Map fields to §4
- [ ] `src/bias_steer/registry.py` (74) — `DATASETS/MODELS/METHODS/JUDGES` + `register()` + `validate()`; the name→component indirection
- [ ] `src/bias_steer/artifacts.py` (32) — safetensors read/write for vectors/residuals
- [ ] **Test:** `tests/test_phase0.py` — schema, config, registry, tracking

## Cluster B — Swappable components (the "science" you'll edit most)

- [ ] `src/bias_steer/datasets.py` (151) — `load_*() -> list[Example]` + dataset-agnostic `sample()` (filter → stratify → cap); wraps `src/data.py`
- [ ] `src/bias_steer/models.py` (132) — `ModelSpec` → `LoadedModel`, chat-template/tokenization, generation
- [ ] `src/bias_steer/steering.py` (107) — **the heart:** `capture` / `build` / `apply` (mean-diff steering). Cross-ref §3.5
- [ ] `src/bias_steer/judge.py` (93) — LLM-as-judge, `ANSWER: <label>` parsing + retry
- [ ] **Test:** `tests/test_phase1.py` — datasets+sampling, steering wiring, judge parsing

## Cluster C — Output / plumbing

- [ ] `src/bias_steer/metrics.py` (132) — verdicts → tidy rows + transition/tally aggregates
- [ ] `src/bias_steer/logs.py` (51) — `run.log` / `train.txt` / `eval.txt` plaintext writers
- [ ] `src/bias_steer/tracking.py` (138) — run-id slug, `manifest.json`, git SHA, `index.csv`

## Cluster D — Orchestration (read last; wires everything above)

- [ ] `src/bias_steer/experiment.py` (183) — **`run(config)`**, the spine. Follow it against the §5.2 data-flow trace
- [ ] `src/bias_steer/cli.py` (84) — entrypoint; default = one run, `--queue` = batch
- [ ] `src/bias_steer/coordinator.py` (191) — the `--queue` engine: route → checkout → drain → commit/push. Read `configs/route.example.json` + `configs/example_bbq.py` alongside
- [ ] `src/bias_steer/__init__.py` (63) + `__main__.py` (5) — public surface / `python -m` glue. Read `__init__.py` last, as a table of contents
- [ ] **Test:** `tests/test_phase2.py` — `run()` end-to-end, metrics, logs, CLI

## Step 2 — Analysis + plug-and-play

- [ ] `analysis/compare.py` (86) + `analysis/__init__.py` — standalone; reads run outputs, never imports the engine
- [ ] **Test:** `tests/test_phase3.py` — standalone analysis + adding a new dataset + method
- [ ] **Test:** `tests/test_phase4.py` — the batch coordinator + phase-signal plumbing

---

## How to work through it

- Read a cluster → read its phase test → make a small change (rename a field, add
  a `metadata` key, register a dummy dataset) → run `python3 tests/test_phaseN.py`
  to confirm you understood the contract.
- Keep `docs/02-architecture-roadmap.md` §3 and §5.2 open the whole time.

---

## Appendix — Legacy layer (optional; read only for contrast)

The original monolithic pipeline the new package replaced.

- [ ] `src/data.py` (162) — raw dataset loaders (still called by `datasets.py`)
- [ ] `src/main.py` (110) — old tokenize/generate/judge flow
- [ ] `src/stereoset-dataloader.py` (224) — StereoSet loader
- [ ] `experiments/farhan-experimentation.ipynb` — the original monolith
- [ ] `examples/main.ipynb`, `examples/data.ipynb` — usage examples
