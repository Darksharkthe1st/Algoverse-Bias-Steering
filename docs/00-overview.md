# Bias-Steering Experiment Pipeline — Re-architecture Docs

This folder tracks the design work for re-implementing the experimentation pipeline
that currently lives entirely inside
[`experiments/farhan-experimentation.ipynb`](../experiments/farhan-experimentation.ipynb).

The goal of the re-architecture is **not** to change the science. It is to take a
working-but-monolithic notebook and turn it into a small, module-based system that fixes the
four things that actually cost time in the research loop:

1. **Traceability** — always know *which* experiment produced a result and *what code* ran it.
   A per-run `manifest.json` (full config + git SHA) and a browsable `runs/index.csv` replace
   guessing from folder names like `Log_220`.
2. **Plug-and-play extensibility** — add a dataset / model / steering technique / judge without
   rewriting code everything else shares. A canonical `Example`/`Result` schema plus small
   name→component registries make each addition one function + one registry line.
3. **Result interpretability** — three parallel views of every run: tidy long-format `results.csv`
   (Excel/pandas), plaintext per-phase `logs/` recording every prompt + response (tail-able live),
   and a tqdm/CLI progress view to watch a run — plus a standalone `analysis/` that only *reads*
   outputs, so comparing experiments never means sifting logs or re-running.
4. **Configurability** — one `ExperimentConfig` object holds every lever (no threading params
   through five functions); it is *also* the traceability record that gets serialized per run.

**Resumability was an earlier goal and is now an explicit non-goal** — batching made runs short
enough that re-running a bad run beats the machinery of resuming one. Runs are still **committed
to git by default** (bulky residuals and an opt-out `runs/_discard/` pile excepted) so an
experiment is never lost by accident — see
[§8](./02-architecture-roadmap.md#8-on-disk-layout-committed-by-default).

For running experiments back-to-back unattended, a **coordinator** (committed to the repo so every
teammate has it; runs a single experiment by default, and only drains a queue under `--queue`) chains
configs one at a time, with **git branches as the durable unit of experimental state** (code + configs
+ results per campaign) and per-phase commit/push for backup. It's sequential, single-node, and
file-controllable by a supervising LLM — see
[§10](./02-architecture-roadmap.md#10-batch-running-the-coordinator).

**Scope:** the existing `experiments/` directory is left untouched — the legacy notebook, `past_logs/`,
`best_vecs/`, and all prior logs stay as a frozen historical record. The new system lives in
`src/bias_steer/` (+ `configs/`, `analysis/`) and neither modifies nor depends on `experiments/`.

## Documents

| Doc | Purpose |
|-----|---------|
| [`01-feature-roadmap.md`](./01-feature-roadmap.md) | What the notebook already does today, and what needs re-implementation. A feature inventory + gap analysis. |
| [`02-architecture-roadmap.md`](./02-architecture-roadmap.md) | How the system should be re-architected: the four abstraction contracts (model/dataset/steering-method/judge), the config model, a source file map + data-flow trace, traceability + results design, on-disk layout, the batch coordinator, and a phased build order. |
| [`findings/`](./findings/) | Run outcomes and diagnoses — one file per investigation, newest first in its README. What we measured, what it means, and what has been **ruled out**. Design docs describe intent; these record what actually happened. |

## Status

- [x] Reverse-engineer the current notebook pipeline
- [x] Feature roadmap
- [x] Architecture roadmap
- [ ] Implementation (not started — pending review of these docs)
