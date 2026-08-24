# Rewrite notes

Design ideas and improvements noticed while reading the codebase. **Not
implemented** — a backlog to revisit, not a spec. Each entry: the idea, why,
the shape, and any caveats/process notes.

---

## 1. Per-run `examples.csv` (snapshot the examples actually used)

**Idea.** Alongside each run's `results.csv`, write an `examples.csv` containing
only the examples that run actually used (the frozen sampled subset), so a run
folder holds both the inputs and the outputs of the experiment together.

**Why.**
- **Self-contained run folders.** Today, recovering an example's prompt text (or
  any metadata not denormalized into `results.csv`) requires reloading the
  dataset → replaying `sample(seed)` → confirming the code SHA matches → rebuilding
  `{id: Example}`. An `examples.csv` collapses that to opening one file.
- **Fixes the id-drift fragility.** Positional ids (`plain-{i}`, `crows-{i}`,
  `hidden-{i}`) are only stable while the source file's line order is unchanged;
  edit the dataset and `plain-42` silently re-points, with nothing detecting the
  drift. Snapshotting the used examples freezes the ground truth.
- **Better than the unbuilt "dataset snapshot hash."** The arch roadmap (§6)
  describes a resolved-dataset hash that was never implemented. Keeping the actual
  rows is strictly stronger than a hash: a hash only *detects* drift; a snapshot
  *survives* it.
- **Restores normalization.** `examples.csv` = parent table (one row per
  `example_id`); `results.csv` = denormalized child (one row per example×condition,
  with `category` copied in for cheap groupbys). Clean relational shape.

**Shape.**
- One row per `Example`, keyed by `example_id`:
  `example_id, dataset, prompt, category, metadata_json`.
  Keep `category` as its own column (matches `results.csv`); JSON-encode the full
  `metadata` dict into one column so nested fields (e.g. BBQ's `answers` list)
  survive losslessly instead of exploding into ragged columns.
- Write once per run in `experiment.py`, next to the `results.csv` write, from the
  same frozen `examples` list. The subset is fixed before the model loop
  (data-flow trace §5.2), so it's shared across models — one file per run folder.
- Join stays on `example_id`: `analysis/` can `merge(results, examples,
  on="example_id")` when it wants prompts, and ignore it otherwise.
- Likely implementation: add `write_examples_csv()` to `metrics.py` (beside
  `write_csv`), call it from `experiment.py`'s write step.

**Caveat.** Partially overlaps `logs/train.txt` / `logs/eval.txt`, which already
record every prompt verbatim. So this isn't "otherwise the prompts are lost" —
it's "otherwise the prompts aren't *queryable*." That's still worth it (the value
is the tidy, joinable form), but frame it honestly.

**Process note.** The project is FROZEN (`CLAUDE.md`), and the on-disk layout is an
owned fact (arch roadmap §5/§8, per `docs/SOURCES_OF_TRUTH.md`). This is additive
engineering — no metric/judge/framing/claim changes, so not a scope reopening — but
adding an output artifact should update the arch roadmap's on-disk-layout section
(§5.2 WRITE block + §8) in the same change, per "link, don't restate / a new
artifact claims its row."

---

## 2. Assert residual tensor shape before stacking

**Idea.** In `artifacts.save_residuals` (and/or `steering.capture`), assert each
per-example residual is `(n_layers, d_model)` before `torch.stack`, instead of
trusting the shape.

**Why.** `CLAUDE.md` §6 records that this project has been bitten by *silent* shape
bugs — e.g. a 1-D hidden-width tensor where `vector[layer]` yields a scalar that
broadcasts across the residual width (a DC offset, not a direction), which
invalidated a whole cross-application. It explicitly asks: "Before shipping ANY
intervention, assert tensor shape against `(n_layers, d_model)` — this class of bug
is silent." `save_residuals` currently does no such check; a malformed `capture`
would stack into a wrong-shaped tensor and fail (or worse, succeed) far from the
cause.

**Shape.**
- Cheap guard before the stack, e.g. `assert items[0].shape == (n_layers, d_model)`
  (needs `n_layers`/`d_model` threaded in, or infer + assert consistency across
  items: all items share one shape and `ndim == 2`).
- Better placed at the *source* in `steering.capture`, so the assertion fires the
  moment a bad residual is produced, not at save time.
- Pair with the same assertion on the built vector in `save_vector` /
  `steering.build`, closing the loop end-to-end.

**Caveat.** Keep asserts cheap and shape-only; don't turn this into heavy runtime
validation on the hot path.

---

## 3. No load path for residuals — they're currently write-only

**Observation.** `residuals.safetensors` is *written* (`experiment.py:128` →
`artifacts.save_residuals`) but never *read back*: there is a `load_vector` but no
`load_residuals`, and nothing in `src/`, `analysis/`, or `tests/` loads the file.
`resids_by_label` only ever exists in memory during a run, consumed immediately by
`steering.build`. So the artifact is produced but unused.

**Idea.** Add a `load_residuals(path) -> dict[label, tensor]` (mirror of
`load_vector`) and give `analysis/` a reason to use it — e.g. re-deriving a vector
with a different contrast/label set without re-running generation, inspecting
per-layer separation between verdict clusters, PCA/probe experiments on the raw
activations, or sanity-checking `build` offline.

**Why.** Capturing residuals is the expensive part of a run (a forward pass per
train example). Persisting them but never loading them means that cost can't be
reused — any re-analysis re-runs the model. A load path turns the saved residuals
into a reusable asset and makes `build` auditable offline.

**Caveats.**
- Residuals are **git-ignored** (`runs/**/residuals.safetensors`) and bulky, so any
  analysis that loads them only works on the machine that produced them (or before
  cleanup) — they are not a portable, committed artifact like the steering vector.
  Document that constraint if analysis starts depending on them.
- If they're going to stay write-only, the alternative is to *stop writing them by
  default* (a config flag) and save the disk/time — decide whether the artifact
  earns its keep before building a loader for it.
