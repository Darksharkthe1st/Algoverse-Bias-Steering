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

**Status: mostly done (2026-08-24).** `steering.py` is now guarded end to end:
- `capture_mean` / `capture_last`: each cached `resid_pre` is `(1, seq, d_model)`
  with batch == 1 (catches the silent batched-cache / `.squeeze(0)` regression), and
  the stacked output is `(n_layers, d_model)`.
- `build_mean_difference`: both contrast labels have captured residuals (clear error
  listing available labels instead of a bare `KeyError`), and the pos/neg means are
  matching 2-D `(n_layers, d_model)`.
- `apply_resid_pre_add`: **the vector is `(n_layers, d_model)`** before hooks are
  built — the exact guard `CLAUDE.md` §6 mandates against the retracted 1-D → scalar
  DC-offset bug (`docs/REVIVAL_AUDIT.md`).
- `resid_pre_hook_names`: `n_layers >= 1`.
- Test updated: `tests/test_phase1.py::test_apply_builds_hooks_structure_without_torch`
  now uses a torch-free `_StubVector` (with `.ndim`/`.shape`) + `_StubCfg.d_model`, so
  the structural test still runs without torch and exercises the new shape guard.

**Still open:** `artifacts.save_residuals` (assert each per-example residual is
`(n_layers, d_model)` before `torch.stack`) and `artifacts.save_vector` (assert the
vector shape at the persistence boundary too).

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

---

## 4. `datasets._resolve`: fail loud on a non-existent resolved path

**Observation.** `_resolve` has a binary model — absolute → use as-is; anything
else → join under repo root. But "not absolute" ≠ "relative to root." A path
anchored at the root's *parent* (or a higher ancestor) — exactly what you get when
the string starts with the repo dir name (`Algoverse-Bias-Steering/datasets/...`,
as when copied from a file browser or tab-completed from `GitHub/`) — is blindly
joined under root, producing a doubled, non-existent
`<root>/Algoverse-Bias-Steering/datasets/...`. It's the most *natural* wrong input,
and there's no check at resolve time: the bad path surfaces much later as an opaque
`FileNotFoundError` deep in a loader's `open()`, showing the confusing doubled path
with no hint that resolution was the culprit. (An explicit `../Algoverse-.../...`
with a leading `..` does resolve correctly — the gap is the *bare* parent-relative
path.)

**Idea.** Validate in `_resolve`: if the resolved path doesn't exist, raise a clear
error naming both the attempted absolute path and the repo root, so the
doubled-name mistake is obvious immediately instead of 40 lines into a loader.

**Why.** The ambiguity is fundamental — a bare relative string like `foo/bar` is
genuinely indistinguishable between "relative to root" and "relative to root's
parent," so no rule can disambiguate from the string alone. The honest fix isn't
smarter resolution, it's **failing loud and early** at the resolution boundary.

**Shape.**
- After computing the resolved path, `if not resolved.exists(): raise
  FileNotFoundError(...)` with a message showing the input, the resolved absolute
  path, and the repo root.
- Keep it in `_resolve` so every loader benefits (they all route paths through it).

**Explicitly not doing:** no fallback heuristic (e.g. retry `root.parent/path` or
strip a leading repo-name segment). It would rescue the natural case but can mask
real typos. Fail loud and proud instead.

---

## 5. Decouple the new package from legacy `src/data.py`

**Observation.** `src/bias_steer/datasets.py` reaches out of its own package into
the frozen legacy module `src/data.py`, importing 3 loaders at call time:
`load_plain_dataset`, `load_crows_pairs`, `load_hidden_bias_dataset`. The other two
loaders are already self-contained — `load_bbq` reimplements the prompt inline (its
docstring only *references* the legacy version) and `load_stereoset` reads JSON
directly. `src/data.py` is retained because the legacy consumers still import it:
`examples/data.ipynb`, `experiments/farhan-experimentation.ipynb`, and (as a
deliberate equivalence check) `tests/test_phase1.py`. Two of its functions
(`load_custom_dataset`, `load_bbq_dataset`) are legacy-only, unused by the package.

**Why.** The new package isn't self-contained — it has a lateral/upward dependency
on frozen legacy code. That makes `src/data.py` effectively un-editable (the
notebook depends on it), so the new pipeline is stuck inheriting legacy quirks it
can't safely fix. Clean separation lets the two layers diverge safely in both
directions. Notably, arch roadmap §3.3 says the legacy loaders should "*become the
bodies* of these functions" — i.e. copy the code in, not `import` and call. So the
current import-wrapping is the *deviation*; inlining **restores** the documented
design rather than reopening it (cleanup, not a scope change).

**Shape.**
- Inline the 3 wrapped loader bodies directly into `datasets.py` (~40 lines of
  small CSV/txt parsing); delete every `from src.data import ...` in the package.
- Leave `src/data.py` byte-for-byte as-is — purely legacy, owned by the two
  notebooks. Do not modify it (keeps `experiments/` untouched, arch §12).
- Keep the `test_phase1` legacy import but re-label it explicitly as a
  *frozen-legacy equivalence anchor* (its purpose is proving new inline BBQ == old
  output).
- Optional: document the rule "`bias_steer` must not import from `src.data`" in the
  arch roadmap / `docs/SOURCES_OF_TRUTH.md` so the coupling doesn't creep back.

**Caveat.** Duplicates parsing logic across the old and new loaders. Acceptable
because legacy is frozen: the duplication is static, and the inlined copy becomes
the sole source of truth going forward. That divergence is the goal, not a regret.

---

## 6. Shuffle inside `sample()` after stratifying

**Observation.** `sample`'s `per_group` stage concatenates groups in sorted order
(`for g in sorted(groups, ...): picked.extend(items[:n])`), so the returned list is
**blocked by group** (all of category A, then all of B, …). The `limit` stage
preserves that order too (it selects random indices but rebuilds in original
order). A positional train/test slice over a blocked list would be badly
imbalanced. Today this is patched *by the caller* — `experiment.py:87` does
`random.Random(config.sample.seed).shuffle(examples)` right after `sample()`,
before the `examples[:n_train]` split — not by `sample()` itself.

**Idea.** Do the final shuffle *inside* `sample()`, so it returns a de-blocked
(interleaved) representative subset, and the caller no longer has to know to shuffle
before slicing.

**Why.** The current split of responsibility is a leaky contract: `sample()`
creates the ordering problem but leaves the fix to whoever calls it. Any other
consumer (analysis, tests, a future caller) that slices `sample()`'s output without
knowing to shuffle first hits a silent category-imbalance footgun. Encapsulating
makes the contract "returns a randomly-ordered representative sample," full stop.

**Shape.**
- Add `rng.shuffle(out)` at the end of `sample()`, reusing the existing seeded
  `rng` (still deterministic).
- Then drop the now-redundant `random.Random(...).shuffle(...)` at
  `experiment.py:87` (or keep it as defensive no-harm double-shuffle).

**Caveat.** This changes the exact ordering — and therefore the exact train/test
partition — versus today, because the in-`sample` shuffle continues the existing
`rng` stream (already advanced by per_group/limit) rather than a fresh
`random.Random(seed)`. Reproducibility going forward is preserved, but historical
splits won't reproduce bit-for-bit. Minor, but worth flagging given the repo's
emphasis on reproducible subsets (seed recorded in the manifest).
