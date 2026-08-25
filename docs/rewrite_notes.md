# Rewrite notes

Design ideas and improvements noticed while reading the codebase. **Not
implemented** — a backlog to revisit, not a spec. Each entry: the idea, why,
the shape, and any caveats/process notes.

---

## 1. Per-run `examples.csv` (snapshot the examples actually used)

**Status: done (2026-08-25).** `metrics.write_examples_csv(path, examples, *, dataset)`
writes one row per `Example` (`example_id, dataset, prompt, category, metadata_json`)
and delegates to a now-generic `write_csv(path, rows, columns)` (both artifacts route
through it; `EXAMPLE_COLUMNS`/`RESULT_COLUMNS` are passed explicitly). Called from
`_run_one` beside the `results.csv` write, from the frozen `train + test` subset — one
copy per model run folder. Arch roadmap §5.2 WRITE block + §8 tree updated to register
the artifact. `test_phase2::test_run_end_to_end_produces_all_artifacts` asserts the file
exists, has the five columns, joins back to `results.csv` on `example_id`, and
round-trips `metadata` through the JSON column.

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

**Done (2026-08-25):** `artifacts.save_residuals` and `artifacts.save_vector` now take
`n_layers`/`d_model` (threaded from `loaded.model.cfg` at the `_run_one` call sites) and
assert the *exact* `(n_layers, d_model)` shape against that ground truth — the passed
check subsumes an infer-and-compare backstop, and matches §6's "assert against
`(n_layers, d_model)`" wording. `save_residuals`' loop runs before `import torch`, so
inputs are validated ahead of the heavy import (and the check is testable torch-free).
`test_phase2::test_save_boundary_asserts_reject_wrong_shape` exercises both rejections
with a shape-only stub; the four wiring fakes (phase2/3/4) gained `cfg.d_model` and
`**kw` on their save lambdas.

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

**Status: deferred (2026-08-25).** Decision held until [[#9]] is settled: #9's
running-sums option (3) would remove the per-example residuals artifact entirely,
which is mutually exclusive with building a loader for it. Revisit the loader-vs-drop
fork once the VRAM/memory direction is chosen — don't commit either way first.

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

**Status: done (2026-08-25).** `_resolve` now raises `FileNotFoundError` when the
resolved path doesn't exist, naming the input, the resolved absolute path, and the repo
root, and calling out the doubled-repo-name mistake explicitly. No fallback heuristic
(per "explicitly not doing"). All current callers are read-loaders that `open()` the
result, so requiring existence at resolve time is safe. `test_phase1::
test_resolve_valid_path_and_fails_loud_on_missing` covers both the valid resolve and the
bare parent-relative doubling case. (Wiring fakes register lambda loaders, so their
`path="ignored"` never routes through `_resolve` — untouched.)

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

**Status: done (2026-08-25).** The 3 wrapped loader bodies (`plain`, `crows`,
`hidden_bias`) are inlined into `datasets.py` (byte-for-byte: plain keeps blanks; crows
uses `newline=''` + flatten + non-empty filter; hidden_bias uses NO `newline=''` and an
f-string identical to legacy `get_question`). Every `from src.data import ...` is gone
from the package (verified: `bias_steer` imports no legacy module). `src/data.py` left
untouched. Three FROZEN-LEGACY EQUIVALENCE ANCHOR tests in `test_phase1` prove each
inline body == legacy output on synthetic temp files; the BBQ anchor comment is
relabeled. **Doc rule** added to arch §3.3 (broadened per feedback): `bias_steer` must
not import from the legacy notebook-owned modules `src/data.py`, `src/main.py`,
`src/stereoset-dataloader.py` — with `src/utils.py` (stdlib-only shared helpers) the one
sanctioned `src.*` dependency; `tests/` may import `src.data` only as an equivalence
anchor. (`docs/SOURCES_OF_TRUTH.md` doesn't exist, so the rule landed in the arch roadmap,
which already owned "become the bodies"; the stale "wraps src/data.py" tree comment fixed.)

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

**Status: done (2026-08-25).** `sample()` now ends with `random.Random(spec.seed).shuffle(out)`
and returns a de-blocked subset; the caller-side shuffle at `experiment.py:87` (and the
now-unused `import random`) are gone. **The caveat below was avoided:** rather than reusing
the advanced `rng` (which would change the permutation), a *fresh* `Random(spec.seed)`
reproduces the exact permutation the old caller-side shuffle produced — verified bit-for-bit
against the old blocked-then-shuffle pipeline — so historical train/test splits still
reproduce exactly. A guard copies the list when no filter/group/limit ran, so the caller's
input list is never mutated in place. `test_phase1::test_sample_returns_de_blocked_order`
asserts the output is interleaved (many category boundaries) and each half is balanced.

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

---

## 7. Log judge retries (transient failures currently vanish)

**Observation.** `judge._call_with_retry` retries a chat completion up to
`_MAX_RETRIES` times with exponential backoff, but a retry that eventually succeeds
leaves **no trace** — the function returns the good result and the failed attempts
are silently discarded. Only the terminal case (all retries exhausted) surfaces,
via `raise last_error`. So a clean run and one that retried 3× on every item look
identical in the outputs; retry cost (tokens/money/latency) and the rate-limit /
flakiness signal are invisible.

**Why.** Arch §7.2 defines `run.log` as the event stream for "counts, timings,
**errors**." Silent retries are errors that never reach it. Retries are also the
earliest warning of API degradation — losing them means you only find out at the
terminal failure, when the pattern (one bad item vs. the whole batch degrading) is
already gone.

**Shape.**
- Emit a `logging.warning` on each retry (attempt number + error type) and a
  per-phase summary count (e.g. "judge: 9 retries across 4 items, 0 terminal
  failures"). Stdlib `logging` is the low-friction start — it doesn't change the
  judge's `(responses, examples, spec) -> list[str]` contract.
- Cleaner but heavier: thread the `RunLogger` into the judge so retry events land
  in `run.log` directly (changes the contract — probably not worth it).

**Related gaps (same function, distinct from logging).**
- `except Exception` retries *any* error, including non-transient ones (a 400 / auth
  error is pointlessly retried 4× with backoff). Consider narrowing to transient
  API/network errors and failing fast otherwise.
- A terminal failure raises inside `asyncio.gather`, whose default propagates the
  first exception — so **one item exhausting retries kills the whole judge phase**
  instead of that item falling back to `UNMATCHED`. Consider
  `gather(..., return_exceptions=True)` + per-item fallback so one dead item can't
  sink the run.

---

## 8. Determinism test: same config + same code → identical results

**Idea.** Since the whole point of the seeds in this pipeline is reproducibility,
add a test that runs the pipeline **twice under the exact same config and code** and
asserts the outputs are identical — guarding against *accidental* randomness (an
unseeded `random`/`torch` call, dict-ordering leakage, etc.) creeping in.

**Why.** Seeds are threaded through `sample()` and the train/test shuffle
(`experiment.py:87`) and recorded in the manifest specifically so a run is
reproducible. Nothing currently *verifies* that promise, so a future edit could
introduce nondeterminism silently and no test would catch it.

**Shape.**
- **Hermetic version (belongs in the test suite):** with a stub model + stub judge
  (as the phase tests already do), run `experiment.run` twice with one config and
  assert bit-identical `results.csv`, steering vector, and the frozen example
  subset + train/test split. Stubs remove model/judge noise, so any diff is
  code-introduced randomness — exactly what we want to catch. Cheap, CI-able.
- Assert on the deterministic artifacts (sampled ids, split, ordering, and the
  built vector given fixed residuals), not on wall-clock/timestamps.

**Caveat.** A *real* end-to-end double-run is not bit-reproducible for reasons
outside our code: the GPT judge is nondeterministic (see needed-experiments §0.2),
and CUDA float ops can differ run-to-run. So the full-stack version (see
needed-experiments §12) must compare only the stable parts (sampled subset,
train/test partition, and the vector given identical residuals), while the hermetic
stubbed test is what actually pins *our* determinism.

---

## 9. Move captured residuals off the GPU (`.cpu()`), not to disk

**Observation.** `_run_one` accumulates `resids_by_label` across the whole train
phase (`experiment.py:115-121`). The concern was OOM from holding "two arrays of
many large residuals," but the footprint is smaller than it looks: `capture_mean`
already collapses the sequence dim (`.mean(dim=1)`), so what's kept per example is
just `(n_layers, d_model)` fp16 ≈ **256 KB (Llama-8B) / 400 KB (Qwen-14B)** — tens
to a few hundred MB even at `n_train`=1000. In *system RAM* that's a non-issue.

**The real problem is *where* it lives: VRAM.** `capture` does `.detach().clone()`
but never `.cpu()`, so residuals accumulate on the model's device, competing with the
model weights and generation activations for scarce VRAM.

**Fix (cheapest first).**
1. **`.cpu()` the captured residual** — do it once on the stacked `(n_layers,
   d_model)` tensor (`torch.stack(per_layer).cpu()`), not per layer. Cost is a
   ~256–400 KB device→host copy per example (~10–80 µs over PCIe gen4), utterly
   dwarfed by the per-example generation + `run_with_cache` pass (tens–hundreds of
   ms). Effectively free; removes the growing VRAM term entirely.
   - **Required companion change:** the built vector then lands on CPU, so
     `apply_resid_pre_add`'s hook (`value += scaled * vec`, `value` on GPU) would hit
     a **device-mismatch error**. Move the vector back before the test phase:
     `vector = vector.to(loaded.device)` after `build` (one ~256 KB transfer). Bonus:
     safetensors saves cleanly from CPU.
2. **Stream to disk** — only if CPU RAM also gets tight at very large `n_train` ×
   large models. More machinery; premature until (1) proves insufficient.
3. **Running sums instead of all residuals** — for `mean_diff` you only need per-label
   running sum + count to get `mean(pos) − mean(neg)`, which is `O(n_layers × d_model)`
   regardless of `n_train`. **Tradeoff:** sacrifices the per-example
   `residuals.safetensors` artifact and breaks methods that need all residuals
   (`capture_last`, future probes) — conflicts with the "make residuals loadable"
   idea in [[#3]]. Method-specific optimization, not a default.

**Sizing (A100 40 GB SXM4).** 7–8B models (~14–16 GB weights) have ~24 GB headroom —
residuals-in-VRAM won't OOM even without the fix. 14B models (~28 GB weights) leave
~12 GB, so `.cpu()` clearly earns its keep. `.cpu()`-ing residuals is basically free
insurance either way.

**Not fixed by this:** the bigger *transient* VRAM spike is the per-batch `caches`
from `generate_with_cache` (all hooks' activations at full seq length for the whole
batch, freed each iteration). That scales with `batch_size` + seq length, not
`n_train` — so a train-phase OOM on 14B is addressed by lowering `batch_size`,
whereas `.cpu()` addresses the slow accumulation across the phase. Different knobs.

---

## 10. Split `_run_one` into per-phase helpers (readability)

**Idea.** `_run_one` (`experiment.py:102-183`) is one ~80-line function doing four
things in sequence: setup (load model) → TRAIN (capture → build → save vector) →
TEST (initial + steered gen → judge) → FINALIZE (metrics + results.csv + summary +
index). Split the phases into `_train_one`, `_test_one`, and a finalize helper, with
`_run_one` reduced to a short conductor.

**Data flow (why the seams fall where they do).** It's a strict linear pipeline with
two hand-offs: `vector` (TRAIN→TEST) and `results: list[Result]` (TEST→FINALIZE).
Everything else is shared setup state (`loaded`, `log`, `handle`, `config`,
`method`, `judge_fn`, `contrast`, `backend`, `n_layers`). Natural signatures:
- `_train_one(ctx, train) -> vector` — owns capture loop, `build`, `save_vector` /
  `save_residuals`, and `on_phase("vector")`.
- `_test_one(ctx, test, vector) -> list[Result]` — owns the eval loop.
- `_finalize_one(ctx, results, train, test) -> RunResult` — owns tidy_rows/CSV,
  counts/quality, summary.md, index row, and `on_phase("eval")`.

**Pros.**
- **Readability / one concern per function** — each helper is ~15–30 lines named by
  intent; `_run_one` becomes a ~15-line conductor.
- **Testability** — unit-test `_test_one` against a fixed vector, or `_finalize_one`
  against a fixed `results` list, without running TRAIN (today the phase tests can
  only exercise the whole `run()` via `Backend` fakes).
- **Enables "re-eval without retrain"** — a load-existing-vector-then-eval path (ties
  to [[#3]] on loading residuals/vectors) becomes calling `_test_one` with a loaded
  vector, instead of duplicating the eval loop.
- **Mirrors structure that already exists** — the `on_phase("vector")` /
  `on_phase("eval")` boundaries already mark these seams.

**Cons.**
- **Parameter threading** — the helpers share ~8–11 locals; naive splitting turns one
  long function into three long *signatures*. That's the exact smell the arch roadmap
  called out in the notebook (long param lists). Mitigation: a small `_RunCtx`
  dataclass (handle, log, loaded, config, method, judge_fn, contrast, backend,
  n_layers) passed to each — but that's added machinery.
- **Implied independence that isn't there** — the phases are sequential with hard
  data deps (TEST needs a built, device-correct vector; FINALIZE needs TEST's
  results). Functions can read as more modular/reorderable than they are.
- **Distributed side effects** — the persistence + `on_phase` boundaries move out of
  one visible place into the helpers (arguably fine — co-locating a save with its
  phase — but it's a change in where effects live).
- **Churn for a pure-readability gain** — `_run_one` is already sectioned with
  `# --- TRAIN/TEST ---` banners and reads linearly top-to-bottom; the marginal gain
  may be modest relative to the diff.

**Recommended shape.** Do it **only if paired with a `_RunCtx`** — otherwise the
signature bloat cancels the readability win. Rename the third helper `_finalize_one`,
**not** `_log_one`: there's already incremental plaintext logging via `RunLogger`
(`log.train`/`log.eval`/`log.event`) interleaved *inside* TRAIN/TEST that must stay
put; the final phase is metrics + persistence, not logging. Lighter-touch
alternative if the goal is purely visual: extract just the two dense loop bodies
(`_capture_residuals(...) -> resids_by_label`, `_eval_examples(...) -> results`),
which are the visually heavy parts, and leave the rest inline — most of the
readability for far less param threading.

---

## 11. Show queue position per run (banner, not a per-line prefix)

**Idea.** When the coordinator drains a queue, make it obvious which experiment
you're on ("1 of 5"). The tempting approach — prefix every echoed stdout line in
`_subprocess_runner` (`coordinator.py:89-90`) with `[001]` — is the wrong one
(see "why not per-line" below). Instead surface the queue position at the *run*
level.

**Shape (option 1 — recommended).**
- Print a one-time banner before launching each queue item, e.g.
  `=== [1/5] config foo.py ===`.
- And/or thread the queue index into the run's header and the tqdm `desc` (already
  `f"{model_key} train"` in `experiment.py:116/133` → `f"[1/5] {model_key} train"`),
  e.g. passed via an env var or CLI arg to the child.
- Rationale: queue position is a *per-run* fact, not a *per-line* one — one banner
  answers "which experiment am I on" without touching every line or fighting tqdm.

**Why NOT a per-line prefix.**
- **tqdm interaction.** tqdm draws with carriage returns (`\r`, no newline) and
  writes to stderr, which is merged into the pipe (`stderr=subprocess.STDOUT`). The
  pipe is `text=True`, so universal-newline translation turns every `\r` update into
  its own `\n`-terminated line — the animated bar already renders as a vertical
  *waterfall* under the coordinator. A per-line prefix then stamps every micro-update
  (`[001] bar 42%` ×200 per bar) — very noisy.
- If a per-line prefix were ever wanted anyway (option 2, not chosen): read the pipe
  in **binary** (`text=False`) and prefix only after a real `\n`, passing `\r` runs
  through untouched, to preserve the in-place bar.

**Gotcha regardless of approach.** If any prefixing is added to the echo, apply it to
the *echoed copy only* and keep `_PHASE_RE.match` on the **original** line —
`line.strip()` removes whitespace, not a `[001] ` prefix, so matching a prefixed line
would silently break phase-sentinel detection (`on_phase`).

---

## 12. Drop the non-ASCII middle dot from coordinator commit messages

**Status: done (2026-08-25).** The coordinator's `add_commit` messages used a
Unicode middle dot (`·`, U+00B7): `f"{config} · {phase} ({run_id})"` and
`f"{config} · finalize"` (`coordinator.py:172,185`). Replaced with a plain ASCII
hyphen (` - `).

**Why.** A non-ASCII separator in commit subjects is a small portability/tooling
liability — encoding surprises in terminals, hooks, log parsers, or CI that don't
assume UTF-8 — for no readability benefit over `-`. No behavior change: the messages
are cosmetic, no test asserts on the separator (phase-4 tests check commit
*counts*/behavior, not text), and phase4 still passes 7/7.
