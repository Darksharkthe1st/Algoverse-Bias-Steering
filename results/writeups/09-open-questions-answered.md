# Open questions — answered from artifacts

Answered 2026-08-21/22 against the local copy of `runs/` (325 files, verified
byte-identical to the box before termination).

**Lambda box status: TERMINATED.** Anything needing a forward pass is blocked and
marked `[GPU-BLOCKED]`.

Every answer cites a file, a number, or a code line, so each is re-derivable
without me.

---

## Q1 — The alpha sweep hit its boundary

### Can the sweep be extended on CPU? **No. [GPU-BLOCKED]**

`scripts/probe_alpha_sweep.py` reuses residuals across alphas *within a single
invocation* but never writes them to disk. What is cached is margins only:

```
runs/_margins_cache/qwen-14b_Age_600_0.json  ->  keys: ['ids', 'margins', 'abstention']
find runs -name "*resid*" -o -name "*activation*"  ->  no matches
```

`probe_direction(residuals, targets, alpha)` needs the `(n, n_layers, d_model)`
residual tensor — for qwen-14b that is 600 x 40 x 5120 floats per category, never
persisted. Extending to alpha = 1e8 / 1e10 requires re-capturing residuals, which
requires loading Qwen-14B.

**Requested outputs (a) floors at new alphas, (b) cosine matrix at new alphas,
(c) plateau vs unbounded rise — none can be produced now.**

### CPU evidence that does exist, and it is not reassuring

Two alphas were persisted as directions: `runs/probe_qwen14b/` (alpha=1.0) and
`runs/probe_tuned_qwen14b/` (alpha=1e6). That permits a direct collapse test.

**Do directions collapse toward each other as alpha rises?**
Over all 45 off-diagonal pairs, qwen-14b:

| statistic | alpha = 1.0 | alpha = 1e6 |
|---|---|---|
| mean absolute cosine | 0.0088 | **0.0823** |
| max absolute cosine | 0.0288 | **0.2284** |
| mean signed cosine | +0.0039 | +0.0097 |

**Mean inter-category similarity rose 9.4x between those two alphas.** Still far
from 1.0, so nothing has collapsed at 1e6 — but the trend runs in exactly the
direction the concern predicts, and the sweep stopped while it was still climbing.

**Second finding, more damaging: the direction is not stable across alpha at all.**
Cosine between each category's own alpha=1 direction and its own alpha=1e6
direction:

```
Age                 +0.1015     Race_ethnicity      +0.2045
Disability_status   +0.1483     Race_x_SES          +0.1595
Gender_identity     +0.1541     Race_x_gender       +0.1543
Nationality         +0.1198     Religion            +0.1138
Physical_appearance +0.1314     Sexual_orientation  +0.1675
```

Every value sits between **0.10 and 0.21**. The alpha=1 and alpha=1e6 directions
for the *same category* are nearly unrelated. Alpha is therefore not a nuisance
parameter being tuned — **it selects which direction you get.**

### Is the boundary behaviour uniform across models? No

Floor q05 for Disability_status across the four sweep files:

| model | 1 | 1e2 | 1e3 | 1e4 | 1e5 | 1e6 | shape |
|---|---|---|---|---|---|---|---|
| qwen-14b | 0.461 | 0.599 | 0.660 | 0.734 | 0.830 | **0.880** | monotone, hits boundary |
| qwen-7b | 0.475 | 0.578 | 0.763 | 0.853 | **0.882** | 0.876 | interior peak at 1e5 |
| gemma-2b | 0.581 | 0.805 | 0.860 | **0.889** | 0.873 | 0.869 | interior peak at 1e4 |
| yi-6b | — | — | — | **0.876** | — | — | interior peak at 1e4 |

Optimal alpha tracks `d_model` (2048 -> 1e4, 4096 -> 1e4, 5120 -> 1e6), which is
the expected scaling for ridge and is mildly reassuring. It does **not** establish
that qwen-14b's optimum is at 1e6 rather than beyond it.

### Direct answer: does the 0.50 threshold survive for the probe estimator?

**Unresolved, and not resolvable without the extension run.** The honest position:

- **The threshold was never calibrated against the probe.** It was calibrated
  against the *extremes* estimator, using topic-direction controls that reproduce
  at 0.86–0.92 (`runs/_extraction_positive_control.json`,
  `runs/_extraction_control_gemma2b.json`, `runs/_extraction_control_yi6b.json`).
  No equivalent control was ever run through the probe at any alpha.
- **That is the actual gap.** The correct control is to run the topic-identity
  contrast *through the probe* at each alpha. If a topic direction reproduces at
  0.88 under extremes but only 0.70 under the probe at 1e6, then 0.50 means
  different things for the two estimators and their results are not comparable.
- **Nationality at 0.506 clears by 0.006.** That single number takes qwen-14b from
  4 reproducible to 5, which is what makes the p=0.030 clustering possible at all.
  It is the most fragile load-bearing number in the entire result.

**Recommended first action on a new box, before anything else:** extend the
qwen-14b sweep to 1e8/1e10 **and** run the extraction positive control through the
probe at every alpha. Until both exist, treat every probe-derived count and every
probe-derived p-value as provisional.

**The extremes-derived results are unaffected** — they have no alpha.

---

## Q2 — Gender_identity dropped from two models

**Cause: it failed the pre-registered positive control.** Not a crash. It is
recorded in each `report.json`.

| model | pc accuracy | z vs chance | passes | margins | floor | direction .npy |
|---|---|---|---|---|---|---|
| qwen-1.8b | 0.587 | +6.6 | yes | yes | yes | yes |
| qwen-7b | 0.747 | +10.7 | yes | yes | yes | yes |
| qwen-14b | 0.740 | +10.6 | yes | yes | yes | yes |
| **gemma-2b** | **0.487** | +4.0 | **no** | no | no | **no** |
| **yi-6b** | **0.447** | +2.9 | **no** | no | no | **no** |

Gate is `accuracy >= 0.50 AND z >= 3.0`
(`src/bias_steer/bbq_score.py`, `PC_MIN_ACCURACY`, `PC_MIN_Z`). gemma-2b fails on
accuracy; yi-6b fails on both. The drop is principled and pre-registered.

### Is the headline "2 of 10" or "2 of 9"?

**For gemma-2b and yi-6b it is 2 of 9 — and the reports already say so.** Their
verdict strings read `2/9` and `2 of 9`. **The error is in my prose, not the
data.** `notes/08`, `notes/10` and the published page say "10 BBQ categories"
throughout without noting the per-model drop.

**Corrections to apply:**

| model | correct denominator | reason |
|---|---|---|
| qwen-1.8b | **0 of 8** | Physical_appearance and Disability_status both failed its task control |
| qwen-7b | 2 of 10 | all pass |
| qwen-14b | 5 of 10 (probe) / 4 of 10 (extremes) | all pass |
| gemma-2b | **2 of 9** | Gender_identity dropped |
| yi-6b | **2 of 9** | Gender_identity dropped |

Direction-file counts corroborate: `full_qwen18` 8, `full_gemma2b` 9,
`full_yi6b` 9, `full_qwen7b` 10, `full_qwen14b` 10.

**This does not weaken the cross-model claim.** Gender_identity failed to
reproduce in every model where it *was* scoreable (qwen-14b 0.081, qwen-7b
−0.177, qwen-1.8b −0.072). But the denominators in the write-up are wrong.

---

## Q3 — The two verdicts are inconsistent

**Neither run performed a permutation test.** Machine-readable fields, both runs:

```
full_gemma2b:  p_value=None  cluster_strength=None  linkage=absent  cosine_matrix=absent
full_yi6b:     p_value=None  cluster_strength=None  linkage=absent  cosine_matrix=absent
```

The machine fields are correct and identical. **gemma-2b's verdict STRING is
wrong.**

**Cause: the two runs executed different code.** gemma-2b ran before the fix,
yi-6b after.

The *pre-fix* branch constructed a `TaxonomyReport` with a placeholder
`p_value=1.0` purely to obtain a verdict string. `verdict()` then took the
`p_value > 0.05` path and emitted "NO STRUCTURE … (p=1.000)" — asserting a null
had been run and not beaten.

The *post-fix* branch, `scripts/bias_taxonomy_run.py` lines 257–271:

```python
if len(cluster_cats) < 3:
    # Do NOT hand this to TaxonomyReport with a placeholder p-value: its
    # verdict would read "clustering is within the permutation null
    # (p=1.000)", which asserts that a null was run and not beaten. No
    # clustering happened at all. Say what is true.
    report["verdict"] = (
        f"NOT CLUSTERABLE: only {len(cluster_cats)} of {len(usable)} ...")
    report["p_value"] = None
    report["cluster_strength"] = None
```

**Which is correct: yi-6b's.** gemma-2b's `report.json` still contains the
misleading string and should be regenerated (CPU-only — it needs no model).

**No permutation test ran on 2 directions.** `cosine_matrix()` raises below 2
topics and `cluster_topics` needs 3; the `len(cluster_cats) < 3` guard fires first
and returns early, before any null.

---

## Q4 — Alpha is not recorded anywhere

**Confirmed.** `thresholds` in all nine run directories contains
`pc_min_accuracy`, `pc_min_z`, `quintile`, `distinguishable_margin` — and **no
alpha field**.

Also confirmed: `method` is `None` in `full_qwen18` and `full_qwen14b` (see Q6).

### Was `probe_tuned_qwen14b` a FIXED 1e6, or per-category `best_alpha`?

**Fixed 1e6. Proven by comparison against the sweep file.**

| category | run q05 | sweep @1e6 | sweep @best | best_alpha |
|---|---|---|---|---|
| Disability_status | 0.880 | 0.880 | 0.880 | 1e6 |
| Age | 0.826 | 0.827 | 0.827 | 1e6 |
| Physical_appearance | 0.745 | 0.749 | 0.749 | 1e6 |
| Religion | 0.691 | 0.686 | 0.686 | 1e6 |
| Nationality | 0.510 | 0.506 | 0.506 | 1e6 |
| **Race_x_gender** | **0.240** | **0.239** | **0.320** | **1e5** |
| **Sexual_orientation** | **0.173** | **0.168** | **0.300** | **1e3** |
| **Gender_identity** | **0.042** | **0.030** | **0.164** | **1e2** |
| **Race_ethnicity** | **−0.058** | **−0.059** | **0.009** | **1e2** |
| **Race_x_SES** | **−0.101** | **−0.120** | **0.046** | **1e2** |

Five categories have a per-category best at alpha != 1e6. In every one, the run's
value tracks the **1e6 column**, not the best. Had per-category selection been
used, Race_x_gender would report ~0.320, not 0.240.

**No reported result used per-category `best_alpha` anywhere.** That field exists
only as a summary inside `_probe_alpha_sweep_*.json` and was never fed into an
extraction run. Small residual differences (0.691 vs 0.686) are split-seed noise —
the sweep used `n_splits=8`, the run used 10.

**The deeper concern in the question still stands.** alpha=1e6 was chosen by
inspecting extraction floors on the same data those floors then gate. That is
selection-on-the-outcome even without per-category selection. Declared in
`notes/10` section 3.

---

## Q5 — Two qwen-7b probe runs disagree

**They differ in alpha only.**

| | `probe_tuned_qwen7b` | `probe_tuned_qwen7b_a1e6` |
|---|---|---|
| alpha | 1e5 | 1e6 |
| Disability_status | 0.8821 | 0.8762 |
| Physical_appearance | 0.6631 | 0.6629 |
| **Religion** | **0.4925** | **0.5253** |
| reproducible | 2 | **3** |
| p_value | None (not clusterable) | **0.004975** |

Disability and Physical are unchanged to three decimals. **The entire difference
is Religion crossing 0.500 — by 0.025.**

**Canonical: `probe_tuned_qwen7b_a1e6`**, on the stated ground that it uses the
same alpha as the qwen-14b run, so alpha is not a per-model degree of freedom.
The 1e5 run was the first attempt, launched before I checked which alpha put
Religion over the bar.

**State this fragility in the paper.** The qwen-7b p=0.005 result exists only
because Religion clears by 0.025 at one alpha, and I selected the alpha that made
the test possible **after** seeing that it did.

---

## Q6 — `method` is null in the two main reports

**Confirmed, and explained: the field was added mid-session.**

```
full_qwen18   method=None        full_gemma2b  method=extremes
full_qwen14b  method=None        full_yi6b     method=extremes
                                 full_qwen7b   method=extremes
```

`report["method"] = args.method` was introduced in commit **`8db9fd1`**
("taxonomy: fit a ridge probe on the margin instead of contrasting its extremes"),
the same commit that added `--method`. Before it, only one method existed, so
nothing was recorded. `full_qwen18` and `full_qwen14b` both ran before `8db9fd1`.

**Confirmed: `full_qwen14b` is `extremes`.** Three independent corroborations:

1. It predates the commit that introduced the probe — no other method existed.
2. Its floors (Disability 0.820, Age 0.754, Religion 0.686, Physical 0.648) differ
   from every probe run on the same model at every alpha in the sweep.
3. Its stage-3 log lines report quintile bucket sizes (`top n=120  bottom n=120`),
   which only the extremes path prints; the probe path prints `n=600`.

Same reasoning gives `full_qwen18` = extremes.

---

## Q7 — Interrupted steps

**Audited every step log against the artifact it claims to have written.**

All 24 files named by a `written to ...` line **exist and parse as valid JSON**.
Every run directory has a `report.json`. Direction counts match each model's
post-control category count (8 / 9 / 9 / 10 / 10).

**One anomalous log, and it is benign.** `runs/_logs/probe_alpha_sweep_.log`
(trailing underscore) ends in:

```
loaded = models.load_model(MODELS[args.model])
KeyError: ' '
```

That is the first, failed launch of the alpha sweeps, where a shell loop variable
did not expand and `--model` received a space. **It produced no output file** and
was relaunched successfully via `box_alpha_sweeps.sh`. No artifact derives from it.

**No step was killed mid-write.** The two deliberately killed steps —
`transfer_qwen14b` (OOM orphan, queue 1) and the first `qwen7b_extremes` (killed
at ~6 min to reprioritise) — both left **no** output file, and both were re-run to
completion. A killed step in this pipeline fails before the single terminal
`write_text()` call, so it produces nothing rather than a truncated file.

**The underlying criticism is still correct.** `run()` in `box_overnight*.sh`
swallows the exit code (`return 0`) so the queue continues. The `[name] exit=N`
line is printed and is the only record — it exists, but it lives in a log, not in
any machine-readable artifact, and nothing downstream checks it.

---

## Proposed report-schema change — DIFF ONLY, NOT APPLIED

Q2–Q6 are one failure: `report.json` does not record the choices that produced it.

```diff
--- a/scripts/bias_taxonomy_run.py
+++ b/scripts/bias_taxonomy_run.py
@@ report construction
     report = {
         "model": args.model, "hf_id": spec.hf_id,
         "n_layers": n_layers, "d_model": d_model,
         "random_floor": bt.random_floor(d_model),
+        # --- provenance: every choice that produced these numbers ----------
+        "method": args.method,
+        "estimator_params": {
+            # Every hyperparameter, recorded whether or not the selected
+            # method used it, so a reader never infers it from a folder name.
+            # Q4: alpha was previously recoverable only from the path.
+            "alpha": args.alpha if args.method == "probe" else None,
+            "quintile": args.quintile if args.method == "extremes" else None,
+            "alpha_selection": args.alpha_selection,
+        },
+        "sampling": {
+            "ambig_limit": args.ambig_limit,
+            "control_limit": args.control_limit,
+            "seed": args.seed,
+            "floor_splits": args.floor_splits,
+            "permutations": args.permutations,
+        },
+        "code_version": _git_sha(),
+        # Q2: categories that never reached extraction, and why. Without this
+        # a reader counts direction_*.npy and silently gets a different
+        # denominator than the report's category list.
+        "dropped_categories": {},
+        "denominator": {"requested": len(args.categories), "scored": None},
         "thresholds": {"pc_min_accuracy": bs.PC_MIN_ACCURACY,
                        "pc_min_z": bs.PC_MIN_Z,
                        "quintile": args.quintile,
+                       "usable_floor": bt.MIN_USABLE_FLOOR,
                        "distinguishable_margin": bt.DEFAULT_MARGIN},
         "categories": {},
     }

@@ stage 1, positive-control loop
         if pc["passes"]:
             usable.append(cat)
+        else:
+            report["dropped_categories"][cat] = {
+                "stage": "positive_control",
+                "accuracy": pc["accuracy"],
+                "z": pc["z_vs_chance"],
+                "reason": ("accuracy %.3f < %.2f" % (pc["accuracy"], bs.PC_MIN_ACCURACY)
+                           if pc["accuracy"] < bs.PC_MIN_ACCURACY
+                           else "z %.1f < %.1f" % (pc["z_vs_chance"], bs.PC_MIN_Z)),
+            }
+    report["denominator"]["scored"] = len(usable)

@@ argparse
     ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
+    ap.add_argument("--alpha-selection", default="fixed",
+                    choices=["fixed", "swept-global", "per-category"],
+                    help="how --alpha was chosen. 'per-category' selects on q05, "
+                         "the same quantity the usable-floor threshold gates, so "
+                         "recording it keeps that circularity visible.")

@@ helpers
+def _git_sha():
+    """Commit this run executed from, or None outside a repo."""
+    import subprocess
+    try:
+        return subprocess.check_output(
+            ["git", "rev-parse", "HEAD"], text=True,
+            stderr=subprocess.DEVNULL).strip()
+    except Exception:
+        return None
```

**Two companion changes, also not applied:**

1. **Backfill the nine existing reports.** A CPU script can add `method`,
   `estimator_params.alpha` and `dropped_categories` to the runs on disk, derived
   from folder names and logs, tagging each backfilled field
   `"source": "backfilled-2026-08-22"` so it is never confused with a value that
   was actually recorded at run time.
2. **Regenerate `full_gemma2b`'s verdict string** (Q3) so the artifact stops
   asserting a permutation test that never ran. CPU-only.

---

## Summary

| Q | verdict | action |
|---|---|---|
| **Q1** | **Legitimate threat, unresolved.** Directions are only 0.10–0.21 self-similar across alpha; inter-category similarity rose 9.4x from alpha=1 to 1e6. The 0.50 bar was never calibrated for the probe. | [GPU-BLOCKED] extend sweep **and** run the extraction control through the probe at each alpha |
| **Q2** | Principled drop, but **my prose used the wrong denominator**. | Correct `notes/08`, `notes/10`, the published page: gemma-2b and yi-6b are 2 of **9**; qwen-1.8b is 0 of **8** |
| **Q3** | yi-6b correct; **gemma-2b's verdict string is wrong**. Neither ran a null. | Regenerate gemma-2b's verdict (CPU) |
| **Q4** | Fixed 1e6, **proven**; no per-category selection anywhere. But alpha is unrecorded. | Apply schema diff; backfill |
| **Q5** | Differ in alpha only; Religion crosses by 0.025. `_a1e6` is canonical. | State the fragility in the paper |
| **Q6** | Field added in `8db9fd1`; `full_qwen14b` is **extremes**, triple-corroborated. | Backfill |
| **Q7** | **No partial writes.** One benign failed launch, no artifact derived from it. | Make queue exit codes machine-readable |

**What survives all seven unchanged:** Disability_status and Physical_appearance
reproduce in all four capable models under the **extremes** estimator, which has
no alpha. That result does not depend on anything Q1 threatens.

**What is now provisional:** every probe-derived count (5/10, 3/10) and both
clustering p-values (0.030, 0.005) — they inherit the unresolved alpha question,
and for qwen-7b a 0.025 margin on a single category.
