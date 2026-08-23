# Overnight status report — night of 2026-08-20 → 21

**Facts only. Interpretation lives in `notes/08-results-2026-08-20.md`.**
Written incrementally as the queue ran; the "queue status" section at the end
records where it had got to.

---

## 1. Models run this session, in order, and why

| # | model | why it was run |
|---|---|---|
| 1 | `qwen-1.8b` | first choice — fast, ungated, most historical comparators in the repo |
| 2 | `qwen-7b` | run early as a fallback when 1.8B's generation-based choices looked like coin flips |
| 3 | `gemma-2b` | different family; highest baseline commit rate in the team's 2025 data |
| 4 | `qwen-14b` | scale test after 1.8B produced no reproducible directions |
| 5 | `yi-6b` | third architecture family, for replication |
| 6 | `qwen-7b` (again) | queued as an extra scale point; **killed before completion**, see §8 |
| 7 | `llama3-8b` | queued last; gated on HF, may not have run |

**Why the switches happened**

- 1.8B → 7B: under generation-based scoring, qwen-1.8b's choice of person changed
  about half the time when the two options were swapped in the prompt
  (person-consistency 48–68%). qwen-7b was tried as a more capable alternative
  and abstained on 70–87% of items, collapsing the named-answer buckets.
- generation → likelihood scoring: neither model's *decoding* was stable, so the
  method changed rather than the model.
- likelihood-with-option-list → likelihood-without: that design failed identically
  on qwen-1.8b and gemma-2b (order-robustness r = 0.14–0.34) because option text
  in the prompt was being copied. Removing the option list fixed it.
- 1.8B → 14B: with the corrected method, qwen-1.8b produced 0/8 reproducible
  directions and the floor did not improve with n. 14B was run to test whether
  that was scale.
- 14B → gemma-2b, yi-6b: replication across families.

---

## 2. What was actually executed per model

| model | task control | margins | extraction | floor | transfer | notes |
|---|---|---|---|---|---|---|
| qwen-1.8b | ✅ 8/10 pass | ✅ 8 cats | ✅ extremes + probe | ✅ | ❌ not run | base rates + choice diagnostics also run |
| qwen-7b | ✅ (base-rate era) | ❌ | ❌ | ❌ | ❌ | only base rates; the taxonomy run was killed |
| gemma-2b | ✅ 9/10 pass | ✅ 9 cats | ✅ extremes | ✅ | ✅ | extraction control also run |
| qwen-14b | ✅ 10/10 pass | ✅ 10 cats | ✅ extremes + probe | ✅ | ✅ + sweep | H2 dose-response also run |
| yi-6b | ✅ | ✅ 9 cats | ✅ extremes | ✅ | ✅ | extraction control also run |
| llama3-8b | — | — | — | — | — | see queue status |

Categories are the ten BBQ files under `datasets/BBQ_Prompt_Sets/`. Which ones
survived each model's task control differs and is recorded in each run's
`report.json`.

---

## 3. Base-rate tables

Base rates were produced only during the **generation-based** phase, which was
later superseded. They are counts over ambiguous items, per category.

See `runs/_base_rates_qwen18_v2.json` and `runs/_base_rates_qwen7b_v2.json` for
the authoritative numerator/denominator per category. The v2 files supersede v1.

**qwen-1.8b (v2, n=200 sampled per category):** biased/scored —
Religion 112/199, Race_ethnicity 91/199, Gender_identity 84/200, Age 98/193,
Nationality 94/191, Physical_appearance 112/196, Disability_status 106/199,
Sexual_orientation 103/200, Race_x_gender 83/198, Race_x_SES 58/148.

**qwen-7b (v2, n=200 sampled per category):** biased/scored —
Religion 32/195, Race_ethnicity 28/198, Gender_identity 24/192, Age 59/189,
Nationality 40/196, Physical_appearance 25/199, Disability_status 56/189,
Sexual_orientation 15/197, Race_x_gender 21/197, Race_x_SES 13/184.

**These numbers are NOT used in any current result.** The generation-based
pipeline that produced them was abandoned.

---

## 4. The base-rate and margin files on the box

| file | produced by | method version | superseded? | in notes/08? |
|---|---|---|---|---|
| `_base_rates_qwen18.json` | `bbq_base_rates.py`, qwen-1.8b | generation + **old** parser (refused on 2 matches) | **YES** by `_v2` | no |
| `_base_rates_qwen18_v2.json` | same, after the parser fix | generation + earliest-mention parser | superseded by the whole generation approach | partially |
| `_base_rates_qwen7b.json` | `bbq_base_rates.py`, qwen-7b | generation + old parser | **YES** by `_v2` | no |
| `_base_rates_qwen7b_v2.json` | same, after the parser fix | generation + earliest-mention parser | superseded by the whole generation approach | partially |
| `_margins_gemma2b.json` | `bbq_likelihood_margins.py`, gemma-2b | likelihood **with** the option list | **YES** — that design was found confounded | yes, as a failure |
| `_margins_qwen18.json` | same, qwen-1.8b | likelihood **with** the option list | **YES** — same reason | yes, as a failure |

Not in notes/08 at all: the v1 base-rate files.

---

## 5. Which labelling path produced the labels

**`target_loc`, from `third_party/bbq/additional_metadata.csv`**, for every run
after commit `5ef9641`. Verified: all 25,814 ambiguous rows across the ten local
files find a metadata row, zero misses, and `target_loc` never points at the
unknown option.

The reconstruction fallback (from `stereotyped_groups` + polarity) remains in the
code but was not used — `AnswerRoles.source` was `target_loc` for 600/600 rows on
the spot-check. 8 rows dataset-wide carry `target_loc="NA"` and are excluded.

The earliest runs of the night (base rates v1/v2, the with-option-list margins)
predate the answer key and used the reconstruction.

---

## 6. Artifacts written this session

All of `runs/` exists **on both** the box and the laptop
(`C:\Users\Jeremiah Zhang\research\soft-refusal-algoverse\runs`), synced by
`sync_from_box.ps1`, which verifies every file is non-empty, that JSON parses and
that `.npy` headers are valid.

Code lives on the laptop in `repo/` on branch `jz/bias-taxonomy` (local only,
never pushed) and is mirrored to the box by scp.

Notes live on the laptop only, in `notes/`.

---

## 7. Things done that are not steps 1–8 of `notes/03-experiment-1-plan.md`

Listed even where they were necessary:

1. **Two complete method changes.** The plan specified generation + a
   deterministic parse. That was replaced by likelihood scoring, and the first
   version of that was replaced again after being found confounded.
2. **BBQ's `target_loc` answer key** was adopted; the plan assumed reconstruction.
3. **`--cluster-usable-only`** — the plan clustered every category.
4. **A ridge-probe estimator** as an alternative to contrasting extremes.
5. **The extraction positive control** (topic directions) — not in the plan; it
   is what distinguishes "extraction is broken" from "nothing to extract".
6. **The H2 hypothesis and its dose-response test** — entirely new.
7. **`direction_norms.py`** — written after the transfer test was found confounded.
8. **Dose normalisation and a coefficient sweep** in the transfer test.
9. **A margin cache**, so estimators could be swapped without re-scoring.

---

## 8. Failures, retries, and results not to be trusted

- **Queue 1 OOM cascade.** The first driver waited for the previous *process* to
  exit but GPU memory release lags process exit. `transfer_qwen14b` started while
  ~29 GB was still held, OOM'd during model load, and then **hung holding the
  memory**, starving the next two steps. `transfer_qwen14b`, `h2_age_qwen14b` and
  `h2_disability_qwen14b` all exited 1 and produced no data on that attempt. The
  orphan was killed manually, the driver was rewritten to wait on GPU memory, and
  all three were re-run successfully.
- **`qwen-7b` taxonomy run: killed at ~6 minutes**, deliberately, to let the
  corrected transfer tests run first. It produced no data.
- **The first transfer tests (qwen-14b, gemma-2b) are confounded.** Direction
  norms vary 5× across categories and correlate with the extraction floor at
  r=+0.91, while the coefficient was fixed — so reproducible categories received
  a 5× stronger dose. Their specificity numbers should not be used. Re-run with
  per-layer normalisation.
- **The probe estimator underperformed.** 0/10 reproducible on qwen-14b versus
  4/10 for extremes. `alpha=1.0` was never tuned and is almost certainly far too
  small for d_model=5120, so this is not a fair test of the probe.
- **A misleading verdict string** was emitted by `gemma2b_extremes`: "clustering
  is within the permutation null (p=1.000)" when no clustering or null had run.
  Fixed after that run; gemma's `report.json` still contains the bad string.
- **The `_smoke*` run directories** are throwaway smoke tests, not results.
- **`llama3-8b` never ran.** `403 — gated repo, not in the authorized list` for
  `meta-llama/Meta-Llama-3-8B-Instruct`. `runs/full_llama3/` exists but is empty.
  Request access on the model page if it is wanted.

---

## 9. Queue status — every step and its exit code

**Queue 1** (`runs/overnight.log`), first attempt — the OOM cascade:

| step | exit | produced data? |
|---|---|---|
| transfer_qwen14b | **1** | **no** — OOM, orphaned process held 29 GB |
| h2_age_qwen14b | **1** | **no** — starved by that orphan |
| h2_disability_qwen14b | killed | **no** — same |

**Queue 1**, after the driver was rewritten to wait on GPU memory:

| step | exit | produced data? |
|---|---|---|
| transfer_qwen14b | 0 | yes — `full_qwen14b/transfer_test.json` (confounded, superseded) |
| h2_age_qwen14b | 0 | yes — `_h2_dose_qwen-14b_Age.json` |
| h2_disability_qwen14b | 0 | yes — `_h2_dose_qwen-14b_Disability_status.json` |
| gemma2b_control | 0 | yes — `_extraction_control_gemma2b.json` |
| gemma2b_extremes | 0 | yes — `runs/full_gemma2b/` |
| transfer_gemma2b | 0 | yes (confounded, superseded) |
| yi6b_control | 0 | yes — `_extraction_control_yi6b.json` |
| yi6b_extremes | 0 | yes — `runs/full_yi6b/` |
| transfer_yi6b | 0 | yes — normalised, `full_yi6b/transfer_test.json` |
| qwen7b_extremes | **killed** | **no** — stopped deliberately at ~6 min to prioritise queue 2 |
| llama3_extremes | not reached | no |

**Queue 2** (`runs/overnight2.log`) — all steps exit 0 except the last:

| step | exit | produced data? |
|---|---|---|
| transfer_qwen14b_norm_c2 / c4 / c8 / c16 | 0, 0, 0, 0 | yes — 4 files |
| transfer_gemma2b_norm_c2 / c5 / c10 | 0, 0, 0 | yes — 3 files |
| norms_full_qwen18 / qwen14b / gemma2b / yi6b / probe_qwen14b | 0 x5 | yes — `direction_norms.json` in each |
| qwen7b_extremes | 0 | yes — `runs/full_qwen7b/` |
| llama3_extremes | **1** | **no** — gated repo, 403 |

**Steps that produced no data, total:** the three queue-1 OOM casualties (all
re-run successfully afterwards), the deliberately-killed first `qwen7b_extremes`
(re-run successfully in queue 2), and `llama3_extremes` (never ran).

---

## 10. Where everything is

- **Laptop:** `C:\Users\Jeremiah Zhang\research\soft-refusal-algoverse\`
  - `runs/` — **275 files, 64.6 MB**, synced and verified (non-empty, JSON
    parses, `.npy` headers valid)
  - `notes/` — this report, the results, the plan, the literature read
  - `repo/` — the lab repo on branch `jz/bias-taxonomy`, **local only, never
    pushed**
- **Local git:** the project folder is its own repo (`repo/` excluded); the lab
  repo has its own history on the branch.
- **Box:** everything under `~/Algoverse-Bias-Steering/runs/` — a superset of
  nothing; the laptop copy is complete.

**The Lambda instance can be terminated.** Everything has been copied and
verified. Terminate it from the dashboard — *terminate*, not stop.
