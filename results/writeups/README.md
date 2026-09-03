# Reading order for the bias-taxonomy writeups

**If you are an agent picking this up: read this file first, then `17`, then
`12`. Do not plan from `00`–`11` alone — they were written before run 1 executed
and before the defect audit, and several of their conclusions have since been
overturned.**

Documents `12`–`23` were added on 2026-08-30. Until then the repository carried
only `00`–`11`, so anything written against this directory before that date was
working without the defect register, without the parser audit, and without the
diagnosis in `17`.

---

## The one-paragraph version

Experiment 1 asked whether different BBQ bias categories occupy distinct
directions in the residual stream. It ran on five models across three families
on 2026-08-20/21. **Its headline numbers should not be treated as established.**
An audit found fourteen defects, eight graded severe, and a later reading of the
reference paper identified a root cause that none of the individual defects
name: **the contrast itself was labelled by the model's own behaviour rather
than by a dataset annotation.** The re-run replaces that contrast.

---

## Read in this order

| # | file | what it is |
|---|---|---|
| **1** | **`17-reference-paper-and-contrast.md`** | **The diagnosis. Authoritative for the science.** Joad et al. get within-category floors of 0.95–0.99 from **32 items per class** using dataset annotations; run 1 used 240–600 and got **−0.45 to +0.82**. Sample size cannot explain that gap. The contrast can, and it is the only thing left that differs. |
| **2** | **`12-retrospective.md`** | **The defect register.** S1–S5, N1–N5, M1–M3, every one recomputed from artifacts before being accepted. Read the consolidated table at the foot. |
| **3** | **`18-parser-audit.md`** | **Defect N6.** The choice parser resolves ties by earliest mention, so it fails *toward position*. Three of seven realistic phrasings parse wrong. The unparsed rate was reported honestly; the **misparsed rate is unmeasurable**, because raw response text was never saved. |
| 4 | `13-preregistration.md` | The frozen spec for the re-run. **Read the header warning, then §15** — §14 is the superseded table. |
| 5 | `14-run-plan.md` | Execution plan, caching requirement, kill criteria, and a hostile self-review. |
| 6 | `19-plan-closure-and-audit.md` | Contradictions found *between* the planning documents, with file and line, plus closures for the three thresholds that were left undeclared. |
| 7 | `16-method-1-reexamined.md` | Why the generation-and-parse method was weak rather than random. **Carries a suspension banner — read it with `18`.** |
| 8 | `22-run1-reanalysis-findings.md` | Seven analyses of run 1 done later from cached artifacts, no GPU. Includes the robustness check that the headline does not depend on which summary statistic is used. |
| 9 | `23-research-programme.md` | What Experiments 2–4 should be, each sized against real data. |

`00`–`11` remain useful as history: `03` is the original plan, `08`/`09`/`10` are
run 1's results, and **`11-EXPERIMENT-PROTOCOL.md` is process rather than science
and all of it still applies.**

---

## The four things most likely to be misread

**1. "Run 1 showed bias has no reproducible directions."** Overstated. It showed
that *a behaviour-derived contrast* does not reproduce. See `17` §9.

**2. "The extraction floor is a validity check this literature skips."** False
with respect to the reference paper — Joad et al. Appendix C Table 8 is the same
statistic. `17` §4.2 orders this removed wherever it appears.

**3. "M1 is not closable by this design."** True of the run-1 design, false of the
one now adopted. `12` records it as unclosable and `17` §4.1 overturns that.

**4. The margin-quintile contrast, the 0.50 usability threshold, and the ridge
probe as primary** are all superseded. They are kept deliberately, marked, and
`13` §15 is the authoritative parameter table.

---

## What the hardening runs do and do not address

`docs/HANDOFF_GPU_HARDENING.md` (WP-43, P0–P3) tests four declared-fragile
numbers, and three of the four can kill one:

- **P0** — does the 0.50 bar transfer to the probe at all? This is defect **S4**.
  If it does not, both clustering p-values are thresholded against the wrong
  reference.
- **P1** — does α ever plateau? This is **S3**: the same category's direction at
  α=1 and α=1e6 agree at only **0.10–0.21**, so α selects a direction rather than
  tuning one. Also persists residuals, closing **S5**.
- **P2** — do the two heavy-tailed positives survive de-tailing? **M2**.
- **P3** — is the race negative about the unit of analysis? Pre-registered split
  in `runs/_p3_manifest.json`.

**None of them changes the contrast.** They are diagnostics on run 1's design,
not a re-run of it. The re-run is the annotation-derived contrast specified in
`17` §5.1 and `13` §15, and it is a separate thing from P0–P3.
