# WORK_LEDGER.md — execution state

Work is organised as **work packages (WP)**, not people. Ownership is an
attribute of a package, not its identity — a package keeps its meaning when it
changes hands. Anyone, human or agent, should be able to pick up an unowned
package from its definition and evidence trail without the previous owner's
private context.

**A package is done when its evidence exists and validates. Not when someone
says it ran.** Status is derived from the evidence column; if the artifact is
absent, the status is not `done` regardless of what anyone reports.

Scientific commitments live in `RESEARCH_CONTRACT.md`. Current gate and headline
state live in `PROJECT_STATE.md`. This file owns *execution only* and may not
redefine a metric, a claim, a deadline, or a definition of done that the contract
sets.

Legend — **Blocks?** = does this block submission. `Y` packages are the critical
path; everything else waits.

---

## Critical path

| ID | Objective | Depends | Status | Owner | Definition of done | Evidence required | Validation | Blocks? |
|---|---|---|---|---|---|---|---|---|
| **WP-01** | Merge the correctness + recovery branch into the working line | — | **done** | Claude | Shape guard, worktree fix, 67 recovered artifacts on one branch | `fix/steering-shape-guard` @ `d2a79f8` | `python3 tests/test_phase{0,1,2}.py` → 31/31 | Y |
| **WP-02** | Reconcile `main` ← `team-kit` | WP-01 | **prepared, not executed** | Farhan | One line holding pipeline + docs; `.gitignore` = union | Merge commit; suite still 31/31 | Re-run suite post-merge | Y |
| **WP-03** | Ablation operator `apply_resid_ablate` (partial, λ-parameterised) | WP-01 | **not started** | Aryaman | `x − λ(x·r̂)r̂` at contract-specified layers/positions; shape-guarded; unit-tested | `src/bias_steer/steering.py` diff + tests | λ=0 is identity; λ=1 zeroes the component; random-direction control runs | Y |
| **WP-04** | Extraction protocol implementation | WP-02 | **not started** | Farhan | Replaces `generate_with_cache()` re-run-the-response-string substrate with the contract's prompt-position capture | Code diff + one direction file with metadata | Direction hash reproducible from committed inputs | Y |
| **WP-05** | Positive control: Arditi replication (E0) | WP-03, WP-04 | **not started** | Farhan | Ablating `r̂_harm` measurably suppresses harm refusal on the harm battery | Run dir with results + provenance record | Effect present and signed correctly, else **project kill gate** | Y |
| **WP-06** | Primary DV extractor `named_a_side` | — | **not started** | Jeremiah | Deterministic extractor over the 296 forced-binary items | `scripts/extract_named_side.py` + per-item output | Human-validated on a sample; error bounded per contract | Y |
| **WP-07** | Validity instrument + blinded annotation | WP-06 | **not started** | Jeremiah | Minimal binary/ternary validity judgement, blinded to condition | Blinded pools, individual label files, agreement computation, frozen version tag | Agreement meets the contract's gate on the load-bearing category | Y |
| **WP-08** | λ-sweep, both directions × both batteries, k generations/item | WP-05, WP-06 | **not started** | Farhan | Full preregistered grid executed | Raw generations + results + provenance per cell | Provenance record complete for every cell | Y |
| **WP-09** | Controls: covariance-matched random direction, wrong-layer | WP-08 | **not started** | Aryaman | Operator-matched ablation controls at the same λ grid | Run dirs, same schema as WP-08 | Same operator as primary; not an additive offset | Y |
| **WP-10** | Analysis + decision rule evaluation | WP-08, WP-09 | **not started** | Jeremiah | Primary statistic computed with CIs; contract's rule evaluated | `analysis/` script + committed outputs | Rerunnable from committed artifacts alone | Y |
| **WP-11** | Preregistration filled and hash-committed | contract frozen | **not started** | Edward | `docs/PREREG.md` populated; hash recorded before WP-08 reads off-target | Commit hash in the contract | Hash predates first WP-08 result | Y |
| **WP-25** | **Stratify the battery (S1/S2/S3)** | — | **not started** | Edward | Stratum column on all 296 items; 3 duplicates removed; 18 unnameable-alternative items flagged for DV exclusion | Committed CSV + adjudication notes | S2/S3 boundary human-adjudicated, not heuristic | **Y — gates WP-04** |
| **WP-13** | Final adversarial freeze review | contract drafted | **running** | Claude | Every paper-killing objection resolved without adding scope | Critic reports + resolutions in contract §12 | No unresolved paper-killer | Y |
| **WP-12** | Judge pinning + k≥3 with majority | WP-02 | **not started** | Aryaman | Dated snapshot model id; k judgments/item; per-item agreement recorded | Config diff + judge agreement file | Rerun on a fixed sample reproduces labels | Y |

## Supporting — required but not gating

| ID | Objective | Status | Owner | Definition of done | Blocks? |
|---|---|---|---|---|---|
| **WP-20** | Item attributes: obvious-answer vs no-privileged-answer over the 296 items | not started | Edward | One committed CSV, one attribute per item, before travel | N |
| **WP-21** | `scripts/kappa_from_csv.py` → Fleiss κ_j + bootstrap CI + Gwet AC1 | not started | Jeremiah | Multi-rater support (2-rater Cohen only today, ~60 lines) | N |
| **WP-22** | Related work + positioning paragraph | not started | Edward | Joad / Arditi / Wollschläger / non-identifiability placed | N |
| **WP-23** | Trim public index rows exposing non-public research paths | not started | Edward | `docs/SOURCES_OF_TRUTH.md` rows removed or pointed at a private index | N |
| **WP-24** | Fix `pytest tests/` registry-teardown poisoning | not started | unowned | Suite green as a suite, not only per-file | N |

## Forensic — separate from the causal controls

| ID | Objective | Status | Owner | Note | Blocks? |
|---|---|---|---|---|---|
| **WP-30** | Deliberate reconstruction of the 2025 scalar-broadcast failure | not started | unowned | Scientifically informative as a demonstration of the historical failure mode. **It is an additive scalar offset, not an ablation. It must never be reported as an operator-matched control for WP-09.** | N |

## Closed

| ID | Objective | Outcome |
|---|---|---|
| **WP-00a** | Determine whether campaign artifacts were lost | **Not lost.** Committed at phase boundaries, dropped from HEAD by a later checkpoint. 67 artifacts across 12 runs restored from history. Only `anchor-qwen-14b` genuinely has no vector commit — left incomplete and marked. |
| **WP-00b** | Determine whether the 1-D broadcast bug is live in the pipeline | **Not live in pipeline outputs.** Every recovered vector is `(24, 2048)`. The failure is confined to the 2025 `experiments/*_vecs` archive, which is still what a person would load — hence the guard in WP-01. |

---

## Builder / validator split

No paper-blocking package is validated by the person who produced it. Validation
is from the **artifact and the contract**, not from a conversation.

| Package | Builder | Validator |
|---|---|---|
| WP-03 ablation operator | Aryaman | Farhan |
| WP-04 extraction protocol | Farhan | Aryaman |
| WP-05 positive control | Farhan | Jeremiah (reads the run dir, not the notebook) |
| WP-06 DV extractor | Jeremiah | Edward |
| WP-08 λ-sweep | Farhan | Jeremiah |
| WP-10 analysis | Jeremiah | Farhan |

**Independence constraint carried from the contract:** whoever selects the layer
and direction for `r̂_stance` must not run or score the harm battery, and the
selection must be committed by hash before any off-target cell is read.
