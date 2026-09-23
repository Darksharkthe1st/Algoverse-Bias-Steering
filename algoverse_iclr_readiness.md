# ICLR 2027 MAIN-track readiness review — Algoverse steering project

**Reviewer role:** ML researcher / area-chair-style read. **Date:** 2026-09-22.
**Repo:** `C:\Users\farha\GITHUB\Algoverse-Bias-Steering\`
**Constraint assumed:** ~3 days experiment/compute + ~1 day writing; H100s available, human hours scarce.

> **Read this first.** The paper described in the brief — *"Characterizing Soft
> Refusal in LLMs via Activation Steering," 9 models, an async LLM-judge with a
> 9-way split validated at ~70% Cohen's κ, aimed at ICLR main* — **does not exist
> as an artifact in this repo.** That description matches an *abandoned* framing.
> The repo contains two *different* things, one incomplete and one nearly-finished,
> and neither is the paper in the brief. Everything below is grounded in files.

---

## 0. What is actually in the repo (the framing gap is finding #1)

The brief's ingredients each map to something real, but the current project has
moved off all of them:

| Brief says | Repo reality | Evidence |
|---|---|---|
| "Soft refusal" is the construct | **Retired.** Renamed *hedging / over-abstention on answerable items* | `RESEARCH_CONTRACT.md` §2 ("**'Soft refusal' is retired**"); `CLAUDE.md` §5; `docs/RUBRIC_v2.md` L8-10 keeps the label name only "for continuity" |
| ~9 open-weight models | Current frozen paper is **1 primary model** (`Qwen3-8B`), fallback `Qwen1.5-7B`. The completed paper uses **5** old models | `RESEARCH_CONTRACT.md` §10; `paper/main.tex` L109-113 |
| 9-way judge validated at ~70% κ | Judge emits 9 labels (real), but the **κ validation was never run** — it is *BLOCKED on human labels*, and the team's own simulation says soft-refusal κ **ceilings at 0.52** | `src/bias_steer/judges/v2.py` L24-28; `scripts/kappa_from_csv.py` header ("STATUS 2026-09-02: BLOCKED on human labels"); `DECISION_LOG.md` D-010 ("at n=2400 median per-category κ for soft refusal is still 0.52") |
| 9-way behavior split is the DV | The 8/9-way cascade was **cut as an instrument**; replaced by a deterministic extractor + a *ternary* human validation | `DECISION_LOG.md` D-010; `RESEARCH_CONTRACT.md` §8 ("The eight-way cascade is dead and may not return") |
| ICLR 2027 main submission | Repo targets a **non-archival NeurIPS 2026 workshop** (deadline 2026-08-28, now passed). No ICLR submission artifact exists | `PROJECT_STATE.md` L8-10; `paper/main.tex` L2 |
| Memory opts, 64× batches on H100 | **Real.** no-grad caching, resid→CPU/vector→VRAM, shape guards | commits `a4ba1b7`, `c60eda3`; `src/bias_steer/steering.py`; `src/bias_steer/models.py` |
| Async OpenAI LLM-judge | **Real.** `AsyncOpenAI` + semaphore concurrency, ret/seed/temp0 pinned | `src/bias_steer/judges/base.py` L67-100 |
| Vector ablation + addition | **Real.** directional ablation (project-out everywhere) + act-add (single-layer) + per-layer linear add | `src/bias_steer/steering.py` L125-280 |

**There are two candidate papers, on different branches:**

- **Paper A — the frozen study** (`main`, `RESEARCH_CONTRACT.md`): *"is hedging on
  answerable comparative questions controlled by the same latent as harm refusal?"*
  on `Qwen3-8B`. **Incomplete.** Its headline mechanism claim was **already cut**
  by the team's own adversarial review to "a measurement + a preregistered bound,"
  because the design *provably cannot distinguish the hypotheses at achievable n*
  (needs 4.7× the data asset — `RESEARCH_CONTRACT.md` §0, §7.1; `DECISION_LOG.md`
  D-009/D-010/D-011). Its sole gating positive control **G1 has never passed** on
  the submission model (`PROJECT_STATE.md` "Current gate"; contract §12-A5/A6).

- **Paper B — the completed workshop draft** (`origin/jz/run3-results`,
  `paper/main.tex`, 5pp): *"Extraction works, the contrast does not: positive
  controls for difference-in-means directions."* A **negative-result / methodology**
  paper: difference-in-means (DiM) bias directions for BBQ categories fail to
  reproduce across data splits on 5 models, while a positive control recovers a
  control contrast. **This is the strongest real asset in the repo** and the only
  thing that looks like a finished paper.

The rest of this review assesses **Paper B against the ICLR-main bar**, because it
is the only viable candidate on a 3+1-day horizon. Paper A cannot be made
main-ready in the time (its central claim is retired-by-design and its positive
control is unbuilt); I say so explicitly in the verdict.

---

## 1. Current-state assessment — Paper B (`paper/main.tex`)

### Solid (genuinely above the workshop bar)

- **The core design is clean and honest.** Split-half cosine stability as the
  reproducibility criterion, with a random-direction noise floor (0.014–0.022) far
  below the 0.500 bar (`main.tex` L123-143). The negative result is real: bias
  floors span −0.45 to +0.82; **every race-related category fails in every model**
  (Table 2 / Fig 1, L235-275).
- **The two-tier positive control is the actual contribution and it is good.** A
  behavioural control (scorer valid on disambiguated BBQ, L159-208) plus an
  extraction control (topic-identity recovers at cos 0.86–0.92 through the identical
  code path, L210-225). This is what "licenses" the negative — it separates
  instrument failure from construct absence, which most probing papers do *not* do.
- **Unusual, credited self-honesty.** §5 reports that reproducible directions
  *don't steer selectively* (own ≤ cross ≤ random on qwen-14b; the gemma-2b "7×
  success" dissolves under a cross-category control and a dose sweep, L360-392). §6
  reports a **forking path** in its own clustering (p=0.179→0.030 under a ridge
  penalty chosen upstream of the p-value, L398-455). §7 catalogues six silent
  pipeline defects (scalar-broadcast bug, first-mention-biased parser, unpinned
  judge, L457-597). The team's own `paper/HOSTILE-REVIEW.md` concludes "**I would
  accept this**" for a workshop.
- **Cross-family replication** (Qwen / Gemma / Yi) and a real predictor analysis
  (floor correlates with direction *norm* r=+0.77…+0.95, not behavioural bias,
  L288-309).
- **Engineering is trustworthy.** Extensive test suite (`tests/test_g1_stability.py`,
  `test_judge_v2.py`, `test_phase0-4`, `test_repro_foundation.py`), numbers tagged
  to `runs/` artifacts, provenance discipline (`paper/VERIFICATION.md`,
  `numbers.json`).

### Thin / load-bearing weaknesses (this is what a main-track reviewer attacks)

- **Single capture site, single estimator, single token position.** The negative
  claim is explicitly scoped to "the residual stream at the final prompt token, by
  difference-in-means" (L511-518). A main reviewer reads a one-site negative as
  *under-powered*, not *general*.
- **The intervention arm is uncontrolled and the paper says so.** "**Not causal.**
  This arm used a norm-matched rather than covariance-matched random control, ran no
  coherence check on generations, and had no system-prompt baseline" (L389-392).
  For a *steering* paper this is the softest spot.
- **Judge is unvalidated and, in the audited past, unpinned.** No human-agreement
  number is reported; §7 admits an unpinned-alias judge whose labels are now
  unreproducible (L592-597). κ was never computed against humans (`kappa_from_csv.py`).
- **Clustering result is a forking path with n=5 and n=3 categories** (L517) —
  honestly reported, but it is not a result you can lean on for main.
- **Old, small models** (1.8B–14B, all Qwen1.5-era + Gemma-1 + Yi), conceded at
  L518. No 2025/2026-class model.
- **One live numeric error** the team already caught but hasn't fixed in the draft:
  a clustering p=1.000 in Table 5 that **was never computed** (`HOSTILE-REVIEW.md`
  A1b — "a p-value you never computed").

### Rigor inventory

- Seeds/pins: judge seeded, temp 0 (`judges/v2.py`); model revision SHAs pinned in
  PREREG §3b. **Good.**
- Significance: 200-permutation null for clustering; bootstrap 95% CIs on floors
  exist in `notes/22` but are **not yet in the draft** (per `HOSTILE-REVIEW.md` A2).
- Ablations: cross-category and dose-sweep controls present in the steering arm;
  no layer/site ablation.
- Multiple seeds on the main floor statistic: 10 splits, but gated on a single q05
  quantile (attack A2).

---

## 2. Judgment against the ICLR-MAIN bar

**Novelty framing.** The adoptable idea — *pre-committed positive controls that
separate "instrument broken" from "construct absent," used to license a negative* —
is genuinely useful and not standard practice. That is a main-track-shaped
*methodological* contribution. But as written the paper sells itself as a **case
study** ("we report a case where…"), which reads as workshop. Main track needs the
control-pair reframed as a **general diagnostic** and demonstrated on more than one
site/estimator/dataset.

**Baseline strength.** Weak for a steering paper: no system-prompt baseline in the
intervention arm (the field's expected cheap baseline, and the repo *already has*
one on `origin/fk/init-prompt-control`, which found **"prompt wins outright"** over
steering — commits `914bce…`/`6c32c02`). No covariance-matched random control.

**Breadth.** 5 models is fine for a workshop, borderline-thin for main; all pre-2025.
One dataset (BBQ), one construct family (social bias), one site.

**Evaluation validity.** The split-half/positive-control machinery is sound. The
LLM-judge is **not validated** (no human κ) and historically unpinned — a reviewer
will flag this hard since judged counts feed the steering table.

**Statistical rigor.** Above workshop median (permutation null, floor bootstrap
available), but the headline clustering is a self-admitted forking path and the
floor gate is a single quantile. Fine if all the CI/robustness material from
`notes/22` is folded in; not fine as currently drafted.

**Clarity of the central claim.** Excellent and appropriately narrow — the title
*is* the claim. The problem for main is the opposite of the usual one: the claim is
**too narrow to be a main-track headline** unless generalized.

**Bottom line vs bar:** This is a **strong workshop paper** and a **borderline /
below-the-line ICLR-main paper** as it stands. Negative-result + methodology papers
*can* land at main, but essentially only when the diagnostic is shown to generalize
and the obvious cheap baselines/controls are all closed. Right now two of those
(system-prompt baseline, covariance-matched control) are open and one (judge
validation) is unmet.

---

## 3. RED FLAGS a reviewer will attack (ranked)

1. **Uncontrolled intervention arm in a steering paper** (`main.tex` L389-392). No
   covariance-matched control, no coherence gate, no system-prompt baseline.
   *Highest-damage, and you already have the infra to close it.*
2. **Unvalidated / historically-unpinned judge** feeding reported counts
   (L592-597; `kappa_from_csv.py`). No human-agreement number anywhere.
3. **A p-value in Table 5 that was never computed** (`HOSTILE-REVIEW.md` A1b). This
   is a factual error in the submitted numbers — must be fixed before *any* submission.
4. **Single site / estimator / token position** → "your negative is under-powered,
   not general" (L511-518).
5. **Forking-path clustering with n=3–5 categories** presented in the results
   (L398-455). Honest, but a main reviewer discounts it.
6. **Old, small models** (L518).
7. **"Positive control is too easy"** — topic identity is trivially decodable
   (`HOSTILE-REVIEW.md` A1). *Already answerable from existing data (Disability /
   Physical reproduce through the same pipeline) — 4 sentences.*
8. **Framing/identity risk** (management, not science): the finished paper lives on
   *Jeremiah's* branch (`origin/jz/run3-results`) and the frozen contract's owner is
   *Edward*; the brief lists Farhan as first author on a *different* (soft-refusal)
   title. Reconcile authorship and which paper is being submitted **before** doing
   any of the work below.

---

## 4. Day-by-day plan (3 experiment days + 1 writing day)

Ordered by impact-per-hour. Every item is scoped to finish in the time and reuses
code/artifacts already in the repo. **All compute is extraction/steering on
`Qwen3-8B`-class models, which the memory-optimized pipeline already handles.**

### Day 0 (first ~2 hours, free — do before touching a GPU)
- **Fold the 5 already-answered hostile-review attacks into the draft**
  (`HOSTILE-REVIEW.md` priority list: A1b one-line fix, A1 four-sentence two-tier
  control, A4 convention, A3 threshold-robustness, A2 the 27% flip + power sentence).
  These are text-only, use data you already have, and close reviewer objections
  #3 and #7 outright. *Closes: the never-computed p-value; "control too easy."*

### Day 1 — Close the intervention arm (kills the #1 red flag)
- Re-run the steering sweep on the reproducible directions with **(a) a
  covariance-matched random control** (not norm-matched), **(b) a generation
  coherence check** (perplexity / repetition-loop guard — the repo already detects
  "total token-level collapse," commits `c8be0dd`/`80a7f98`), and **(c) a
  system-prompt baseline**. The system-prompt harness already exists on
  `origin/fk/init-prompt-control` and already found *prompt beats steering* — port
  it, don't rebuild it.
- Deliverable: replace §5's "Not causal" disclaimer with a properly-controlled
  table. *Closes reviewer objection #1, and the "prompt wins" result is itself a
  publishable point (cheap baseline > steering).*

### Day 2 — Make the negative *general*, not single-site (kills #4)
- **Layer/token-position sweep of the split-half floor** for the bias contrast vs
  the two positive controls, on 2–3 existing models. This is cheap: it reuses
  *cached residuals*, no generation. Show the bias contrast fails across sites while
  the control succeeds across sites. This converts "not present at the one site we
  probed" into "not linearly recoverable across the residual stream." *This is the
  single most reviewer-swaying experiment for main.*
- **Add one modern model** (the contract's own `Qwen3-8B`, plus optionally a
  Llama-3.1-8B) to the floor analysis. Extraction-only, so it fits in hours and
  answers "old, small models" (#6) and demonstrates the *method* is model-agnostic.

### Day 3 — Judge validation + statistical hardening (kills #2, blunts #5)
- **Validate the judge** with a small but real human-agreement number: pull ~120
  responses (the ternary instrument in `RESEARCH_CONTRACT.md` §8 / WP-07 is
  designed for exactly this), have 1–2 people label them, and report Cohen's κ
  (judge↔human) via `scripts/kappa_from_csv.py`. Even κ on the *collapsed* view
  beats reporting nothing. **Pin the judge to a dated snapshot** while you're there.
  *If human hours truly aren't available:* at minimum pin the model and report
  judge self-consistency across seeds + a second-judge cross-check — but a real
  human κ is worth far more.
- **Fold the floor bootstrap CIs and the min/q05/median/mean threshold-robustness
  table** (`notes/22` §A/§C/§F) into the paper. De-emphasize the forking-path
  clustering to a clearly-labeled robustness note rather than a headline (#5).

### Day 4 — Writing / reframing (the load-bearing writing day)
- **Reframe from case study → general diagnostic.** Lead with "pre-committed
  positive controls to license negatives about linear representation," with BBQ as
  the demonstration and the new multi-site/multi-model evidence as the generality.
- Tighten the central claim to exactly what the (now stronger) evidence supports;
  write a real related-work section (Arditi, Joad et al., Wollschläger — all already
  cited); move Table 2 to appendix per the page-budget note; expand the appendix
  with the new controls.
- Sanity-pass every number against `runs/` (repo rule: numbers trace to artifacts).

**Cut if time runs out:** Day 3's clustering rework before Day 3's judge validation;
the extra Llama model before the site-sweep. Never cut Day 0 or Day 1.

---

## 5. If you only do ONE thing

**Close the intervention arm (Day 1): re-run steering with a covariance-matched
random control, a coherence gate, and a system-prompt baseline.** It removes the
single most damaging red flag ("Not causal," admitted in the paper), it is the
control a *steering* paper is judged on, and the infrastructure and even the
punchline (**prompt beats steering**) already exist on
`origin/fk/init-prompt-control`. One GPU-day converts the paper's weakest section
into a genuine result.

*(Do the free Day-0 text fixes regardless — especially the never-computed p-value
in Table 5, which must not go into any submission.)*

---

## 6. Candid verdict

**Solid ICLR-main is not realistic in 3+1 days from this material — and the paper in
the brief is not the paper you have.**

- **Paper A (soft-refusal / hedging on Qwen3-8B) cannot be main-ready.** Its central
  mechanism claim was retired *by the team's own preregistered analysis* (the design
  can't separate the hypotheses at achievable n), and its one gating positive control
  has never passed. Reviving it as a main submission in four days would mean
  submitting a claim the authors have already shown they cannot support. Don't.

- **Paper B (the BBQ DiM negative-result paper) is a genuinely good workshop paper**
  and, with the plan above executed, becomes a **strong workshop paper / borderline,
  rebuttal-dependent ICLR-main submission** — not a confident main accept. The honest
  ceiling in the time is: close the intervention arm, generalize the negative across
  sites and one modern model, and attach a real judge-κ number; that gets you to a
  paper a sympathetic AC could champion and an unsympathetic one could still reject
  on breadth (one dataset, one construct family, 5–6 models). Its natural, high-odds
  home remains a strong workshop or a Findings/DBLP-style track.

What would actually move Paper B to *solid* main is out of budget: multiple datasets
and construct families beyond BBQ social bias, SAE / non-linear-probe cross-checks to
back the "not linearly recoverable" claim, a fully validated judge with a
human-labeled test set, and 2–3 more modern models. That is weeks, not days.

**Recommendation:** aim Paper B at a top interpretability **workshop** with the 3+1
plan (it will be well above that bar and the honesty is a real asset there), and use
the same work as the spine of a *later* main-track submission once the multi-dataset /
multi-site generality and judge validation are actually built. Trying to force it into
ICLR main this cycle risks a confident-reject on breadth and controls that a workshop
would have accepted enthusiastically.

---

## Path to an ICLR-submittable paper  (CORRECTED — SSOT = Farhan's own results; all `jz/*` excluded)

> **THIS SECTION SUPERSEDES EVERYTHING ABOVE IT.** The passes above were built on
> Jeremiah's `jz/*` branches (`origin/jz/run3-results`, `paper/main.tex`, the
> "negative-result / positive-controls" framing, the "headline contradiction").
> Per Farhan (2026-09-22), **all `jz/*` code is non-authoritative AI slop, never
> produced valid results, and is NOT the paper.** Discard the negative-result
> framing, the `main.tex` on that branch, and the §A "contradiction" from the prior
> pass — that contradiction was an artifact of trusting jz. The real paper is a
> **POSITIVE steering result** and its single source of truth is Farhan's own work
> (`main` + `fk/*` + `farhan-*` branches). Everything below is grounded only in those.
> Repo docs were not modified.

### The actual paper (positive steering), stated from Farhan's artifacts

**Claim shape:** *Activation steering reliably and controllably moves an open-weight
model between opinionated and neutral answering — at high success rates, across many
models — and an adaptive per-layer linear schedule pushes the success/coherence
frontier further than a fixed scale, beyond what the fixed baseline reaches.* The
behaviour ("opinionated vs neutral" on forced-choice/comparative prompts) is the same
family the frozen contract calls *hedging / over-abstention* — the honest construct
name is opinionated↔neutral steering, and "soft refusal" is the informal label for it.

### 1. SSOT inventory — FINISHED & trustworthy vs INCOMPLETE

**FINISHED / trustworthy (Farhan's own, numbers read from committed artifacts):**

- **The ~year-ago 90%+ fixed-add opinion baseline, 7 models.**
  `experiments/past_logs/general_experiments/better_coeff_tests/Better_Coeff_tests.csv`
  (also `.xlsx`) is the clean table. Under the opinionated-steering arm, judged
  **opinionated-retention (Opin→Opin) out of 99**: Qwen1.5-1.8B **86**, Qwen1.5-7B
  **89**, Qwen1.5-14B **94**, Yi-6B **90**, gemma-2b **92**, gemma-7b **93**,
  Llama-3-8B **77**. That is the "90%+ across models" result (6 of 7 ≥86%,
  best 94%). Per-model raw logs + steered/pre-steering responses:
  `experiments/past_logs/past_vecs/best_opinion_vectors/Log_113…119_*`; the saved
  vectors are `experiments/best_vecs/log_11{3..9}_*_steer_vec.pkl`. **Caveat to
  carry:** the neutral arm (Neut→Neut) is much weaker/more variable (89/84/71/50/78/
  49/96) — the *opinionated* direction is the strong result, "neutral" is not
  symmetric. Judge here is the OLD 3-way opin/neut/nonsense judge; n≈99; layer −1;
  max_tokens 128; coeff-swept.
- **The adaptive-linear extension on Qwen3-8B (today), judged.** Branch
  `origin/fk/adaptive-steering-qwen3-run`, configs
  `configs/exp/adaptive_add_linear_c{8,16,20,30,40}_qwen3_8b.py`, runs
  `runs/20260903-01{4921,…}_adaptive-add-linear-c{16,20,30}-qwen3-8b_*/summary.md`.
  Opinionated (steered_pos) count out of 200, from a ~50/200 opinionated baseline:
  **c16 → 114**, **c20 → 126**, **c30 → 190** (95%), with c30 opinion-quality
  good=138 / bad=0. steered_neg→neutral ≈ 180–187/200. The linear schedule lifts
  POS above fixed_add as dose rises (commits `7fec77a`, `66e3320`). **Hard caveat:
  c30 carries an explicit quality caveat and c40 is committed as "INVALID — found the
  coherence ceiling" (`fe90279`)** — so 95% sits right at the coherence edge and the
  coherence gate is load-bearing, not optional.
- **The fixed-add vs adaptive-ablation mechanism + judged comparison on Qwen3-8B.**
  `experiments/adaptive_vs_fixed/summary.md` + `mechanism_summary.json` (fixed_add
  c=8: neutral→opinionated 66/146 under +c; the two operators are structurally
  distinct — ablation zeroes the projection, add is a uniform translation). This is
  the "adaptive ≠ fixed" evidence, clean and reproducible.
- **The steering operators themselves** (trustworthy, unit-tested):
  `src/bias_steer/steering.py` — `apply_resid_pre_add` (fixed add, `c/n_layers·vec[L]`),
  `apply_directional_ablation` (project-out), the adaptive per-layer hooks
  (`adaptive_ablation`, `adaptive_add_linear`, commit `16a8378`), shape guards,
  `check_direction`. Tests: `tests/test_apply_vector.py`, `test_phase*`. Async OpenAI
  judge with seeded temp-0 and rubric-SHA pinning: `src/bias_steer/judges/base.py`.
- **Extracted Qwen3-8B opinion vector** (no refit needed for new runs):
  `runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors` `(36,4096)`.
- **AxBench / IssueBench opinion-transfer batteries already run on Qwen3-8B:**
  `runs/20260901-093135_apply-opinion-vec-on-axbench…`, `…160507_…issuebench…`
  (branch `fk/adaptive-steering-qwen3-run` / `fk/init-ax-issuebench`). A generalization
  arm already has data.

**INCOMPLETE — GPU-killed, do NOT trust the numbers (per Farhan):**

- **Prompt/system-prompt baseline** (`origin/fk/init-prompt-control`). *Note the
  discrepancy honestly:* the branch **does** contain committed artifacts
  (`runs/20260903-093536_prompt-baseline-opinion_qwen3-8b/summary.md`,
  `runs/20260903-102805_…/comparison_summary.md`) and commits claiming *"prompt wins
  outright / prompt still wins bigger"* (`914aaf5`, `6c32c02`). **Farhan says these
  runs were interrupted (GPU killed) and are unreliable — so treat "prompt beats
  steering" as UNVALIDATED and re-run before citing.** Infra is real: `intervention ∈
  {steer,prompt,both}` in `experiment.run`, per-item paired `metrics.beat_rate` with
  item-bootstrap CI, prompts `DEFAULT_POS_SYS`/`DEFAULT_NEG_SYS` in
  `src/bias_steer/config.py`, config `configs/prompt_baseline_opinion.py`, handoff
  `docs/HANDOFF_prompt_control.md`, unit tests `tests/test_prompt_control.py` (3/3).
- **New (9-way v2.1) judge** (`origin/fk/calib-v2-gpu-run`, `fk/init-better-rubric`).
  Built and unit-tested (`src/bias_steer/judges/v2.py`, `tests/test_judge_v2.py`;
  labels in `docs/RUBRIC_v2.md`). Run so far only over a **40-item** calibration sheet
  (`d29e12d` "40/40 clean", `2336b74` "40/40 matched") — NOT the full steering battery,
  and NOT scored against human labels in-repo. Treat as not-yet-validated.

**The ~70% Cohen's κ:** Farhan confirms it is real but **never committed to code.**
It is not in the repo: `datasets/Calibration/calibration_v2_labeled.csv` carries only
judge columns (`judge_label_oai`, `judge_label_preview`) — **no human column.** The
κ-scoring harness is built and waiting (`scripts/kappa_from_csv.py`, header "BLOCKED on
human labels"; `scripts/build_annotation_pools.py`). **MUST-FIND item:** locate the
human label sheets Farhan used, commit them as a data artifact, recompute κ with a
bootstrap CI. (Do not alter existing repo docs; add the sheets as new files.)

### 2. State of the two unfinished runs, and what finishing costs

- **Prompt baseline.** How far it got: a full `both`-mode head-to-head on Qwen3-8B was
  *attempted* and partial artifacts exist, but per Farhan the GPU was killed mid-run so
  the completion is not trustworthy. **To finish:** re-run `configs/prompt_baseline_opinion.py`
  in `intervention=both` on Qwen3-8B to completion, MANDATORY with
  `config.enable_thinking=False` and `max_tokens ≥ 512` — the model defaults to thinking
  mode and at 128 tokens ~1 in 4 completions truncate mid-`<think>` and every metric
  silently computes on garbage (`docs/HANDOFF_prompt_control.md` §0.3; this is likely
  what made the prior run junk). Then extend to the same models as the fixed-add baseline.
  **Compute:** this arm *generates* and is judged → ~0.5–1 H100-day/model + OpenAI judge
  cost. **This is the pivotal experiment** (see §5).
- **New judge.** How far it got: 9-way v2.1 runs cleanly on 40 calibration items; the
  full-battery judging and the human-κ validation are not done. **To finish:** (a) run
  v2.1 over the full opinion/adaptive steering batteries so the paper's headline counts
  use the pinned judge; (b) collect/commit human labels and compute κ+CI. **Compute:**
  ~0 GPU (judging is API-side, a few hours of wall-clock + OpenAI cost); **~1 human-day**
  for the label collection — the binding human-hours cost.

### 3. The positive claim, and what a skeptical ICLR reviewer requires

**Central contribution (strongest defensible version of Farhan's evidence):**

> *A single difference-in-means "opinion" direction, added to the residual stream,
> steers open-weight LLMs from neutral to opinionated answering at high judged success
> (86–94% across seven models at a fixed dose; up to 95% on Qwen3-8B). An adaptive
> per-layer linear schedule advances the success–coherence frontier beyond a fixed
> scale. Crucially, the effect is a genuine steering effect — it survives a
> matched-norm/covariance random-direction control and a coherence/fluency gate — and
> it adds control beyond what a system prompt alone achieves.*

**What a reviewer must be given to believe each clause (the acceptance bar):**

1. **"High success" is not degenerate text.** A **coherence/fluency gate** on every
   reported cell (the c40 "INVALID coherence ceiling" proves this is the live risk).
   Right now coherence is only the judge's "nonsense/bad" bucket — a reviewer wants an
   independent gate (perplexity under an unsteered reference LM, or repetition/
   distinct-n-gram rate) so "95% opinionated" can't be "95% opinionated-sounding mush."
2. **The effect is the direction, not the dose or the prompt.** A **random-direction
   control matched on norm and covariance** at every dose, AND a **system-prompt
   baseline** (steering must beat simply asking). The system-prompt comparison is the
   single most important thing a *positive* steering paper is judged on — if a one-line
   prompt matches steering, the contribution collapses to "an expensive prompt."
3. **Breadth.** The high-rate result on **many models AND ≥1 modern model** under **one**
   judge and one protocol. Today the 90%+ is old-judge/old-models (Qwen1.5/Yi/gemma/
   Llama-2/3) and the modern result is Qwen3-8B-only — these must be unified.
4. **Adaptive > fixed is a real frontier gain, not noise.** The adaptive-linear vs
   fixed-add comparison at matched coherence, with CIs, on >1 model.
5. **Validated judge.** Real judge↔human κ with a CI (the ~70% recomputed and committed),
   plus the pinned v2.1 judge used for the headline counts.
6. **Statistical rigor.** Multiple seeds, item-bootstrap CIs on every rate, significance
   vs the random control. (The judge drifts ±1–2 labels/100 at temp 0 —
   `HANDOFF_prompt_control.md` §0.2 — so single-run margins inside that band are not results.)

### 4. Experiment inventory — by the reviewer objection each closes

Legend: **EXISTS** (runnable/finished in Farhan's tree), **PARTIAL** (infra there, needs a
run), **FIND** (must locate/obtain), **BUILD** (must write).

**OBJ-A "Your 95% is just broken text" — coherence/fluency gate.**
- (a) Do steered generations stay fluent at the doses that hit high opinion rates?
- (b) **PARTIAL.** The judge already emits a nonsense/bad bucket (c30 opinion bad=0,
  c40 INVALID), and collapse detection exists in the coeff-sweep history
  (`configs/exp/adaptive_add_linear_c40…`, commit `fe90279`). **BUILD** a dedicated
  gate: reference-LM perplexity + distinct-n-gram/repetition rate on every completion.
  Metric hooks live in `src/bias_steer/metrics.py`.
- (c) All doses × both arms × all models; report opinion-rate *and* coherence on the
  same axis; define the reportable dose as the max that stays under a fluency threshold.
- (d) Decisive: high opinion rate at a coherent dose ⇒ claim holds; if the only doses
  that hit ≥90% are past the fluency cliff (as c40 hints) ⇒ the headline drops to the
  highest *coherent* rate (likely c20–c30, ~63–95%). **Closes the #1 attack on any
  high-success steering claim.**
- (e) ~0 GPU if run on existing completions (re-judge/score saved text); hours.

**OBJ-B "It's the dose/prompt, not the direction" — random control + system-prompt baseline.**
- **E-rand:** matched-norm **and covariance** random direction at every dose. (b)
  **PARTIAL** — norm-matched randoms appear in older sweeps; covariance-matching must be
  added (a few LOC over cached residuals). (d) random ≈ 0 opinion shift while V_opinion
  ≫ random ⇒ specificity established. (e) ~0.5 H100-day/model.
- **E-prompt (PIVOTAL):** finish the killed prompt baseline (§2) — per-item paired
  `beat_rate` steer-vs-prompt with bootstrap CI. (d) steer beats prompt (CI excludes 0)
  ⇒ steering adds value, the paper stands. Prompt matches/beats steer ⇒ **reframe
  required** (see kill-criteria): pivot to "adaptive-linear beats *both* fixed-steer and
  prompt," or to controllability/compositionality prompting can't do. (e) ~0.5–1
  H100-day/model + judge cost.

**OBJ-C "Old models / old judge" — breadth + unified judge.**
- (a) Does the high-rate result hold on a modern model under the pinned v2.1 judge?
- (b) **PARTIAL.** Qwen3-8B vector + adaptive runs exist; re-judge the 7-model baseline
  and the Qwen3-8B runs under v2.1; add ≥1 non-Qwen modern model (**FIND** Llama-3.1-8B-
  Instruct, TL-compatible) so "modern" isn't confounded with "Qwen family."
- (c) Fixed-add + adaptive-linear + controls, one judge, all models.
- (d) Pattern holds on modern model ⇒ breadth closed. Modern model needs far higher dose
  or won't steer coherently ⇒ scope the claim to a model class (survivable, honest).
- (e) ~1–1.5 H100-days/new model.

**OBJ-D "Adaptive vs fixed is cherry-picked" — the method comparison.**
- (a) Does adaptive-linear dominate fixed-add on the success–coherence frontier?
- (b) **EXISTS (single model), extend.** `experiments/adaptive_vs_fixed/` +
  `runs/…adaptive-add-linear-c*` vs `…fixed-add…`. Add CIs and a 2nd–3rd model.
- (c) Sweep both methods over dose; plot opinion-rate vs coherence; compare frontiers.
- (d) Adaptive frontier dominates at matched coherence ⇒ a real methods contribution
  (this is the novelty beyond "steering works"). If they overlap within CI ⇒ drop
  adaptive to a secondary result and lead with the multi-model baseline.
- (e) ~0.5–1 H100-day/model.

**OBJ-E "Unvalidated judge" — human κ + pinned judge.**
- (a) Does the judge agree with humans? (b) **FIND** the human sheets (Farhan's ~70%
  κ), commit them, run `scripts/kappa_from_csv.py` with a bootstrap CI on κ; use v2.1
  for headline counts. (d) κ lower-CI in "substantial" range ⇒ judged rates credible;
  κ≈0.5 ⇒ lean on judge-light signals and soften behavioural claims. (e) ~0 GPU, ~1
  human-day.

**OBJ-F "Single-point numbers" — seeds + CIs + significance.**
- (a) Is every rate a distribution? (b) **PARTIAL** — `metrics.beat_rate` already does
  item-bootstrap; extend item-bootstrap CIs to every opinion-rate cell; ≥3 seeds
  (generation + judge); permutation/paired test vs the random control. (d) de-risks the
  tables. (e) ~0 GPU + rejudge cost; ~0.5 day analysis.

**OBJ-G "Does it generalize past one battery?" — transfer arm.**
- (a) Does the opinion direction transfer to AxBench/IssueBench-style prompts? (b)
  **EXISTS (data), needs writeup/controls** — `runs/20260901-…axbench/issuebench…`.
  Apply the same coherence + random + CI treatment. (d) transfer holds ⇒ breadth of
  *behaviour*, strong for main; fails ⇒ an honest boundary result. (e) ~0.5 H100-day.

### 5. Phased path (honest calendar + H100-days)

Assumes one engineer, H100 access, scarce human hours; GPU-days are single-H100 wall-clock.

**Phase 0 — Lock the spine (BLOCKING). ~2–4 days, ~1–2 H100-days.**
Re-run the killed **prompt baseline** to completion on Qwen3-8B (E-prompt, with the
`enable_thinking=False` / `max_tokens≥512` fix) and stand up the **coherence gate**
(OBJ-A) on the *existing* adaptive/fixed completions. These two decide whether you have
a positive paper at all (does steering beat prompting? is the 95% coherent?). Commit the
human-κ sheets if they can be located this week (OBJ-E).
*Exit: a validated steer-vs-prompt number and a coherent-dose success rate.*

**Phase 1 — Minimum viable ICLR submission. ~2–3 weeks, ~6–9 H100-days + ~1 human-day.**
Add covariance-matched random control (E-rand) and coherence gate to **all** headline
cells; re-judge the 7-model fixed-add baseline **and** the Qwen3-8B adaptive runs under
the **one pinned v2.1 judge** (OBJ-C); item-bootstrap CIs + ≥3 seeds everywhere (OBJ-F);
commit κ+CI (OBJ-E). *Exit: "steering controls opinionation at high, coherent, judged
rates across 7 models and one modern model, beats a system prompt, with a validated
judge and CIs."* A credible, if not yet flashy, main submission.

**Phase 2 — Competitive / defensible. +1.5–2 weeks, ~5–8 H100-days.**
Full **adaptive-linear vs fixed** frontier with CIs on ≥3 models (OBJ-D) — this is the
methods novelty; add **≥1 non-Qwen modern model** (OBJ-C); land the **transfer arm**
(OBJ-G) with controls. *Exit: a genuine methods contribution (adaptive schedule) on top
of a broad, controlled steering result.* An AC can champion this.

**Phase 3 — Strong. +2–4 weeks, ~8–12 H100-days.**
Layer/site sweep of where the opinion direction lives (cheap from cached residuals if you
**FIND** the tensors), compositional/multi-direction steering, a second behaviour beyond
opinion (reuse the refusal pipeline `src/bias_steer/refusal.py`), and scale to a 3rd
model family. *Exit: survives the "another model / another site / another behaviour"
reviewer reflex.*

**Honest total:** MVP ≈ **2.5–3.5 weeks** (Phase 0+1), competitive ≈ **5–6 weeks**,
strong ≈ **~2 months**. Weeks, not days. Binding constraints: (i) human-label collection
for κ, (ii) re-judging everything under one pinned judge (OpenAI cost + wall-clock), (iii)
locating any out-of-band residual/vector artifacts.

### 6. Kill-criteria / decision points

- **E-prompt:** if a one-line system prompt matches or beats steering on `beat_rate`
  (CI) on modern models → the "steering adds value over prompting" claim is dead; pivot
  the paper to **adaptive-linear beats both fixed-steer and prompt**, or to a
  control/compositionality axis prompting can't reach. Highest-probability reframe.
- **OBJ-A coherence gate:** if ≥90% opinion only occurs past the fluency cliff (c40 is
  already INVALID) → headline drops to the best *coherent* rate; do not report 95% naked.
- **OBJ-C modern model:** if Qwen3-8B/Llama-3.1 only steer coherently at low rates →
  scope the strong claim to the older class and frame the modern model as a boundary.
- **OBJ-D adaptive:** if adaptive ≈ fixed within CI → adaptive is not the contribution;
  lead with breadth instead.
- **OBJ-E κ:** if committed human κ ≪ 70% → soften every judged rate, report the judge's
  limits, and lean on the judge-light margin/transition signals.
- **Provenance hazard (not an experiment):** older headline runs used the OLD 3-way judge
  and 128-token cap; the adaptive Qwen3-8B thinking-mode truncation bug is real
  (`HANDOFF_prompt_control.md` §0.3). Every re-run must pin the v2.1 judge snapshot, set
  `enable_thinking=False`, `max_tokens≥512`, and persist raw completions, or you will
  re-manufacture invalid numbers.

### 7. The single highest-leverage thing to do today

**Finish the killed prompt-baseline run on Qwen3-8B (E-prompt), correctly configured,
and score coherence on the completions you already have (OBJ-A) in the same sitting.**
Rationale: (1) For a *positive* steering paper, "does the fitted direction beat simply
asking?" is the pivotal question — the whole contribution hinges on it, and it is the one
number you do not yet trustably have. (2) The infra is built and unit-tested
(`fk/init-prompt-control`, `intervention=both`, `beat_rate` with bootstrap CI); you are
completing a run, not building an experiment — just apply the `enable_thinking=False` /
`max_tokens≥512` fix that the prior (killed) run lacked. (3) Pairing it with the
coherence gate on the existing adaptive/fixed completions immediately tells you whether
the 95% at c30 is real or past the fluency cliff — the two answers together determine
whether your headline is "steering beats prompting at high coherent rates" (submit) or a
reframe. (4) Both are cheap (≤1 H100-day + judge cost; the coherence pass is ~0 GPU) and
neither is blocked on the human-κ collection. Do E-prompt + OBJ-A first; commit the κ
sheets in parallel if they can be found this week.

---

## Session addendum — 2026-09-23 (prompt-baseline run done + next experiments)

**Run completed:** `runs/20260923-030712_prompt-baseline-opinion_qwen3-8b`
(intervention=`both`, clean `20260903-105600` vector, `enable_thinking=False`,
n=200, seed=0). Result on the toy GPT comparison prompts:

- System-prompt baseline hits **100% both directions** (`prompt_pos` 200x opinionated,
  `prompt_neg` 200x neutral); steer 0.805 / 0.390; **`steer-only`=0** discordant -> the
  prompt *strictly dominates* fixed-add c=8. This is a **ceiling regime** (the eval
  prompts are trivially side-able), NOT evidence prompting beats steering in general.
- **Leakage ruled out:** the judge (`judge.py:88-92`) sees only `PROMPT: {question}` +
  `OUTPUT: {response}` -- the behaviour system prompt is never passed to it. The 100% is
  a genuine instruction-following result, not contamination.
- Fixed-add c=8 barely steers (opinion 150->161, **+11**) and **degenerates** (repetition
  loops the judge still labels "opinionated" -- coherence gate still unbuilt).

**Next experiments (denoted; not yet run):**
- **EXP-A -- IssueBench prompt-vs-steer.** Re-run the `beat_rate` contest on the
  `issuebench` loader (contested political/social prompts, arXiv:2502.08395) instead of
  the toy GPT set, to break the 100% ceiling and test the method's limits. Extract runs
  exist (`20260901-091212_extract-issuebench...`); no apply/eval yet. Verify
  `third_party/issuebench/` parquet is present or `scripts/fetch_issuebench.py --split
  sample`. *Closes: "ceiling regime / data can't test the limits."*
- **EXP-B -- Instruction-conflict / override.** Apply the OPINION vector while the system
  prompt says "stay strictly neutral" (and the mirror). Tests whether steering overrides
  a *competing* instruction -- the regime prompting structurally can't win (locked or
  adversarial system prompts). Small harness change: the steer arm uses the *opposing*
  system prompt. *The strongest "steering does what prompting can't" test for a
  directly-instructable behaviour.*
- **EXP-C -- graded/continuous control:** OUT OF SCOPE per FK (2026-09-23).

**Blocking task before EXP-A -- human-readable run outputs.** `logs/eval.txt` is one flat
file AND omits the prompt arms entirely (`logs.py` `eval()` hardcodes
INITIAL/STEERED_POS/STEERED_NEG). Redesign the per-run output format before running A.
DONE (commit `3b10b29`): per-condition files under `logs/by_condition/`.

---

## Findings log — 2026-09-23 (EXP-A + extraction confound)

- **EXP-A (transfer: toy vector -> IssueBench debug, n=150), run `20260923-072454`.**
  The prompt CEILING BROKE: `prompt_pos` 86% opinionated, `prompt_neg` 91% neutral
  (vs 100/100 on the toy set) -> a real contest. But the toy opinion vector **fails to
  transfer**: `steered_pos` opinionated 84->50 (+7 nonsense), opinion quality good=13/
  **bad=47**; `beat_rate` prompt beats steer both ways (opinion Δ=-0.527 [-0.593,-0.453];
  neutral Δ=-0.293 [-0.367,-0.220]), `steer-only`≈0. Neutral pole partially transfers.
  Judge leakage ruled out (`judge.py:88-92` sees question+output only).
- **Natural-variation extraction on IssueBench is CONFOUNDED**, run `20260923-080817`,
  buckets 160 neutral / 40 opinionated. The 40 opinionated prompts are ~95% ones whose
  TEMPLATE embeds a stance ("X being a bad thing", "... being morally justifiable") —
  IssueBench pro/con-polarity items. The direction fits PROMPT FRAMING, not behaviour.
  -> Do NOT use that vector as a result; keep only as a negative-control comparator.
- **Fix built:** forced-contrast extraction (`contrast_mode="forced"`,
  `configs/extract_issuebench_forced.py`) — see Aryaman prior-art below.

## Prior art in-repo — Aryaman's manual `prompt_classification` (2025)

Branch `origin/aryaman_experimentation`, `experiments/prompt_classification/
{opinionated,neutral}_manual.txt` (later semi-automated as `{opinionated,neutral}.txt`
per model; `prompt_classification_oai_*` = OpenAI-judged). Aryaman built the
opinionated-vs-neutral contrast by **hand-curating (prompt, response) examples into two
groups** — opinionated = the model took a side ("Procrastination is generally more
effective…"; forced-choice + BBQ answers), neutral = the model hedged ("As an AI language
model, I cannot make subjective judgments…"). The vector = mean-activation difference of
the two hand-picked groups. Same *intuition* as forced-contrast (control what counts as
each pole) but realized differently:

| | Aryaman manual (2025) | Natural-variation (auto) | Forced-contrast (new) |
|---|---|---|---|
| grouping | hand-picked examples | judge labels default-prompt outputs | induced by POS_SYS/NEG_SYS |
| prompts per group | DIFFERENT | DIFFERENT | **SAME (paired)** |
| content confound | yes (+ "As an AI…" boilerplate in neutral) | yes | **no — content cancels** |
| scale / objectivity | small, subjective | automated | automated, balanced by design |

**The difference:** Aryaman's two groups (like natural-variation's) use DIFFERENT prompts,
so the diff-of-means mixes opinionation with topic/phrasing differences — and his neutral
set is largely "As an AI language model…" refusal boilerplate, so the direction risks
encoding *that* rather than neutrality. Forced-contrast holds the prompt identical and
varies only the inducing instruction, so the difference isolates the behaviour. Aryaman's
instinct was right; forced-contrast is the confound-free realization of it.

## Open tasks (to resolve)

- [ ] **Run forced-contrast extraction** (`configs/extract_issuebench_forced.py`); confirm
      balanced ~200/200 buckets; compare its vector vs the toy transfer vector AND the
      confounded natural vector on IssueBench.
- [ ] **Forced-mode unit test** for `_extract_vector` (`contrast_mode="forced"` -> balanced
      buckets, no judge call in train phase).
- [ ] **Benchmark construct-alignment (RESEARCH_CONTRACT §12 decision).** IssueBench measures
      *issue-writing bias*, not pure "opinionate-vs-hedge on answerable questions." Whether to
      add a construct-matched eval (OpinionQA / GlobalOpinionQA) is drafted as a proposal for
      the team in [`docs/PROPOSAL_opinionqa_construct_eval.md`](docs/PROPOSAL_opinionqa_construct_eval.md)
      (on `main`, commit `a1ddedb`) — team to decide; §12 lists "another benchmark" as NOT a
      reopening trigger, so the default is *no*. Do NOT wire in a new dataset unilaterally.
- [ ] **EXP-B — instruction-conflict / override** (still pending; small harness change).
- [ ] Mark the natural IssueBench vector (`20260923-080817`) as confounded in any writeup.

## For the final paper (running list of things to produce)

Living checklist of artifacts/claims we intend to include. Add as they solidify.

- [ ] **Headline claim** — stated per `RESEARCH_CONTRACT.md` (framing centralized; do not
      coin here). Fill once the forced-contrast + baseline results are in.
- [ ] **System-prompt baseline table** (steer vs prompt, per-item `beat_rate` + CI) on a
      real dataset where the prompt is NOT at ceiling — IssueBench gives this (86/91%).
- [ ] **Extraction-method comparison** — natural vs forced-contrast vs (historical) manual
      curation; make the prompt-framing confound and its fix an explicit methods point.
      *Candidate Related-Work/Methods sentence: forced-contrast as the principled successor
      to early manual prompt-grouping (Aryaman 2025, in-repo).*
- [ ] **Coherence gate** — independent fluency metric on every reported cell (still unbuilt).
- [ ] **Transfer / generalization** — does an opinion direction move across prompt
      distributions? (toy->IssueBench transfer already shows a negative for the toy vector.)
- [ ] **Judge validation** — human κ + CI, pinned judge version.
- [ ] **Per-example distributions** (3x3 confusion), not just means (2026 steering-claim bar).
- [ ] **Side-effect audit** (capability + safety) for any headline intervention.
