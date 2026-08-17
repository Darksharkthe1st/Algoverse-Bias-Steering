# RESEARCH CONTRACT — supersedes all prior framing

**Status: PROPOSED, for ratification Tue 2026-08-18.** On ratification this file
replaces `PAPER_FRAMING.md`, `docs/2026-08-01_sprint_plan.md` and
`docs/2026-08-02_sprint_proposal.md` as the single source of truth. Those three
get a one-line pointer here and stop governing agents or people.

Everything below was verified this session against primary sources. Claims that
could not be verified are marked **UNVERIFIED** and must not be relied on.

---

## 0. What verification changed

| Inherited belief | Verdict |
|---|---|
| The bibliography may be AI-hallucinated | **FALSE — 19/19 arXiv IDs are real.** Control test (`2607.99999` → 404) confirms the checker reports failures |
| Fafuła 2607.17427 "explicitly requests this experiment" | **FALSE.** No "soft refusal", "neutrality", "both sides", "benign" anywhere in it; no future-work section; it studies decision disposition on 60 equities and states the task elicits *no refusals*. **Must not appear as motivation** — one reviewer opening the PDF costs the paper |
| Soft-vs-hard dissociation is novel | **LARGELY SCOOPED.** Joad et al. 2602.02132 ran DiM over **eleven** refusal categories (incl. CoCoNot-Indeterminate, XSTest-OverRefusal), found them geometrically distinct **and behaviourally collapsed onto "a shared one-dimensional control knob"** |
| "Soft refusal" is our coinage | It re-coins **CoCoNot indeterminate/subjective** (Brahman et al., NeurIPS 2024 D&B). Cite it; do not re-coin |
| Qwen3.5/3.6/3.8-27B are byte-identical, differ only in post-training | **BOTH FALSE.** Architecture *is* identical (1199 tensors, same shapes, 27,781,427,952 params) but dtypes differ (3.5 stores 96 tensors F32). No `-Base` checkpoint exists at 27B, releases are 6 months apart, and no card claims shared pretrained weights. **C3 has no control** |
| Qwen3.5-27B is a dense transformer | **FALSE.** Hybrid: 48 Gated-DeltaNet linear-attention + 16 full-attention layers, a 27-layer vision tower, weight prefix `model.language_model.layers.N`, `head_dim` 256 ≠ 5120/24. Residual-stream DiM survives (hooks are architecture-agnostic); head-level analysis does not |
| Venue assumptions were invented | **CORRECT, to the day.** Aug 29 AoE is NeurIPS's recommended common workshop deadline |
| Cosine/orthogonality shows functional separation | **NO.** Wollschläger 2502.17420 defines *representational independence* precisely because orthogonality does not imply independence under intervention |

**The one live gap Joad et al. leave:** all eleven of their categories are
*non-compliance* — the model declines the task. Our construct is the model
**complying at length while declining to commit**. That is not obviously in their
taxonomy. It is a narrow gap. It is the only one we have.

## 1. The question

> Refusal categories are geometrically distinct yet behaviourally collapse onto
> a shared one-dimensional control (Joad et al. 2602.02132). **Does stance
> avoidance on benign evaluative prompts sit on that same control, or is it a
> separate one?**

## 2. Headline claim, and its falsifier

**Claim.** Across a dose sweep, the four dose–response curves
`{d_stance, d_harm} × {benign-evaluative battery, harm battery}` do **not**
collapse onto a single one-parameter family after threshold alignment.

**Falsifier — pre-registered, and we expect it may fire.** They do collapse: one
shared sigmoid with per-battery thresholds fits all four within bootstrap CI.
That outcome is the paper — it extends Joad et al.'s shared-knob result to a
behaviour class outside their taxonomy.

**Why curve shape and not a 2×2.** Reviewer analysis this session: a
selectivity ratio ≠ 1 arises *generically* from one shared axis sampled at two
baseline rates, so a 2×2 cannot discriminate the hypotheses. An equivalence test
on off-diagonal cells needs **n ≈ 690 items per cell** for 80% power at a ±5-point
margin; we have 296. **Curve shape is a within-item test powered at n ≈ 150–300.**
It is the only analysis available to us that separates "two mechanisms" from
"one knob sampled at two thresholds."

## 3. Minimum experiment set

| | Experiment | Gate |
|---|---|---|
| **E0** | Arditi replication on the primary model — extract `d_harm`, confirm ablation suppresses harm refusal | **Kill gate.** No replication → no paper |
| **E1** | Extract `d_stance` by DiM on benign evaluative items. Selection on **on-target metrics only**, held-out split, layer/position committed **by hash** before any harm cell is scored, executed by someone not running the harm arm | Selection-contamination guard |
| **E2** | Dose sweep, both directions × both batteries, **k = 5 generations per item** | The headline |
| **E3** | Controls: covariance-matched random direction, **norm-matched uniform DC offset**, wrong-layer | Without these, nothing is interpretable |

The DC-offset control arm is the one piece of the 2025 disaster worth keeping:
it converts our own bug into a named null.

## 4. Cut from this submission

C3 post-training trajectory (no shared-base control exists) · SAE/Qwen-Scope
work (exists only for 3.5-27B, cannot span a trajectory) · the ACE / cone /
gradient technique survey · the bias taxonomy across political/racial/religious
categories · the byte-fallback perturbation arm · **Fafuła as motivation** ·
any claim about bugs in the published literature · any 2×2 selectivity ratio as a
primary statistic.

## 5. Measurement gate — must pass before any curve is interpreted

The 8-way cascade **fails as designed** and is replaced.

- *[verified by simulation this session]* Five of eight categories have expected
  count < 7 at n = 150. Hard refusal has a **53% chance of landing below 5 items**.
  Cat 0's 95% interval is [0.00, 1.00].
- **This is not a sample-size problem.** At n = 2400, median κ_j(soft) is still
  0.52 and κ_j(non-engagement) 0.26. The ceiling is set by the 4↔5 and 6↔7
  confusions. To reach κ_j(soft) ≥ 0.70 you need P(says non-engagement | truly
  soft) ≤ 0.05.
- **"Per-category Cohen's κ" is not a defined statistic** for 4 raters. Cohen's is
  two-rater; 4 raters gives 6 pairs × 8 categories with no aggregation rule.
- A per-category κ gate **systematically punishes the rare categories** whose
  reliability matters most (Feinstein & Cicchetti's prevalence paradox: identical
  raters, identical agreement, κ falls 0.64 → 0.12 as prevalence goes 0.50 → 0.02).

**Replacement.**
1. **Two facets, not one cascade.** Facet V = validity (readable / degenerate),
   scored first and independently. Facet B = behaviour, scored **only on
   readable** items. This removes the confound where degraded output is scored as
   abstention, and it is why the 2025 "garbage judged neutral" pathology existed —
   the label space had nowhere else to put it.
2. **Facet B collapses to 3–4 categories:** named a side / declined to commit /
   declined the topic. Merge 5 into 4. Drop 6-vs-7 — it asks about the *world*,
   not the text, which is a category error in a behavioural screen.
3. **Fleiss κ_j** (or Krippendorff's α if raters miss items — one annotator has
   exams), reported with **bootstrap CI**, gated on the **lower bound ≥ 0.70** for
   **only the two load-bearing categories**, pre-registered. Everything else
   descriptive, plus **Gwet's AC1** as a prevalence-robust sensitivity, plus the
   full 4-rater confusion table.
4. **Stratified enrichment** for rare categories — do not rely on natural
   prevalence.

Under the 8-way scheme only 33% of items get a unanimous 4-rater label and 23%
have no 3-way majority. Under 4 categories that becomes 80% / 2%.

## 6. Protocol and correctness checks before a run is trusted

- [x] **Shape guard** — `assert_steering_shape` rejects 1-D vectors; regression
      test added. *Committed on this branch.* Every archived `.pt` is still 1-D,
      so this was one `torch.load()` from recurring.
- [x] **`get_repo_root()` sees worktrees** — was `.is_dir()`, now `.exists()`.
      Suite went 22/29 → 31/31. *Committed on this branch.*
- [ ] **Fix artifact persistence FIRST.** 12 of 13 campaign runs wrote a 167-byte
      log and nothing else — no generations, no vectors, no `results.csv`;
      `runs/index.csv` has 1 row for 13 runs. **Generating before this is fixed
      means generating twice.**
- [ ] Unit-normalise directions; report norms separately.
- [ ] Pin the judge model to a dated snapshot; **k ≥ 3 judgments per item**, keep
      the majority, record per-item agreement. Currently one call, unpinned alias.
- [ ] Minimum-n guard on DiM buckets (2025 ran 75/25 imbalanced).
- [ ] Write the ablation op. **It does not exist.** `src/interventions.py` has
      never existed in any ref — a planning doc invented it.
- [ ] Fill `docs/PREREG.md` (79-line skeleton, every field blank) and commit the
      hash before E2.

## 7. Model set

| Role | Model | Why |
|---|---|---|
| **Primary** | `Qwen3-8B` | Dense `Qwen3ForCausalLM`, TL adapter `qwen3.py` exists, uniform full attention |
| **Fallback** | `Qwen1.5-7B` | The pipeline has 13 successful runs on it; 6.1 min per ~400 generations |
| **Gated upgrade** | `Qwen3.5-9B` or `-27B` | Only if it generates coherent chat-templated text under a forward hook **by end of Aug 20**. Load via `TransformerBridge.boot_transformers()` — the `qwen3_5.py` adapter *raises* on the published `ForConditionalGeneration` class |

TransformerLens **3.7.2 shipped 2026-08-15** — alive, 142 adapters,
`TransformerBridge` is the canonical v3 interface. Keep it for the fallback path;
prefer **plain HF forward hooks** for anything hybrid/multimodal, since the method
only needs residual-stream reads and adds.

**GPU is not the bottleneck** — 4.3–7.5 min per ~400 generations on a 40GB A100.
Human labels are. Lambda expiring Aug 28 removes *re-run* capacity, so the last
day to discover a generation bug is **Aug 24**.

## 8. Post-training checkpoint comparison

**Cut.** The premise fails: no shared base weights, no `-Base` at 27B, 6 months
apart, dtypes differ. It would be an uncontrolled comparison sold as a control.

## 9. Venue

**Recommend InterpScience — "Interpretability as a Science", Aug 28 AoE, Sydney,
long track ≤ 9 pages,** non-archival. Its scope *is* what standards and evaluation
criteria the field should adopt for measurement, causal claims and falsifiability
— which is exactly this paper. Nine pages lets us show the controls rather than
assert them.

Alternatives, all verified: **Interp4Discovery** Aug 29, ≤5pp, double-blind —
the inherited target, but its framing is "what do models know that we don't"
(protein structure, climate, astronomy); fit is mediocre and 5pp is too tight.
**JUDGe "Can We Trust the Judge?"** Aug 29, 6/4/2pp — has a **2-page junior
spotlight** worth a separate submission for the newest member. **ATTRIB** Sept 1
— latest deadline, but reciprocal reviewing (reviews due Sept 22) and weak fit.
**XAI4Science** Aug 29 — anonymity *optional*, which would dissolve the public-repo
problem.

**InterpScience forbids papers under concurrent workshop review. Pick one.**

**Anonymity:** a public repo is *not* a desk-reject risk — NeurIPS states
non-anonymous preprints do not cause rejection. A GitHub username or link *inside
the PDF* is. Interp4Discovery explicitly says to search the manuscript for names
and GitHub/HF usernames, and recommends `anonymous.4open.science`.

## 10. Execution order with kill gates

| Date | Must close | Kill condition |
|---|---|---|
| **Aug 17** | Artifact-persistence bug fixed; merge this branch; codebook v3 (two facets, 4 categories) drafted | — |
| **Aug 18** | Ratify this contract. PREREG filled + hash committed | Not ratified → stop, do not run |
| **Aug 19** | **Blinded stimulus pools in annotators' hands.** Generate on the *validated* pipeline, not the 2026 model | Missed → human component is dead |
| **Aug 20** | E0 Arditi replication passes. 2026 runtime gate | E0 fails → no paper. 2026 model not generating → drop it permanently, ship on the fallback |
| **Aug 21–22** | Calibration r1 + r2; E1, E2 running | κ lower bound < 0.70 after **exactly two** rounds → report Facet V + extractor only, no third round |
| **Aug 24** | **Numbers freeze.** Last day with re-run capacity | — |
| **Aug 26** | Red team | — |
| **Aug 28** | Submit | — |

## 11. Ownership — non-overlapping

| Person | Owns | Hard boundary |
|---|---|---|
| **Farhan** | Runtime, E0, E1, E2, E3, artifact persistence | Does **not** touch `d_stance` selection (contamination guard) |
| **Jeremiah** | Codebook v3, annotation protocol, Fleiss κ_j + bootstrap CI + Gwet AC1 in `scripts/kappa_from_csv.py` (currently 2-rater Cohen only, ~60 lines needed by Aug 22) | Does **not** run the harm arm |
| **Edward** | This contract, PREREG, venue, related work, the 296 items' prompt-level attributes — **delivered before travel, zero downstream dependencies** | Assume unavailable from Aug 20 |
| **Aryaman** | `d_stance` extraction and the ablation op, executed blind to the harm arm | Annotation only 3h, asynchronous |
| **All four** | Independent labels | Pre-register **3-rater Fleiss as primary**, 4th as robustness |

Blinding to arm is not blinding to hypothesis — all four annotators authored it.
State that as a limitation.

## 12. Repository changes for one source of truth

1. **Merge `main` ← `team-kit`.** Verified safe: purely additive from the merge
   base (main +71 files, team-kit +47), one `.gitignore` conflict, resolve as the
   union. Nothing is destroyed.
2. **Merge this branch** (`fix/steering-shape-guard`) — shape guard, worktree fix,
   this contract.
3. On merge, reduce `PAPER_FRAMING.md`, `2026-08-01_sprint_plan.md` and
   `2026-08-02_sprint_proposal.md` to pointers here. Keep `REVIVAL_AUDIT.md`,
   `PRIOR_ART_2026-08-07.md` and `VERIFICATION_2026-08-07.md` as history.
4. `docs/SOURCES_OF_TRUTH.md` currently, **publicly**, indexes material marked
   do-not-share (BASI dossier, harvest notebooks, Blueprint vault paths). Trim
   those rows.
5. Fix `pytest tests/` registry-teardown poisoning (7 failed / 38 passed as a
   suite; 45/45 per-file).
6. Delete or clearly mark the 28 stale remote branches.

## 13. Genuinely unknown

- Whether **E0 reproduces Arditi on any model we can actually run**. Nobody has
  tried. Every number in every plan so far is simulated or assumed.
- Whether `d_stance` is extractable at all on an instruction-tuned model below the
  coherence cliff.
- Whether the 16×A100 cluster materialises. **Assume not.**
- Whether Qwen3.5-9B loads and generates under a hook. That is the Aug 20 gate.
- Prevalence of each behaviour class in unsteered output. **Measure on day 2,
  n = 100** — it sets the maximum attainable effect and therefore which paper is
  possible.
