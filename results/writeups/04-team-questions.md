# Questions for the team meeting — Fri 2026-08-21 (evening)

Ordered so the ones that block work come first. Each says why it matters, so it
can be asked quickly.

---

## Blocking — cannot proceed without an answer

### 1. Farhan: what exactly is "the average vector" method? (blocks Experiment 2)

Experiment 2 is defined as "give each layer its own vector, instead of Farhan's
average." But reading `src/bias_steer/steering.py:apply_resid_pre_add`, the bias
path **already** gives each layer its own vector — `vector[layer]` at line 122.
What gets divided by the layer count is the *coefficient* (`coeff / n_layers`),
not the vectors.

So one of these is true and it changes the whole experiment:

- (a) "Average" meant the `coeff / n_layers` strength split, and per-layer
  application already exists — in which case Experiment 2's real content is
  *porting per-layer application to the **refusal** path*, which currently uses a
  single `(d_model,)` direction applied at one layer. That is still novel, and
  the apply code already exists, so it is cheap.
- (b) Farhan actually collapsed the per-layer stack into one averaged vector
  somewhere else, and we should be comparing against that.

**Do not build the comparison until this is settled** — otherwise we benchmark
against a method nobody used.

### 2. Which extraction convention should bias directions use?

The repo has two, and they are not interchangeable:

| | reads | when | used by |
|---|---|---|---|
| `capture_mean` | mean over the **response** tokens | after generating | bias/opinion work |
| `capture_prompt_positions` | last few **prompt** tokens, no generation | before generating | refusal repro (Arditi) |

Experiment 1 is a bias study, so `capture_mean` is the default. But if we ever
want to compare a bias direction against the refusal direction (JZ-3), the two
must be extracted the same way or the cosine between them is meaningless.

**Ask: do we commit now to extracting bias directions the refusal way, so JZ-3 is
possible later?** Cheaper to decide before the first run than to re-run.

### 3. Lambda scheduling — who else needs the box tomorrow?

`PROJECT_STATE.md` says G1 is the sole gate, is blocked on the box, and blocks
everything downstream. If Edward is running G1, our runs queue behind it. Need to
know the order and whether we can run concurrently.

Also: Jeremiah has never SSH'd to Lambda (has used RunPod). Someone should
confirm the access path — is it a shared team account, a shared key, or a
per-person key?

---

## Design — affects what we can claim

### 4. The intersectional categories may not be scoreable

`Race_x_gender` and `Race_x_SES` resolve at only ~66%. The reason is structural,
not a bug: both answers share the stereotyped race (e.g. "The Black woman" vs.
"The Black man"), and BBQ's `stereotyped_groups` names only the race — so the
field cannot say which answer is the biased one. Our code refuses to guess and
counts those rows as unresolved.

This matters because the intersectional sets were going to be a free sanity
check: if `Race_x_gender` lands between `Race_ethnicity` and `Gender_identity` in
the clustering, the geometry is tracking something real.

**Ask: does anyone know BBQ's intended handling for the intersectional subsets?**
Otherwise we run on the 8 base categories and treat intersectional as a bonus.

### 5. Should the extraction-variance floor be its own contribution?

`RUNBOOK_JEREMIAH.md` says this number gates the entire geometry section and has
never been measured — the two archived Qwen1.5-7B vectors that could have given
it are byte-identical copies. Experiment 1 has to measure it anyway, so we get it
for free.

**Ask: is that a result the paper wants in its own right,** or just an internal
control? It is the kind of thing a reviewer asks for and nobody has.

### 6. Do we add CrowS-Pairs as a second source?

All 10 BBQ categories share one format, so the within-BBQ comparison is clean.
The moment we add politics from a different file, format and topic get tangled
and a "politics is distinct" result could just mean "that file looks different."

The fix is to never add exactly one topic from a new source. CrowS is already in
the repo and has stereotype categories — adding two topics from it would let us
test directly whether directions cluster by topic or by source.

**Caveat found in the code:** `datasets.load_crows_questions` warns the committed
CrowS CSV is anonymized and has **no `bias_type` column**, so per-category splits
are not derivable from it. Would need the full CrowS release.

---

## Scope and process

### 7. Is this workstream in the paper, or after it?

The project is frozen (`RESEARCH_CONTRACT.md` §12) and `PROJECT_STATE.md` lists a
bias taxonomy under "**Does not block the paper**". If our results are meant to
appear in the NeurIPS submission, that needs a dated §12 amendment. If it is the
next paper, we have more freedom and less deadline pressure.

Worth asking plainly, because the answer changes how much control work is
required before anything is reportable.

### 8. Model choice for our runs

Proposal: develop on `qwen-1.8b` (fast, ungated, most historical comparators),
confirm on `qwen-7b`, re-run headline numbers on the pinned `qwen3-8b` so they sit
next to the team's. **Any objection?**

Also: does anyone hold a Hugging Face token with **gemma** access approved?
gemma is the one model family with a flat per-layer norm profile (2-3x spread vs.
600-1391x on Qwen/Yi), which makes it the most informative second family for
Experiment 2. Not needed for Experiment 1.

### 9. Edward: where are the 1-D refusal vectors?

`RUNBOOK_JEREMIAH.md` points at `experiments/past_vecs/calculated_refusal_vecs/*.pt`
as "the actual 1-D tensors" from the retracted result. **That path does not exist
in the repo.** The only `.pt` files are under
`experiments/past_logs/past_vecs/official_refusal_vecs/`, and those are 3-D
`(n_positions, n_layers, d_model)` — Arditi's, correctly shaped.

Not urgent, but the fault-susceptibility workstream in the runbook depends on
having the actual broken artifacts to reproduce.

### 10. Does the timeline still hold?

`PROJECT_STATE.md` (Aug 17) says numbers freeze **Mon Aug 24** and the deadline is
**Fri Aug 28 AoE**, with G2 scheduled for Thu Aug 20 — which has now passed with
G1 still unrun. Confirm whether those dates are still real, since they determine
whether our work is for this paper or the next.

---

## Things to report to the team (not questions)

- The bias-taxonomy analysis layer is written, unit-tested (34 tests), and
  committed on branch `jz/bias-taxonomy`. Full suite: 156 passed.
- BBQ can be scored **without an LLM judge** — it is multiple choice, and on
  ambiguous items the correct answer is the unknown option, so choosing the
  stereotyped group is objectively wrong. No judge version attaches to any number
  from this workstream. Given how much time the team has spent on judge rubrics,
  this is worth saying out loud.
- Ambiguous-item coverage across all 10 categories is **80.4%** (20,750 of
  25,814 rows) after fixing the group-matching. Unresolvable rows are counted and
  reported, never guessed.
