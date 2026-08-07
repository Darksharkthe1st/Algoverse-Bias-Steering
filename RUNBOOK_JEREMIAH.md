# RUNBOOK — Jeremiah (owner: fault-susceptibility study · measurement geometry)

> Read `docs/THE_CORRECT_PROBLEM.md` first — four pages, and it's the point of
> the project. Then `PAPER_FRAMING.md` before writing any paper text.

**You own a workstream, not a task list.** The injected-fault study below is, on
current evidence, the paper's central contribution — an external prior-art scan
concluded that without it we have "a blog post with a bibliography," and with it
we have a measurement nobody in the field has made. That workstream is yours:
its design, its controls, its figures, and the call on what it can and cannot
claim. Nobody else on the team is going to tell you how to build it, because
nobody else has thought about it harder than you're about to.

Two consequences worth stating plainly. **You have design authority** — if you
think the fault taxonomy below is wrong, change it and say why; the version here
is a starting hypothesis, not a spec handed down. And **you have veto power over
your own results** — if the study doesn't support a claim, it doesn't go in the
paper, and that call is yours to make and defend.

Nothing in your workstream needs a GPU.

---

## Workstream 1 (yours) — the fault-susceptibility study

### The question

Last year's refusal experiment produced a clean, convincing table — 1 unsafe out
of 99, then 27 out of 99 after "steering." It measured nothing. A `(4096,)`
vector was indexed as if it had layers, so a single number was broadcast across
all 4096 dimensions: a DC offset, not a direction. Nothing crashed. Nothing
warned. The table just looked like a finding, and it survived a year.

The question nobody has answered: **which silent faults manufacture which
artifacts, at what magnitude, and would any standard reported statistic catch
them?**

### The design (starting hypothesis — change it if you find better)

Take a pipeline that works. Inject one fault at a time. For each, measure what
comes out.

| fault | what it does | expect |
|---|---|---|
| scalar broadcast | 1-D vector indexed by layer | ? |
| wrong model's vector | ordering mismatch between two lists | ? |
| extraction folding | judge `none` counted as a behavior class | ? |
| unseeded split | train/test membership drifts between runs | ? |
| semantic mislabel | marginals reported under transition column names | ? |
| cumulative artifact | records from a previous model left in the file | ? |

Per fault, the numbers that matter: the effect size it manufactures out of
nothing; whether it fakes a **positive** or a **null**; the fraction of result
cells that change; and whether any commonly-reported statistic — means,
per-example distributions, confidence intervals — would reveal it.

The output is a **susceptibility profile**. That converts a claim we cannot
support ("maybe published steering nulls are bugs") into one we can measure.

### Why this is load-bearing rather than nice-to-have

The prior-art scan found our first framing was partly taken: `arXiv:2607.02586`
("Auditing the Audit," Jul 2026) already defines "silent" as
invisible-in-the-reported-numbers and ships a disclosure protocol. What it does
**not** cover — and what nobody has measured — is silent *broadcast*: shapes
that are legal but semantically wrong. That gap is the one you'd be filling.

### Where to start

The scalar-broadcast case, because we have the real example to check yourself
against: `experiments/past_vecs/calculated_refusal_vecs/*.pt` are the actual 1-D
tensors, and Logs 210–214 are the actual output. Reproduce the artifact
deliberately, then you know your injection harness works before you trust it on
anything else. That's a positive control, and it's the same discipline the rest
of the project runs on.

### Open design questions that are yours to decide

1. What counts as "a pipeline that works"? Ours after Farhan's refactor, or a
   third-party one so the result isn't about our code?
2. How many faults is enough? Six is a guess.
3. Do you inject into the real archive, or synthesize a clean baseline where
   ground truth is known exactly? There are real arguments both ways.
4. Does the profile generalize, or is it specific to difference-in-means
   steering? What would you need to run to find out?

## Workstream 2 (also yours) — measurement geometry

The numbers that tell us whether any geometric claim means anything.

- **Per-layer norm profiles.** Started: the chart under *Verification* on the
  dashboard, from `dashboard/data/vector_norm_profiles.json`. It already
  produced a real finding — norms span 2–3× on gemma but 600–1391× on
  Qwen/Yi/Llama, which is why "all-layer" steering is effectively late-layer on
  most families, and why per-model coefficients never stabilised.
- **The extraction-variance floor, which does not exist yet.** How much does a
  direction move when you resample the contrast set? We have *no* estimate: the
  two archived Qwen1.5-7B vectors are byte-identical copies. **Until this
  number exists, no cosine we report means anything** — a cross-direction cosine
  of 0.35 is uninterpretable without knowing whether re-extracting the *same*
  direction gives 0.97 or 0.60. This is a bootstrap over contrast-set redraws
  and it gates the entire geometry section.
- **Unit-normalization policy.** Given the norm profile, any cross-model depth
  comparison must normalize first or it mostly recovers the norm plot. Working
  out the right normalization and defending it is a real methods contribution.

This workstream is where the project's long-horizon direction lives — see
`docs/RESEARCH_PROGRAM_GEOMETRY.md`. You'd be building its foundation.

## Shared work (not yours alone)

**Gold-set annotation, with Edward.** ~150 archived responses under the six-way
rubric, independently, blind to arm, targeting per-category Cohen's κ ≥ 0.70.
Two phases, and the difference matters. **Calibration** comes first: ~20-30
responses, argue about the hard cases, change the rubric as much as you want.
That is instrument design and you should do it. **Then the scored pass**: a
disjoint ~150, labeled independently and blind to arm, rubric frozen by commit
hash, touched once. Don't discuss cases until both passes are done — agreeing
because you talked is not agreement. Disagreements are data about rubric
ambiguity, not mistakes.

**The screen is defined in `docs/RUBRIC_v2.md` — the single canonical
source. Do not work from a copy.** It is not frozen yet; the ordering and
the category count are live questions for Saturday, and your read on whether
eight classes can hold κ ≥ 0.70 is worth more than anyone's guess.

The one line that must survive review: **stance-factual vs stance-evaluative**.
That split is what 2025 collapsed, and its absence is the leading explanation
for the CrowS transfer failure.

**Heads-up:** the stored text has a `PROMPT:`/`OUTPUT:` scaffold, chat control
tokens in 85–93% of responses, and a truncated prompt echo. Judge the model's
answer, not the scaffolding.

**Paper.** You're writing lead. Create `paper/` with a 5-page `.tex` skeleton
early — no `.tex` has ever existed in this project, and "write it in week 4" is
the line most likely to fail.

## Getting up to speed (interleave with the work, don't front-load)

1. `docs/THE_CORRECT_PROBLEM.md`
2. Arditi et al., arXiv:2406.11717 — the paper ours descends from. The math is
   what Farhan described: subtract two mean vectors, add the result back.
3. 3Blue1Brown's transformer/attention chapters — so "residual stream" means
   something physical.
4. `docs/VERIFICATION_2026-08-07.md` — a worked example of this kind of checking.
5. `scripts/verify_2025_results.py` — read before writing any recount of your
   own. It encodes two traps: the pickles are **cumulative** (log_236 holds 672
   records, all seven models; only the last 96 belong to the named model), and
   the denominator is **96**, with arrow-named columns being **per-arm
   marginals, not transitions**.

## Two standing rules

**Never quote a number without its denominator and its judge version.** Most
mislabeled results anywhere are a numerator on the wrong denominator, and this
archive has already produced one.

**If something looks too clean, check it.** The retracted result looked great.
Ask the question that would have caught it a year earlier — *what shape is that
tensor?* — and ask it out loud, including when it feels too basic. That bug
survived a year because nobody did.
