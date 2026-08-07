# RUNBOOK — Farhan (first author · pipeline owner)

> Read `PAPER_FRAMING.md` before writing any paper text. Disagree → PR that file.
> Claim status lives in the dashboard's **Claim ledger**; update it in the same
> PR as any change to what we assert.

**This runbook is written to be handed to a coding agent.** Each task has a
spec, a set of gotchas that will silently produce wrong answers if ignored, and
an acceptance test. You have Claude Code with room to run — the intent is that
you supervise and review rather than type, and that Edward and Jeremiah work
off the artifacts you produce.

---

## Before anything: five facts established on CPU, 2026-08-06/07

These came out of two independent audits (`docs/REVIVAL_AUDIT.md`,
`docs/VERIFICATION_2026-08-07.md`). Read them once; they change how the code
must be written.

1. **Your headline result is real.** `Batched_Gen.csv` reproduces **7/7** rows
   exactly, verified twice by different code over different artifacts
   (text logs, and the response pickles). The steering effect is not a
   spreadsheet artifact. `python3 scripts/verify_2025_results.py` re-proves it
   and exit-codes.
2. **n = 96 per arm, not ~100.** Every archived count is out of 96. gemma-2b
   steered-to-opinion is 96/96 = **100%**, a ceiling.
3. **The arrow-named CSV columns are per-arm marginals, not transitions.**
   `Init->Opin` means "the initial arm was judged opinionated." Real
   transitions need prompt-level pairing, which the CSVs cannot give.
4. **The response pickles are cumulative.** `log_236` holds 672 records — all
   seven models appended. Only the **last 96** belong to the named model.
5. **The refusal cross-steering runs were invalid, not null.** Details in §1
   below, because they dictate a coding rule.

---

## Task 1 — Ship the shape assertion. Thirty minutes. Do this first.

**Why.** The archived refusal `.pt` files are 1-D tensors of hidden width
(`Qwen-1.5-1.8B: (2048,)`, `llama-2-7b: (4096,)`). The steering code does
`steering_vector[layer]`. On a `[n_layers, d_model]` stack that correctly
yields a `(d_model,)` direction — which is why the opinion arm is sound. On a
**1-D** tensor it yields a **scalar**, which then broadcasts across the entire
residual width: a DC offset, not a direction. Five runs (Logs 210–214) produced
a clean-looking result table that tested nothing. Separately, the model loop
and the `vector_files` list were ordered differently, so each run loaded a
*different model's* vector.

Neither failure raised an exception. Both produced plausible numbers.

**Spec.** A single guard, called at every hook site and every vector load:

```python
def assert_direction(vec, n_layers, d_model, *, name):
    """Fail loudly before a silent broadcast can happen."""
    if vec.ndim != 2:
        raise ValueError(
            f"{name}: expected 2-D [n_layers, d_model], got {vec.ndim}-D "
            f"{tuple(vec.shape)}. A 1-D tensor indexed by layer yields a SCALAR "
            f"that broadcasts across the residual width — see docs/REVIVAL_AUDIT.md."
        )
    if tuple(vec.shape) != (n_layers, d_model):
        raise ValueError(
            f"{name}: shape {tuple(vec.shape)} != model ({n_layers}, {d_model})"
        )
```

**Acceptance.** A unit test that feeds a `(4096,)` tensor and asserts the guard
raises; a second that feeds a correct `(32, 4096)` and asserts it passes; and
the guard actually called on the load path, verified by tracing the call, not
by grepping for the function name.

**Also:** bind vectors to models by an explicit `{model_name: path}` mapping,
never by two lists that have to stay in the same order.

## Task 2 — Package the notebook. This is the critical path.

**Correction to an earlier assessment:** I previously reported that no reusable
runner existed, inferring from `src/` being 524 lines of loaders. That was
wrong and it was made without opening the notebook. It has roughly **30 named
functions and classes** covering the whole pipeline:

| stage | exists as |
|---|---|
| model loading | `getDevice`, `get_model` |
| tokenization | `tokenize_prompts` |
| residual capture | `batch_resids` (hooked, batched) |
| judging | `get_judgements` |
| data model | `Response`, `SteeredResponses`, `ModelResiduals` |
| vector calc | `get_opinion_vec_from_resids` |
| generation + steering | `normal_generation`, `batched_generation`, `steer_model` |
| results | `GeneralResults`, `TestResults` |
| geometry | `compare_vectors` |
| logging + resume | `setup_logging_directory`, `log_*`, `textlog_*`, `csvlog_*`, `get_*` |

**This is packaging, not building.** You are the only person who can say how
long it takes; the estimate here is a day or two, and it is yours to overrule.

**Spec.** Lift the cells into importable modules under `src/`, preserving
behavior exactly. Add: a config file (YAML/JSON) describing a run; a CLI entry
point; and a **write-once per-layer residual cache** keyed by *content hash* of
(model revision, prompt set, template, dtype), not by name or count — a cache
keyed on names silently serves stale data when contents change.

**Gotchas that will bite the agent:**
- Do not "fix" behavior while moving it. Port first, verify identical outputs,
  then change anything — in a separate commit.
- New writers must **not** reproduce the cumulative-append pattern. One run,
  one file, model name recorded *inside* the record, not inferred from position.
- The pickles execute arbitrary code on load. New artifacts should be written
  in a format that does not require `pickle` to read
  (`src/textlog_parse.py` shows the sanctioned path for the old ones).

**Acceptance.** `python3 scripts/verify_2025_results.py` still exits 0 after the
refactor, and one grid cell runs end-to-end unattended and writes a cache
another person can read with no GPU.

## Task 3 — Coefficients: the mystery is solved, and it changes the method

The per-layer L2 norms of your committed vectors:

| vector | last quarter of layers | first quarter | max/min |
|---|---|---|---|
| Qwen1.5-14B | 63.3% | 0.9% | **1391×** |
| Qwen1.5-7B | 65.0% | 1.1% | 961× |
| Qwen1.5-1.8B | 70.3% | 1.3% | 703× |
| Yi-6B | 69.2% | 0.8% | 603× |
| Llama-3-8B | 53.6% | 3.3% | 234× |
| **gemma-2b** | 33.4% | 23.6% | **3×** |
| **gemma-7b** | 31.6% | 22.4% | **2×** |

Because the method adds `(coeff / n_layers) · vec[layer]` with **one scalar
coefficient**, and the vector inherits the residual stream's norm growth:

- On Qwen / Yi / Llama, "all-layer steering" is in practice **late-layer
  steering** — the first quarter of the network receives ~1% of the injected
  norm.
- On gemma the profile is nearly flat, so it genuinely is all-layer.

That is why per-model coefficients never stabilised and why `farhan-YACF-Coeffs`
existed: the coefficient was silently compensating for an architectural norm
profile, not for anything about the models' opinionatedness. gemma needing ~5
where Qwen1.5-1.8B needed ~14 is close to the norm-profile ratio.

**Implication for the method:** report unit-normalized directions with the norm
profile stated separately, and prefer percentile-based activation capping over
a raw additive coefficient. **Implication for any depth analysis:** normalize
first, or the result will largely recover this plot. The chart is on the
dashboard under *Verification*, generated from
`dashboard/data/vector_norm_profiles.json`.

## Task 4 — Adopt the manifest protocol for anything new

`docs/REVIVAL_AUDIT.md` §"Provenance-backed measurement protocol" specifies it
in full. The short version: every future result row must be reconstructible
from an append-only record carrying identity (full repo-relative path, never a
Log number — Log numbers are **not** unique, two different `Log_200_*` dirs
exist), model revision hashes, data hashes plus **stored split indices** (not
just a seed), vector construction + payload SHA-256 + asserted shape,
generation parameters + rendered template hash, judge rubric hash + judge
revision + raw response + parse status, and the analysis commit.

Two archive lessons behind this: the 2025 judge used the unpinned `gpt-4o-mini`
alias, so those exact weights are unrecoverable; and the BBQ splits were
unseeded, so exact training membership must be recovered from artifacts before
any retraining claim.

## Task 5 — Never let extraction failures become behavior labels

There are **2,032** case-insensitive `none` markers across 107 archived files.
They are judge-**extraction** failures — not degeneration, not "nonsense."
Folding them into a behavior class inflates whatever condition had the most
parse failures.

This is why the zero-vector ablation control ("collapses to 99% nonsense") is
marked **under review** on the dashboard rather than cited. If you want it back
as a control, the fix is to separate that run's `none` markers from genuine
incoherence — a CPU job, no GPU.

`src/judging.py` already implements `ok` / `no_match` / `ambiguous` as explicit
states. Use it.

---

## What Edward and Jeremiah are doing, so you can plan around it

- **Edward:** measurement and geometry — the six-way rubric, gold-label
  annotation with Jeremiah, archive re-judging, the norm/geometry analysis, and
  keeping the claim ledger honest. Agent-assisted; off your critical path.
- **Jeremiah:** onboarding into annotation (`RUNBOOK_JEREMIAH.md`), then
  audits, figures, and writing. Everything of his lands by Aug 24.

**The one time-critical decision** is freezing the judge rubric before
annotation starts — relabeling afterwards throws the annotation away. That is
the only item that genuinely cannot wait.

## Open questions only you can answer

1. How long does Task 2 actually take? Every schedule downstream depends on it.
2. Is there anything in the notebook the audits mischaracterised? Both were
   done without talking to you, and you are the ground truth on intent.
3. Do you want the sprint to extend past the audit (the perturbation /
   dissociation work in `docs/2026-08-02_sprint_proposal.md`), or stay tight on
   the construct-validity paper? Two independent analyses landed on the audit as
   the primary; the extension is an upgrade if the Week-2 gate passes. **You are
   first author — this is your call.**
