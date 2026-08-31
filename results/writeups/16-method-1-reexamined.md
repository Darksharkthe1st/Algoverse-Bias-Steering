# Method 1 re-examined: it is weak, not random

> ### ⚠ SUSPENDED IN PART 2026-08-23 by defect N6 — read `18-parser-audit.md` first
>
> **The convergence claim in "The part that was missed" below is suspended and
> must not be reported.** `17-reference-paper-and-contrast.md` §8 suspended it;
> until 2026-08-28 that suspension existed only in `17`, so this file read as a
> live instruction to report it. Nothing here is deleted — the banner is the fix.
>
> N6: `parse_choice` resolves a response to the **earliest-mentioned** option, so
> every failure mode lands on whichever option is named first. A first-mention
> parser produces the exact signature read here as "person-consistency below
> 100%" **even if the model is perfectly consistent**, and run 1 cannot separate
> the two because the raw response text was never saved (S5, biting twice).
>
> So the pooled **58.0%** and the whole per-category ranking are contaminated by
> an unknown amount. The ranking may well survive validation — it is a striking
> correlation — but it is **UNVERIFIED**, not converging evidence.
>
> The separate retraction at the foot of this file (the Spearman +0.67) stands on
> its own, different grounds: mixed models and the untrusted estimator.
>
> What would lift the suspension: `19-plan-closure-and-audit.md` §5 — the
> order-swap qualification, the mirror-pair test, and the hand-labelled accuracy
> audit. Until parser accuracy is reported with an interval, these numbers are
> not evidence.

*Written 2026-08-23, after Jeremiah asked whether method 1's failure could have
been chance, given that published papers score BBQ this way.*

## The correction

The artifact says of method 1 (generate a response, parse which option it named):
"Person-consistency ran 48-68% against a 50% coin-flip line. We were measuring
the model's decoding, not its representations."

The range is right. **The conclusion was stated too strongly.** Pooled across all
ten categories, person-consistency is **58.0%** (n = 1142 paired items),
z = +5.39, p = 7.2e-08. That is not a coin flip. The model does carry a real,
order-independent preference; it is just weak.

## The part that was missed, and it matters

Consistency is not uniform across categories. Sorted:

| category | person-consistency | binomial p vs 50% | extraction floor |
|---|---|---|---|
| Disability_status | 73.1% | 0.0000 | reproduces in 4/4 models |
| Physical_appearance | 68.4% | 0.0001 | reproduces in 3/4, marginal in 1 |
| Gender_identity | 65.8% | 0.0005 | — |
| Nationality | 60.0% | 0.032 | marginal (q05 0.5103) |
| Age | 58.0% | 0.089 | — |
| Religion | 54.2% | n.s. | — |
| Race_x_SES | 52.4% | n.s. | fails |
| Race_x_gender | 50.4% | n.s. | fails |
| Race_ethnicity | 48.3% | n.s. | fails in every model |
| Sexual_orientation | 47.5% | n.s. | fails |

**The two methods agree.** The categories where the model answers the same way
regardless of option order are exactly the categories whose directions survive
the extraction floor. The categories at chance consistency are exactly the ones
with no reproducible direction. These are different measurements — one is
behavioural and order-based, the other is geometric and split-half — and they
rank the categories the same way.

This is converging evidence and it strengthens the headline result. It should be
reported.

## So why was method 1 still the wrong instrument here?

Not because it is invalid. Because of what we needed it *for*.

Published BBQ papers report an **aggregate** bias score over thousands of items,
and BBQ ships each item in counterbalanced option orderings precisely so that
position effects cancel in the average. At that level 58% consistency is fine —
the noise averages out and the aggregate is meaningful.

We needed something different: a **per-item** label, to sort individual items
into a "biased" pole and a "not biased" pole and take the difference of means.
At 58% consistency roughly **42% of items get a label that flips if you reorder
the prompt**. Averaging cannot rescue that, because the label *is* the thing
being averaged. Corrupted poles produce a corrupted direction.

**The lesson is not "method 1 is broken."** It is that a measurement can be sound
for aggregate reporting and unusable for per-item labelling, and which one you
need depends on the analysis downstream. That distinction should have been made
before any GPU was provisioned.

## Action for run 2

- Report the consistency-vs-floor convergence above as a secondary result.
- Correct the artifact's wording from "we were measuring the model's decoding,
  not its representations" to something accurate: mostly decoding, with a real
  but weak representational signal underneath that points the same way the floor
  does.

## Quantifying the convergence — RETRACTED 2026-08-23

A Spearman rank correlation of **+0.67 (n = 10)** was computed here between
behavioural consistency and the extraction floor, and reported as converging
evidence. **It does not support that claim and must not be used.** Tracing the
provenance of every value showed two disqualifying confounds:

1. **The two measures come from different models.** The consistency numbers are
   from `_bbq_choice_diagnostics.json`, whose `model` field reads **qwen-1.8b** —
   the one model where *no* category produced a reproducible direction, and which
   the artifact deliberately excludes from the "four capable models." The floor
   values were pooled across gemma-2b, qwen-7b, qwen-14b and yi-6b. So it
   correlated behaviour in one model against geometry in four others.

2. **The floors came from the untrusted estimator.** Every value was scraped from
   the `_probe_alpha_sweep_*.json` files — the *probe*, whose regularisation
   constant is the source of defects S3 and S4, and which the artifact's triage
   marks as inheriting the alpha question. None came from `extremes`, the
   estimator with no free parameter that produces the results which survive audit.

The per-category consistency numbers earlier in this document are sound — they are
a single model's own paired items, binomial-tested. **The convergence claim is
not.** It is a hypothesis for run 2, not a finding.

## What would actually demonstrate it

Measure both quantities **on the same model, in the same run**:

- behavioural consistency from paired order-swapped items,
- the extraction floor from `extremes`, with the interval run 2 provides,

then correlate across categories within that model, and repeat per model so the
correlation itself has a spread rather than being one number from pooled sources.
Run 2 caches residuals, so this costs no extra GPU time — it is arithmetic on
stored arrays. Worth doing; not yet done.
