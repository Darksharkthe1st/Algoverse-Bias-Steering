# 23 — What comes after Experiment 1

**Written 2026-08-29.** You asked: *we did one experiment, shouldn't there be
more? there are layers to this.* Yes. Here is the programme, with each stage
sized against real data rather than guessed at.

**None of this is for Sep 2.** The workshop paper stands on run 1 alone
(`notes/20`). This is what the project becomes afterwards, and knowing it now
changes one decision today: **it is the strongest argument yet for the residual
caching requirement**, because three of the four stages below cost *zero*
additional GPU time if and only if run 2 caches residuals.

---

## The state of the question

The headline question is *"is bias one mechanism or several?"* Run 1 could not
answer it, and it is worth being precise about why, because the reason determines
what comes next.

Run 1 built directions by ranking items on the model's **own** stereotype margin
and contrasting the extremes. Under that contrast:

- the pipeline demonstrably works (topic-identity control, 0.86–0.92);
- 10 of 46 model-category cells clear the bar, and the two that clear everywhere
  are Disability_status and Physical_appearance;
- no race-related category clears under **any** summary statistic (`notes/22` §F);
- where a direction does steer, it steers everything equally (`notes/10` §2).

So run 1 answered a narrower question — *does this contrast recover a
reproducible direction?* — with a clean no, and left the headline question open.
Four stages close it.

---

## Stage 1 — the annotation contrast · **PLANNED, ~3–5 h on one box**

`notes/20` §3.1, `repo/scripts/run2_annotation_contrast.py`.

Replace the behaviour-derived contrast with `context_condition`, a label BBQ
ships and the model never sees. Same items, same models, same pipeline, so the
comparison is controlled.

**What it decides.** Whether run 1's negative is a fact about bias or a fact
about the contrast. The reference paper gets floors of 0.95–0.99 from 32 items
per class using annotation labels; we got −0.45 to +0.82 from 172–320 using
behavioural ones. If annotation labels reproduce here too, the paper's claim
becomes *"the contrast decides, not the sample size"* — which is much harder to
dismiss than *"our contrast failed."*

**Risk, stated plainly.** The two arms differ in prompt length by 2.0–2.3×, and
the specificity control (`notes/19` §3.3) is the load-bearing test. It may fail.
`notes/19` §8 already names this as the weakest point in the plan.

---

## Stage 2 — sub-group resolution · **FREE if Stage 1 caches residuals**

The most interesting open lead in `README` §8, never sized. **Now sized.**

Race_ethnicity is not one thing. It pools nine annotated target groups, and a
pooled direction is an average over however many distinct directions those
groups have. If Black-targeted items alone yield a direction where pooled
Race_ethnicity does not, then run 1's negative is about the **unit of analysis**,
not about race — and that is a far more interesting finding than the negative
itself.

**Feasibility, measured** (`python -m scripts.subgroup_feasibility`):

| failing category | ambiguous items | annotated groups | groups ≥172 items | largest sub-group |
|---|---|---|---|---|
| Race_x_gender | 7,980 | 7 | **5** | African American+Black = 3,480 |
| Race_x_SES | 5,580 | 4 | **4** | Black+Hispanic+Latino = 2,160 |
| Race_ethnicity | 3,440 | 9 | **6** | African American+Black = **1,400** |
| Gender_identity | 2,836 | 5 | **3** | F = 1,672 |
| Nationality | 1,540 | 10 | **3** | (10-nationality African set) |
| Sexual_orientation | 432 | 5 | 0 | gay = 144 |

**Five of the six categories that fail everywhere have at least two sub-groups
larger than any n run 1 ever used.** Race_ethnicity's Black-targeted subset alone
has 1,400 ambiguous items — **more than four times** the 320 the failing pooled
direction was fit on. So "not enough data" is weakest exactly where the lead is
strongest.

**Cost: zero GPU.** It re-partitions the same items. With cached residuals it is
CPU arithmetic; without them it is a whole new rental. *This is the concrete
payoff of the caching requirement, and it is worth putting in `notes/14` as such.*

**The confound to pre-declare, before looking.** Splitting a category into *k*
sub-groups multiplies the number of floors computed by *k*, each measured on less
data. Both push toward finding something above the bar by chance. **Pre-declare
the sub-group list and a multiple-comparison correction before running it**, or
this becomes the single cleanest example of the defect the paper is about.

---

## Stage 3 — the causal arm · **the binding constraint, needs GPU**

`notes/14` §6.4 says it directly: of the four things that would lift this to
conference quality, three are reachable with the current plan and the fourth —
*"a causal arm that survives its controls"* — is a separate experiment of similar
size that **does not exist in any plan**. Run 1 has **none** of the three
required controls:

| control | why it is required | run 1 |
|---|---|---|
| covariance-matched random direction | norm-matching is too weak; a random direction with the right covariance is the real null | only norm-matched |
| coherence check on generations | distinguishes "moved bias" from "broke the model" | never run |
| system-prompt baseline | is activation steering doing anything a prompt could not? | never run |

Until these exist, no steering number is causal, and `notes/19` §4 shows why that
matters: run 1's own data has **three of four qwen-14b directions performing at or
below their random control**, and Age's real direction moving the margin *less*
than a random one at every dose.

**Do this only after Stage 1.** Steering a direction that does not reproduce
against itself is not interpretable (`notes/11` §8.4), so this stage needs Stage
1's directions to exist first.

---

## Stage 4 — universality across models · **partly free, mostly blocked**

Does a category's direction mean the same thing in two different models?

`notes/22` §E tried this on cached data and reports the honest answer: the
cross-model correlation of cross-category structure is +0.39, and it is
**uninterpretable**, because it is computed over directions most of which do not
reproduce against themselves. The test that would fix it — restrict to categories
reproducing in both models — cannot be run, because no model pair has three such
categories.

**So Stage 4 is gated on Stage 1 succeeding.** If the annotation contrast lifts
6–8 categories over the bar in several models, this becomes answerable and is
then also nearly free. If it does not, Stage 4 stays closed and should be said to
be closed rather than left dangling in `README` §8.

---

## The sequencing, and the one thing that decides it

```
Stage 1  annotation contrast      3-5 h GPU      decides everything downstream
   |
   +-- Stage 2  sub-groups        0 h  (cached)  the most interesting lead
   |
   +-- Stage 4  universality      0 h  (cached)  only if Stage 1 succeeds
   |
   +-- Stage 3  causal arm        separate run   the conference-quality blocker
```

**Stages 2 and 4 cost nothing *if and only if* residuals are cached.** Run 1 did
not cache them, which is why both have sat untouched for a week and why
`notes/22` had to reconstruct what it could from saved directions instead. The
caching requirement in `notes/14` §1 has until now been argued from principle —
*every analysis must be redoable with the GPU returned*. This is the version with
a number attached: **it is the difference between two more experiments costing
zero and costing two more rentals.**

---

## What I would do, in order

1. **Sep 2: submit the workshop paper** on run 1 alone. Nothing above is needed.
2. **Then Stage 1**, unhurried, with the pilot green and hole (d) resolved.
3. **Stage 2 immediately after**, from the cache, with the sub-group list
   pre-declared *before* the residuals are looked at.
4. **Stage 3 as the ICLR arm.** It is the thing that moves this from a strong
   workshop paper to a conference paper, and it is the only stage that needs its
   own planning cycle.

Stage 4 decides itself on Stage 1's results.
