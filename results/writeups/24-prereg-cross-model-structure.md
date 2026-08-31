# 24 — Pre-registration: cross-model structure and the shared-axis question

**Written 2026-08-31, after the qwen-1.8b R1 result and BEFORE any second
model's R1 analysis ran.** yi-6b was mid-download at freeze time. The
qwen-1.8b cosine matrix has been seen; nothing here conditions on any other
model's matrix. Committed and hashed so the reader can check the order.

Run 1 could never test whether cross-category structure replicates across
models: at most three categories reproduced in two models, and three points
have no power. R1 changes that. All ten categories reproduce on qwen-1.8b, so
if they reproduce on the other models too, every pair of models shares a full
45-entry off-diagonal structure. That comparison must be pre-registered now,
before the second matrix exists.

## A. Structure replication across models

**Hypothesis.** The cross-category cosine structure of annotation-derived
directions replicates across models.

**Primary statistic.** For each model pair: Spearman rank correlation between
the 45 off-diagonal entries of the two cosine matrices, categories matched by
name (models that drop categories use the shared subset; report its size).

**Null.** Jointly permute the category labels (rows and columns together) of
one matrix, 10,000 draws, seed 0; p = fraction of permuted draws with
Spearman >= observed. One-sided by design: only positive agreement counts as
replication.

**Secondary, reported not gated:** Pearson on the same entries; mean absolute
off-diagonal per model (does the shared-component level itself replicate).

**Recoverability replication, reported per model:** count of categories whose
floor beats its own negative control with disjoint 95% CIs (the run-2
usability criterion, unchanged).

## B. What is the shared axis? The abstention-alignment test

The qwen-1.8b directions share a large component (median off-diagonal 0.81)
that does not reduce to context length (projection control). One candidate
has a behavioural signature we can test with data already on disk: an
answerability or abstention axis. Ambiguous BBQ items are exactly the items
where "can't answer" is correct; the cached margin files for qwen-7b,
qwen-14b, gemma-2b, and yi-6b record a per-item abstention margin
(logP(unknown) − max logP(named)) from run 1.

**Test, fixed now.** For each model with both an R1 run and a margins cache:

1. Shared axis := the per-layer mean of the ten unit-normalised (per layer)
   category directions from that model's R1 run.
2. For every R1 ambiguous item that also appears in the margins cache
   (matched by item id), project its residual onto the shared axis per layer;
   summarise across layers by the median (the project's standard summary).
3. Primary statistic: Spearman correlation between projection and cached
   abstention margin, computed within category and summarised as the median
   across categories (guards the between-category confound that invalidated
   the pooled abstention correlation in notes/22 §D). Bootstrap CI over items.

**Reading, fixed now.** A median within-category |Spearman| >= 0.30 with CI
excluding zero counts as behavioural alignment (the axis relates to
abstention); below that, the axis is not behaviourally identified by this
test and the paper keeps calling it unnamed. Either way the number is
reported. This test cannot run on qwen-1.8b (no margins cache for it); it is
not a licence to skip it elsewhere.

**Not claimed in advance:** that the shared axis IS abstention, hedging, or
uncertainty. The test can fail. Its purpose is to replace speculation about
the axis with one measured alignment, in either direction.

## C. What is deliberately not pre-registered here

Steering or ablation along any R1 direction (needs the controls listed in
the manuscript §3), the matched-items behaviour-arm comparison (17 §5.4, a
GPU pass run 2 does not include), and any SAE analysis. Camera-ready scope.
