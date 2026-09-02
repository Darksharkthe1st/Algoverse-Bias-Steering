# Jeremiah's workstream — overview

**Last updated:** 2026-08-20 (from Jeremiah, in conversation)

## Status of the source docs

- `repo/RUNBOOK_JEREMIAH.md` (Aug 7) is **DEPRIORITIZED, not dead.** The
  fault-susceptibility study and measurement-geometry work in it are not the
  current priority, but Edward may still want them — treat them as deferred and
  revisit later, do not discard. Its claim that "nothing in your workstream needs
  a GPU" no longer holds, since the two experiments below do need one.
- `repo/docs/work-splits/jz-task-list.md` (Aug 18) is closer, but the two
  experiments below are the priority and are stated here in Jeremiah's own terms.
  The remaining JZ items are lower priority and mostly implementation work.
- Canonical project state still lives in `repo/PROJECT_STATE.md` and
  `repo/RESEARCH_CONTRACT.md`. Those govern the *paper*; the two experiments
  below are Jeremiah's contribution into it.

## Timeline

Behind schedule. Numbers freeze Mon 2026-08-24, deadline Fri 2026-08-28 AoE.
**Experiments run tomorrow (2026-08-21) at the latest.**

---

## Experiment 1 — Is bias one direction, or several?

**Motivation.** A recent paper (Jeremiah to share) splits *refusal* into
categories rather than treating it as a single direction. The team wants the
analogous study for *bias*: do different kinds of bias have different
representations?

> **Two framings are on the table and BOTH are kept.** Framing B below is the
> current design as of 2026-08-20. Framing A is the earlier top-down version —
> Jeremiah considers it still valid and interesting and wants it retained. Neither
> is discarded.

---

### Framing A — top-down (earlier; retained, not current)

Define bias categories first, then test whether their vectors differ. Candidate
category axes discussed:

- **demographic** — race vs. gender vs. religion
- **temporal** — currently controversial vs. formerly controversial
- **motivational** — safety bias vs. don't-hurt-feelings bias (*why* the model
  hedges rather than *what about*)

Jeremiah noted these were said off the cuff, and that BBQ only supports the
demographic cut — the temporal and motivational cuts would need prompt sets that
do not exist yet. Still worth revisiting: if Framing B produces clusters, these
are ready-made hypotheses for what the connecting factor might be.

**The question.** Extract a steering vector per bias category and ask:

- Are the vectors meaningfully different from each other, or effectively the same
  direction?
- Does a single vector work for all categories? If so, why — what is the shared
  structure?
- If they are all different *and* each works on every category, what is the
  connection?
- Is the grouping structure just the surface taxonomy (politics vs. race vs.
  religion), or is there a deeper organization — some categories more connected
  to each other than others?

**Approach.**
1. Start with **BBQ** (`repo/datasets/BBQ_Prompt_Sets/*.jsonl` — has ground-truth
   stereotype categories: race, gender, religion, …).
2. Find additional datasets covering more categories.
3. Build our own prompt sets for categories nothing covers.
4. Extract a direction per category with a consistent pipeline.
5. Compare them geometrically and behaviorally; adjudicate the grouping.

**Output.** A taxonomy of bias types, grounded in both the data and the geometry —
i.e. groups that are defensible because the vectors actually differ, not because
the category names differ.

---

### Framing B — bottom-up (CURRENT DESIGN, 2026-08-20)

Do **not** define categories in advance. Let them fall out of the data. The
groupings may not even need names at first.

1. Extract a direction for **many fine-grained bias topics** — race, political,
   religion, gender, age, nationality, SES, … as many as we can get data for.
2. Compute the pairwise similarity structure among them.
3. **Cluster, don't assign** — let the groups emerge.
4. *Then* look at what landed together and ask what it has in common.

Jeremiah's framing: *not* "this group is the temporal one," but **"I see race and
political are connected — what do they have in common?"**

The taxonomy is an **output of the experiment, not an input to it.**
Interpretation is the last step, never the first. This also keeps us clear of the
team's rule against coining terminology: we are not inventing categories, we are
reporting which topics group and then asking why. Framing A's axes become
*candidate explanations* for observed clusters rather than assumptions going in.

**Approach.**
1. Start with **BBQ** (10 ground-truth stereotype categories).
2. Add CrowS-Pairs categories and any other topic we have or can build data for.
   More leaves = more structure to find.
3. Extract a direction per topic with one consistent pipeline.
4. Similarity matrix → hierarchical clustering. The dendrogram shows which topics
   merge first, i.e. which are most connected.
5. Interpret the clusters last.

**Output.** A data-derived grouping of bias topics, with the connecting factor of
each group proposed *after* seeing what grouped.

### Two dangers that must be controlled for (apply to both framings)

**1. Clustering noise looks beautiful.** Random vectors produce convincing
dendrograms. Without a null we will find structure whether or not it exists —
exactly the "it looked too clean" failure the team already lived through.
Controls: the noise floor (re-extract the same topic from split halves), and a
permutation null (shuffle prompts across topic labels, re-extract, cluster — does
that produce equally pretty structure?).

**2. Dataset/format confound — potentially fatal.** If race comes from BBQ and
political comes from a homemade set, and they cluster by *source* rather than by
content, we would be measuring **prompt format, not bias type.** Required
controls: multiple topics from the *same* dataset, and at least one topic drawn
from *two* datasets. If `race-BBQ` sits next to `race-CrowS`, the signal is
topical. If everything BBQ clumps together regardless of topic, we are measuring
BBQ. This check must happen before any cluster is interpreted.

---

## Experiment 2 — Per-layer steering vectors

**Background.** The team can already compute refusal vectors that push a model
between "sorry, I can't answer that" and a direct, opinionated answer, by adding
or subtracting the vector. Two approaches exist so far:

| approach | who | what it does |
|---|---|---|
| best single layer | the Arditi paper | pick the single best layer's vector, apply it |
| all-layer average | Farhan | average the per-layer vectors into one, apply it |

**The idea.** Compute a **separate vector for each layer**, and apply each layer
its own vector — rather than one vector everywhere. Does layer-specific steering
control refusal better than either prior method?

**Output.** A three-way comparison: best-single-layer vs. all-layer-average vs.
per-layer-specific, on the same prompts and the same metric.

---

## After these two

The rest of `repo/docs/work-splits/jz-task-list.md` (JZ-1 → JZ-4). Lower
significance; largely coding work.

---

## Settled (2026-08-20)

- **Lambda:** Jeremiah is a collaborator on Farhan's Lambda team account. Credits
  are plentiful and GPU time is not a constraint. He has not set up SSH to Lambda
  before (has used RunPod), so walk him through access step by step.
- **Models:** open-weight only, otherwise our choice. Pick for iteration speed.
- **Budget:** as much GPU time as needed.
- **Metric (Experiment 1):** there is no pre-specified metric. The study is
  exploratory — the goal is to show that **bias has subtypes**, i.e. that it is
  not one undifferentiated thing, by finding vectors, numbers and behaviours that
  separate categories and reveal which are more connected than others. We define
  the category names and attributes ourselves.
- **Mandate:** the team assigned these two experiments. Whether they are on the
  paper's critical path is not the deciding factor — they are the task.

## Still open

- The refusal-categories paper Jeremiah mentioned — arXiv link to come.
- Lambda instance type / region / how the box gets provisioned.

## Constraints inherited from the team (apply to both experiments)

- **Assert tensor shape `(n_layers, d_model)` before any cosine or steering
  application.** A 1-D vector indexed by layer yields a scalar broadcast — a DC
  offset, not a direction. This bug produced a convincing table that survived a
  year. See `repo/docs/REVIVAL_AUDIT.md`.
- Say "**a** direction", never "the direction" — steering success does not
  identify the representation.
- Never quote a number without its denominator and its judge version.
- Don't coin terminology. The behaviour is *hedging*; "soft refusal" is retired.
- Numbers must trace to a committed artifact under `experiments/` or `runs/`.
