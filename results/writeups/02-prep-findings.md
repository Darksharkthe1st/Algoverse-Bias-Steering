# Prep findings — 2026-08-20 (no GPU)

## 1. Vector shape audit — the norm profiles ARE trustworthy

I flagged a concern that the per-layer norm profiles underpinning Experiment 2
might have been computed from the tainted 1-D vectors. **Checked; concern is
resolved.** Do not re-litigate this.

`dashboard/data/vector_norm_profiles.json` records its own provenance:

```
"generated_from": "experiments/best_vecs/*_steer_vec.pkl"
```

and every entry carries an explicit `n_layers` / `d_model` / `dtype`, e.g.
`log_103_Qwen1.5-1.8B-Chat_steer_vec.pkl` → `n_layers: 24, d_model: 2048,
torch.float16`. These are the **opinion/bias** steering vectors, correctly shaped
2-D. They are *not* the 1-D refusal `.pt` files that caused the scalar-broadcast
bug. The Qwen1.5-1.8B profile is smooth and monotonic, 0.022 at layer 0 →
15.47 at layer 23 — the 703× spread is real.

**Conclusion:** Experiment 2's design may rest on the norm profiles.

### Shapes of every vector file in the repo (derived from file size)

| file | inferred shape | dtype |
|---|---|---|
| `official_refusal_vecs/Qwen-1_5-1_8B.pt` | (5, 24, 2048) | float64 |
| `official_refusal_vecs/gemma-2b-it.pt` | (5, 18, 2048) | float64 |
| `official_refusal_vecs/meta-llama-3-8b-instruct.pt` | (5, 32, 4096) | float64 |
| `official_refusal_vecs/llama-2-7b-chat-hf.pt` | (6, 32, 4096) | float64 |
| `official_refusal_vecs/yi-6b-chat.pt` | (6, 32, 4096) | float64 |
| `runs/*_qwen-1.8b/steering_vector.safetensors` | (24, 2048) | float16 |
| `runs/*_{yi-6b,qwen-7b}/steering_vector.safetensors` | (32, 4096) | float16 |

Sizes were used as a shape proxy deliberately — `docs/AGENTS.md` warns that
unpickling these `.pt` files executes arbitrary code. No file was unpickled.

Arditi's "official" vectors are **3-D**: `(n_positions, n_layers, d_model)`. The
2026 run artifacts are correctly **2-D**: `(n_layers, d_model)`.

### Stale path in the runbook

`RUNBOOK_JEREMIAH.md` points at `experiments/past_vecs/calculated_refusal_vecs/*.pt`
as "the actual 1-D tensors." **That path does not exist in the repo.** The only
`.pt` files are under `experiments/past_logs/past_vecs/official_refusal_vecs/`,
and those are 3-D, not 1-D. So the 1-D artifacts that produced the retracted
result are not in the tree — the bug cannot be reproduced from committed files.
Worth asking Edward where they went if that workstream is ever revived.

## 2. BBQ inventory

`datasets/BBQ_Prompt_Sets/` — 10 files, **52,628 rows total**.

| category | rows |
|---|---|
| Race_x_gender *(intersectional)* | 15,960 |
| Race_x_SES *(intersectional)* | 11,160 |
| Race_ethnicity | 6,880 |
| Gender_identity | 5,672 |
| Age | 3,680 |
| Nationality | 3,080 |
| Physical_appearance | 1,576 |
| Disability_status | 1,556 |
| Religion | 1,200 |
| Sexual_orientation | 864 |

**Eight base categories plus two intersectional ones.** Data volume is not a
constraint — even the smallest, Sexual_orientation at 864 rows, is far more than
a difference-in-means direction needs. This means we can afford split-half
extraction (for the noise floor) on every category without running short.

The two intersectional sets are interesting for Framing B specifically: if
`Race_x_gender` lands between `Race_ethnicity` and `Gender_identity` in the
clustering, that is a strong sign the geometry is tracking something real rather
than dataset artifacts.

**Note:** BBQ has **no political category.** Politics would have to come from
another source — `LLM_Values_PCT/` (Political Compass prompts, unused in 2025) or
a set we build.

## 3. Other dataset directories

| dir | files | note |
|---|---|---|
| `Homemade_Prompt_Sets/` | 19 | largest source of team-built prompts; not yet inspected |
| `LLM_Values_PCT/` | 3 | Political Compass; **unused in 2025** — candidate politics source |
| `Do_Not_Answer_Dataset/` | 2 | refusal thread |
| `GPT_Prompts/` | 2 | ~1,300 synthetic comparison prompts (2025 primary) |
| `Crows_Pairs/` | 1 | stereotype categories; the 2025 transfer failure target |
| `Snapshots/` | 1 | |

## 4b. SUPERSEDED 2026-08-20 — we now use BBQ's own answer key

**The table in §4 below is history.** It described reconstructing which answer is
biased from `stereotyped_groups` + polarity. We no longer do that.

BBQ ships `supplemental/additional_metadata.csv` in the `nyu-mll/BBQ` GitHub
repo — not part of the per-category jsonl release, which is why nobody here had
it. It carries a **`target_loc`** column: the index of the biased answer, per
`(category, example_id)`. Fetched by `scripts/fetch_bbq_metadata.py` into
`third_party/bbq/` (untracked; re-fetch on any new machine).

| | reconstruction (§4) | authors' key (now) |
|---|---|---|
| ambiguous rows scoreable | 20,750 / 25,814 (80.4%) | **25,806 / 25,814 (99.97%)** |
| Race_x_gender | 66.2% | **100%** |
| Race_x_SES | 66.7% | **100%** |
| Age | 77.8% | **100%** |
| only losses | — | 8 rows BBQ itself marks `NA` |

The intersectional sets are fully back, which restores the free sanity check.

**Where the two disagreed:** on the 20,750 rows both could score, they agree on
20,676 and disagree on **74 (0.36%)**. The reconstruction was close but not
exact, and it is now a fallback only. `AnswerRoles.source` records which path
produced each label.

⚠️ **`target_loc` already includes polarity — never flip it.** Verified on
`Race_ethnicity`: example 0 (negative, "who committed a crime?") points at the
stereotyped group; example 2 (non-negative, "who was innocent?") points at the
other one. Applying our own flip on top would invert half the labels back to
wrong and raise no error. There is a test asserting the key survives both
polarities unchanged.

The §5 note on polarity below still stands — it is why the fallback path works,
and it is now independently confirmed by the authors' labels.

---

## 4. BBQ labelling coverage — measured on real data *(superseded; see §4b)*

Scoring BBQ needs one thing: knowing which of the three answers counts as the
*biased* choice. That comes from matching `stereotyped_groups` against the
answers. Naive string equality only resolved **30.7%** of ambiguous rows, with
three categories at zero — because the categories do not agree on how groups are
written:

| category | answer group says | stereotype says |
|---|---|---|
| Religion | `Muslim` | `Muslim` — direct match |
| Race_x_gender | `F-Black` | `Black` — compound label |
| Nationality | `Europe` | `British` — region vs. nationality; the nationality appears only in the answer text |
| Gender_identity | `F` | `F` — mostly direct |

Fixed by matching on **token subsets** of the group label, falling back to the
answer text. Coverage after the fix, on `ambig` rows only:

| category | ambig rows | usable | coverage |
|---|---|---|---|
| Religion | 600 | 600 | 100% |
| Race_ethnicity | 3,440 | 3,440 | 100% |
| Nationality | 1,540 | 1,540 | 100% |
| Physical_appearance | 788 | 788 | 100% |
| Disability_status | 778 | 778 | 100% |
| Sexual_orientation | 432 | 432 | 100% |
| Gender_identity | 2,836 | 2,740 | 96.6% |
| Age | 1,840 | 1,432 | 77.8% |
| Race_x_SES | 5,580 | 3,720 | 66.7% |
| Race_x_gender | 7,980 | 5,280 | 66.2% |
| **TOTAL** | **25,814** | **20,750** | **80.4%** |

**The remaining 19.6% is refused on purpose, not lost.** A row resolves to
`None` when zero answers match the stereotype, when *two* do, or when the
question polarity is unrecognised. Two matches is the normal case for the
intersectional sets: both answers share the stereotyped race ("The Black woman"
vs. "The Black man") and `stereotyped_groups` names only the race, so the field
genuinely cannot say which answer is biased. Guessing there would silently
mislabel rows.

Age dropped from an apparent 100% to 77.8% for the same reason — the stricter
rule now rejects rows it previously mislabelled. That is a correction, not a
regression.

Even the smallest usable category (Sexual_orientation, 432) is far more than a
difference-in-means direction needs, so split-half extraction for the noise floor
is affordable everywhere.

## 5. Polarity is load-bearing

BBQ pairs each context with a negative question ("Who likely planted the bomb?")
and a non-negative one ("Who was the victim?"). On a **non-negative** question
the biased answer is the **non-stereotyped** group — the stereotype implies the
targeted group is the bad actor, so naming the other group as the victim is the
biased response.

Getting this backwards would silently invert half the labels and would not throw
an error. Religion's ambiguous rows are an exact 300/300 neg/nonneg split, so
this affects half the data. `resolve_answer_roles` requires a recognised polarity
and returns `None` otherwise rather than guessing; there is a unit test for the
inversion specifically.

## Still to do in prep

- Inspect the BBQ row schema (fields, how a contrast pair would be formed).
- Read `src/bias_steer/refusal_extract.py` + `steering.py` — what extraction gives
  us for free vs. what needs writing for a per-topic pipeline.
- Inspect `Homemade_Prompt_Sets/` and `LLM_Values_PCT/` for usable topics.
- Draft the magnitude-matched design for Experiment 2.
