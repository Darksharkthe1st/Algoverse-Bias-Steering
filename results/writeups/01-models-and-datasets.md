# Models and datasets — what the team has actually used

**Sources:** `repo/src/bias_steer/models.py` (the live catalog),
`repo/docs/2026-08-01_project_analysis.md` (2025 campaign),
`repo/docs/PREREG.md` §3b (frozen submission model),
`repo/docs/VERIFICATION_2026-08-07.md` (per-layer norm profiles).

## The registered model catalog

Every one is open-weight. `src/bias_steer/models.py:182-198`.

| key | HF id | size | chat_template | notes |
|---|---|---|---|---|
| `qwen-1.8b` | `Qwen/Qwen1.5-1.8B-Chat` | 1.8B | yes | 2025 workhorse; the 2026 refusal repro ran here |
| `qwen-7b` | `Qwen/Qwen1.5-7B-Chat` | 7B | yes | PREREG **fallback** model |
| `qwen-14b` | `Qwen/Qwen1.5-14B-Chat` | 14B | yes | |
| `yi-6b` | `01-ai/Yi-6B-Chat` | 6B | yes | |
| `gemma-2b` | `google/gemma-2b-it` | 2B | no | **gated** — needs HF approval |
| `gemma-7b` | `google/gemma-7b-it` | 7B | no | **gated** |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | no | **gated** |
| `llama-2-7b` | `meta-llama/Llama-2-7b-chat-hf` | 7B | yes | added for the Arditi repro |
| `qwen3-8b` | `Qwen/Qwen3-8B` @ `b968826d9c46` | 8B | yes | **frozen submission model** (contract §12 A4) |

The 2025 campaign additionally used Llama-2-13b-chat and an abandoned Qwen2.5
0.5B→32B scaling ladder. Those are not in the catalog.

**Gating matters:** gemma and llama need Hugging Face access approval on whatever
token we log in with. Qwen and Yi do not. Plan around this.

## Which models actually respond — measured, not guessed

`docs/VERIFICATION_2026-08-07.md` recounted the 2025 campaign from raw records,
7 of 7 rows reproducing. Counts are **out of n = 96** (not percentages):
`(Init, Opin, Neut)` = baseline opinionated / steered toward opinion / steered
toward neutral.

| model | Init (baseline) | Opin | Neut | read |
|---|---|---|---|---|
| gemma-2b-it | **82**/96 | 96 *(ceiling)* | 21 | most willing to commit |
| gemma-7b-it | **75**/96 | 89 | 37 | very willing |
| Qwen1.5-14B | **62**/96 | 86 | 17 | willing |
| Yi-6B-Chat | **55**/96 | 88 | 47 | willing; weak downward steering |
| Qwen1.5-7B | **48**/96 | 78 | 3 | middling |
| Qwen1.5-1.8B | **30**/96 | 75 | 2 | low but usable; huge steering range |
| **Meta-Llama-3-8B** | **6**/96 | 24 | 12 | **effectively inert — avoid** |

**`Init` is the column that matters for Experiment 1.** It measures how often the
model commits to a side instead of hedging. Our biased bucket only fills when the
model is willing to name a group rather than say "Can't answer", so a low `Init`
predicts an empty positive pole and no direction.

**Llama-3-8B is the model that "didn't show anything"** — 6/96 at baseline,
topping out at 24/96 even when steered. Confirmed, not a memory. Do not spend GPU
time on it.

Caveat, stated honestly: this measured *opinionation on comparison prompts*, not
*stereotype choice on BBQ*. The two are related — both are willingness to commit
rather than hedge — but not identical. It is the best available evidence, and it
is why `scripts/bbq_base_rates.py` measures the real quantity before we extract
anything.

### Recommended model set — three families, ordered

1. **`qwen-1.8b`** — develop and iterate. Fast, ungated, most historical
   comparators. `Init` 30/96 is lower than ideal but workable.
2. **`gemma-2b`** — highest `Init` (82/96), a **different family**, and the only
   flat per-layer norm profile. **Access GRANTED 2026-08-20** for both
   `gemma-2b-it` and `gemma-7b-it` (one licence acceptance covers the family).
   Jeremiah holds a Hugging Face **read** token; it goes on the Lambda box via
   `huggingface-cli login` or `export HF_TOKEN=...`. Not yet verified against a
   real download — confirm on the box before relying on it.
3. **`qwen-14b`** — `Init` 62/96, scales within the Qwen family.
4. **`yi-6b`** — `Init` 55/96, a third family, ungated. Good insurance.

Avoid `llama3-8b`. `qwen3-8b` stays reserved for headline numbers that must sit
beside the team's paper.

One caution on gemma-2b: steered toward opinion it hit **96/96, a total ceiling
with zero neutral responses left**. Saturation is a problem for measuring
steering strength, though not for measuring base rates.

## Which to use for our two experiments

Recommendation: develop on **`qwen-1.8b`** (fast, ungated, most historical
comparators), confirm on **`qwen-7b`**, and if a headline number needs to sit
next to the team's paper, re-run on **`qwen3-8b`**.

For Experiment 2 specifically there is a strong reason to include **`gemma-2b`**
as a second family — see the norm profile below — but it is gated, so request HF
access early if we want it.

## Per-layer norm profiles — directly relevant to Experiment 2

From `docs/VERIFICATION_2026-08-07.md`. This is the ratio between the
largest-norm and smallest-norm per-layer vector within a model:

| model | peak layer | max share | min share | spread |
|---|---|---|---|---|
| Qwen1.5-1.8B | 24 | 70.3% | 1.3% | **703×** |
| Qwen1.5-7B | 32 | 65.0% | 1.1% | **961×** |
| Qwen1.5-14B | 40 | 63.3% | 0.9% | **1391×** |
| Yi-6B | 32 | 69.2% | 0.8% | **602×** |
| gemma-2b | 18 | 33.4% | 23.6% | **3×** |
| gemma-7b | 28 | 31.6% | 22.4% | **2×** |

**Why this matters.** On Qwen/Yi/Llama the profile is so steep that an
"all-layer average" vector is dominated by a handful of late layers — Farhan's
method is *already* approximately a best-layer method there. On **gemma the
profile is nearly flat**, so all-layer steering genuinely is all-layer.

That makes gemma the model where per-layer-specific steering has the most room
to differ from both baselines, and Qwen the model where the three methods may
partially collapse into each other. Running Experiment 2 on one steep-profile
and one flat-profile model would make the comparison interpretable rather than
model-specific.

## Datasets in the repo

Under `repo/datasets/`:

- `BBQ_Prompt_Sets/` — BBQ, **10 categories** with ground-truth stereotype labels.
  Starting point for Experiment 1.
- `Crows_Pairs/` — CrowS-Pairs stereotype categories.
- `Do_Not_Answer_Dataset/` — the refusal thread.
- `GPT_Prompts/` — ~1,300 synthetic GPT-generated comparison prompts (the 2025
  primary set; opinionation-style, not stereotype-style).
- `Homemade_Prompt_Sets/` — team-written prompts.
- `LLM_Values_PCT/` — Political Compass prompts. **Unused in 2025.**
- `Snapshots/`

## The 2025 transfer failure — load-bearing context for Experiment 1

Vectors trained on the synthetic comparison prompts **largely stopped working on
CrowS-Pairs** (gemma-7b: 78/18 → 77/19 steered — no effect; Llama-2-7b likewise)
and only semi-worked on BBQ.

This is not a footnote — it is direct evidence *for* Experiment 1's hypothesis.
If one bias direction transferred cleanly across all bias types, the "subtypes"
question would already be answered no. It failing to transfer is the observation
the experiment is built to explain. Do not soften it and do not overclaim it.

## Useful docs for tomorrow

- **`repo/docs/03-gpu-bringup.md`** — an existing GPU bring-up walkthrough,
  including `huggingface-cli login` and the gated-model caveat. Read this before
  the Lambda session.
- `repo/docs/05-refusal-repro.md`, `repo/docs/06-refusal-generation.md` — how the
  refusal extraction and generation actually run.
- `repo/docs/findings/2026-08-16-refusal-repro-qwen-1.8b.md` — the last real run;
  box was a **Lambda A100-40GB**.
