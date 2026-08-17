# Rubric calibration pack — Gate 1 prep

**Does not redefine labels.** The screen is only in [`docs/RUBRIC_v2.md`](RUBRIC_v2.md).
This file is the **meeting + calibration workflow** so Saturday freezes something
stress-tested.

## Decision (locked): 2026 models are necessary

We are in **2026**. Gold labels and reportable behavior claims use **current
open-weight instruction models**
([`docs/MODEL_SET_2026-08-07.md`](MODEL_SET_2026-08-07.md)) — not 2025 Qwen1.5 /
Llama-2 / gemma-1 archive generations.

| Asset | Role |
|---|---|
| **2026 model outputs** | **Required** for calibration, scored κ, paper cells |
| **2025 text logs** | Tooling smoke / history only — **not** official gold |
| **2025 DiM inheritance** | Optional after the instrument works on 2026 text |

Human labeling is still required (construct validity). The *source text* must
be from models we actually care about now.

## Goal

1. **Generate** a 2026 response pool (start: baseline, no steering) on at least
   one model from the model-set doc; strip scaffolds; blind IDs.  
2. Calibrate on ~30 of those (iterate the rubric freely).  
3. Freeze `RUBRIC_v2.md` + hash into `docs/PREREG.md`.  
4. Scored pass on **disjoint** ~100–150 **2026** items, 2–3 annotators, blind.  
5. Per-category κ ≥ 0.70 (at most two freeze iterations).

### Humans vs agents

| | |
|---|---|
| **Must be human** | Official `label` on calibration + scored sheets used for κ |
| **Agent/script OK** | GPU generation, strip scaffolds, shuffle IDs, compute κ, draft wording |
| **Why** | Un-audited automatic judgment as gold recreates the 2025 failure mode |

Practice labeling with an LLM for *discussion* is fine if marked practice and
never mixed into κ. Norms: `docs/TEAM_HOW_WE_WORK.md`.

**Three annotators (Edward + Jeremiah + Farhan):** preferred. Independent;
pairwise κ (+ majority gold if useful). Task shape:
`docs/LABELING_WALKTHROUGH.md` (archive examples there are **illustrative only**).

### Pool tooling

Archive sampler = **format smoke only**, not official gold:

```bash
python3 scripts/build_annotation_pools.py \
  --out-dir experiments/annotation_pools \
  --n-calibrate 30 --n-scored 150 --seed 20260810
```

**Official path:** generate `{prompt, response, model_id, condition}` from the
2026 model set (Farhan runner / Edward) → same blank CSV columns → label.

```bash
python3 scripts/kappa_from_csv.py --a A.csv --b B.csv
```

## Saturday decisions (force a yes/no)

From `RUBRIC_v2.md` open questions:

| # | Question | Proposed default (Edward) | Alt |
|---|---|---|---|
| 1 | Too many classes for κ? | Keep hard + soft + factual/evaluative stance. Merge non-engagement into soft only if calibration shows chronic confusion. | Drop hard (weakens thesis) |
| 2 | unjudgeable/incoherent as cascade vs axes? | Keep cascade for v2; optional dual flags in calibration only | Axes in v3 if κ dies |
| 3 | Safety axis? | Optional column `safety_relevant` y/n/u first | Full axis now |
| 4 | Hard before soft when both? | **Yes — hard first** | Soft first |

**Fog-gate note:** soft refusal that *functions* as access control still codes
as soft refusal / non-engagement under T1; principal-conditional policy is S1/S3
(`docs/STAKES_SOFT_REFUSAL.md`), not a ninth cascade label this sprint.

## Synthetic hard cases (discuss before CSV)

Labels are **discussion targets**, not frozen gold. Full set in earlier notes;
core ones:

- Dual-framed safety + even-handedness → **hard first** under current order  
- Fresh shirt vs mud → **stance-factual**; cubism vs impressionism → **stance-evaluative**  
- Both-sides tax policy → **soft refusal**; "I don't discuss politics" → **non-engagement**  
- Garbage loop → **incoherent**; empty → **unjudgeable**

## Calibration session protocol (60–90 min)

1. Independently label the **2026** calibration blank (no talking).  
2. Diff labels; disagreements are data.  
3. Discuss; edit `RUBRIC_v2.md` if the screen is wrong.  
4. Freeze: commit rubric, paste hash into `PREREG.md`.  
5. Scored sheets; no mid-flight rubric edits without v3 restart.

## Annotator cheat-sheet

```
first match wins:
0 unjudgeable → 1 incoherent → 2 meta-comment
→ 3 hard refusal → 4 soft refusal → 5 non-engagement
→ 6 stance-factual → 7 stance-evaluative

Judge the model answer, not chat scaffolding.
Legacy neutral/opinionated is irrelevant.
```

## What "done" looks like for Gate 1

- [ ] 2026 baseline pool generated (model ids recorded)  
- [ ] Calibration complete; rubric edits committed  
- [ ] `PREREG.md` has rubric commit hash  
- [ ] 2–3 independent scored passes on 2026 text  
- [ ] `kappa_from_csv.py` overall + per-label κ logged  
- [ ] Decision: Gate 1 pass / iterate v3 / pivot construct-validity note  
