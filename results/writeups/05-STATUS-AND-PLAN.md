# STATUS & PLAN — read this first

**Written Thu 2026-08-20, ~2:15 AM.** Straight answers to: what is done, what is
left, what could go wrong, and can this actually work.

---

## Short answers to the four questions

**Is anything broken?** No. 34 new tests pass, the full suite is 156 passed /
0 failed, and the BBQ loader was verified against the real data files. There are
known *limitations* (below), but nothing is broken.

**Do you need to find answers to questions before tomorrow?** No. Nothing blocks
Experiment 1. The blocking questions in `04-team-questions.md` are for
Experiment 2 (on hold anyway) and for work that comes later.

**Is anything else needed before the GPU?** Yes — one piece, and it is small.
See "The one remaining gap" below. It needs no GPU and no decisions, so it can be
built before tomorrow.

**Can we actually get this to work?** Probably, with one genuine scientific risk
that we will know about within the first ~20 minutes on the box. See "Risks".

---

## What got done tonight

| | |
|---|---|
| Repo cloned, git installed, long-path problem fixed | ✅ |
| Read the control plane, both runbooks, the task list, the G1 handoff | ✅ |
| Confirmed the Joad et al. paper and how it maps onto our study | ✅ |
| Audited every vector file's shape — norm profiles are trustworthy | ✅ |
| Experiment 1 designed, bottom-up framing, both framings preserved | ✅ |
| `src/bias_steer/bias_taxonomy.py` — the analysis layer | ✅ |
| `tests/test_bias_taxonomy.py` — 34 tests, all passing | ✅ |
| BBQ labelling fixed: 30.7% → 80.4% coverage, all 10 categories | ✅ |
| Committed to branch `jz/bias-taxonomy` (`98efa63`), **not pushed** | ✅ |
| 10 questions drafted for the team meeting | ✅ |

**What the analysis layer actually contains:** the two floors (random ~1/√d, and
the split-half extraction floor nobody here has ever measured), the pairwise
cosine matrix, hierarchical clustering, the permutation null, and a verdict
function deliberately written so "no separable subtypes" is a reportable result
rather than a failure.

---

## The one remaining gap

The project already has the hard half. `experiment.run` does:

> generate → capture residuals → judge → **bucket by verdict** → build direction

That is exactly the shape Experiment 1 needs. What is missing is the adapter that
teaches it to speak BBQ:

1. **A BBQ judge** (~60 lines). The registry expects
   `(responses, examples, JudgeSpec) -> list[label]`. Ours calls
   `resolve_answer_roles` + `parse_choice` and returns
   `"biased"` / `"unknown"` / `"other"` / `None`. All the logic behind it is
   already written and tested — this just wires it in.
2. **A run config per category** (~10 short files, or one parameterised). Copy
   `configs/example_bbq.py`; set dataset to the category's `.jsonl`, filter to
   `context_condition == "ambig"`, method `mean_diff`, contrast
   `("biased", "unknown")`.
3. **A base-rate script** (~30 lines) that runs step 1 of the procedure and
   prints the stereotyped-answer rate per category.

**None of this needs a GPU or a decision.** It can be written and unit-tested
tonight, which would make tomorrow purely "get on the box and run it."

---

## Tomorrow, in order

| # | step | needs GPU | ~time | why it is here |
|---|---|---|---|---|
| 0 | `git pull` — see what Farhan/Edward pushed | no | 1 min | they may have changed the pipeline |
| 1 | Lambda access, walked through end to end | — | 20-30 min | Jeremiah has not SSH'd to Lambda before |
| 2 | Verify env: `torch.cuda.is_available()`, `transformer_lens` imports | yes | 5 min | the exact thing that blocks G1 |
| 3 | Load `qwen-1.8b`, confirm shapes | yes | 5 min | cheapest possible failure |
| 4 | **Base rates** — stereotyped-answer rate per category | yes | 20-40 min | **decides whether the experiment is viable at all** |
| 5 | Extract one direction per usable category | yes | 1-2 hr | the actual data |
| 6 | Noise floor — split-half re-extraction per category | yes | +30 min | reuses cached activations, cheap |
| 7 | Cosine matrix, clustering, permutation null | no | minutes | already written and tested |
| 8 | Read the verdict, interpret clusters last | no | — | interpretation comes last, always |

**Stop at step 4 and look at the numbers before spending money on step 5.**

---

## Risks, honestly

### 1. Base rates too low — the real scientific risk

The direction is built by contrasting responses where the model **was** biased
against ones where it **was not**. If `qwen-1.8b` almost never picks the
stereotyped answer on ambiguous items, the biased bucket is too small to average
and no direction can be built for that category.

Nobody has measured this. It is unknowable until step 4, which is why step 4 is
early and cheap.

*If it happens:* try a larger model (bias rates generally rise with capability —
the 2025 table shows Qwen1.5-14B far more opinionated than 1.8B), or fall back to
the prompt-level contrast, which needs no generation at all. Not fatal, but it
would cost a day.

### 2. TransformerLens / environment on the box

Unverified for us. It is the same blocker G1 has been sitting on. Step 2 resolves
it in five minutes. If Edward has already gotten G1 running, this risk is gone.

### 3. The noise floor could come back low

If re-extracting the *same* category gives a cosine of ~0.6, then no pair of
categories can be called distinguishable at this sample size.

**This is a result, not a failure** — and it would be the first measurement of
that number anyone on this project has. It is worth reporting either way, and
the verdict function is written to say so plainly.

### 4. Time

Numbers freeze Mon Aug 24 (4 days), deadline Fri Aug 28. Whether our work is in
*this* paper is question #7 for the team. If it is, the schedule is tight; if it
is the next paper, there is room.

---

## Known limitations to state in any writeup

- **19.6% of ambiguous rows cannot be labelled** and are excluded by design — not
  lost, refused. Mostly the intersectional sets, where both answers share the
  stereotyped race and BBQ names only the race.
- **Age resolves at 77.8%**, down from an apparent 100% under the old naive rule,
  because ambiguous rows are now refused rather than mislabelled.
- **Only BBQ so far**, so all 10 topics share one prompt format. That makes the
  within-BBQ comparison clean, but it also means we cannot yet tell whether
  clustering tracks topic or format. Adding a second source needs **two** topics
  from it, never one.
- Politics is **not** in BBQ. Phase 2.

## What is genuinely good about this design

- **No LLM judge.** BBQ is multiple choice and on ambiguous items the correct
  answer is the unknown option, so scoring is a string match. Every other number
  in this project carries a provisional judge version. Ours will not.
- **The experiment can come back negative** and still be publishable, because the
  floors and the null are built in from the start rather than added under review.
- **A free sanity check** exists if the intersectional sets become usable: they
  should land between their parent categories.

---

## Queued next — bias STEERING (Jeremiah's call, 2026-08-20)

**Green-lit conditionally: if Experiment 1 goes well tomorrow, run this.**

Once we have a direction per category, test whether those directions *cause*
behaviour rather than merely correlate with it — add the vector and see if
stereotyped answering goes up, subtract it and see if it goes down, then check
whether a race vector moves race prompts more than gender prompts.

Full design, the controls it needs, and the one overclaim to avoid:
**`notes/06-steering-extension.md`**.

Cheap for us, because the direction Experiment 1 produces is the same object
`steering.apply_resid_pre_add` already steers with. It is a config change, not a
build. But it comes **after** the extraction floor and the permutation null — a
steering result built on directions we have not shown to be reproducible cannot
be interpreted.

## If you read nothing else

Nothing is broken, nothing is blocked, and you do not need answers from anyone
before tomorrow. One small adapter is left to write and it needs no GPU. The one
thing that could genuinely derail Experiment 1 is base rates, and we find that
out in the first half hour on the box — before spending real money.
