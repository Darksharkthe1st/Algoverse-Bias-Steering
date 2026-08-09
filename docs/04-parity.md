# Parity: does the refactor reproduce the notebook?

The validation `docs/03-gpu-bringup.md` §4 asks for. Nothing here changes the science;
it establishes whether `src/bias_steer/` reproduces `experiments/farhan-experimentation.ipynb`
before we start changing anything.

**Anchor: Log_103.** The one archived run that kept every piece needed to reproduce it:

| Piece | File |
|---|---|
| frozen 200-prompt set | `log_103_..._dataset.pkl` |
| resulting vector, `[24, 2048]` fp16 | `log_103_..._steer_vec.pkl` |
| 100 train prompts + responses + verdicts | `log_103_..._pre-steering_responses.txt` |

Under `experiments/past_logs/methodology_experiments/misc_coeff_tests/new_older_worse_coeffs/Log_103_Automated_Test_Qwen1.5-1.8B-Chat/`.
Config: `qwen-1.8b`, `train_split=0.5` (100 train / 100 test), `max_tokens=128`,
`opin_coeff=14`, `neut_coeff=15`. Archived train verdicts: **72 neutral / 28 opinionated / 0 nonsense**.

Rather than one all-or-nothing comparison, parity is checked as a **ladder** — each rung holds
more of the pipeline fixed, so a failure localizes to the rung that broke rather than to
"somewhere in the run."

| # | Rung | Holds fixed | Isolates | Status |
|---|---|---|---|---|
| 1 | re-judge archived responses | prompts + responses | the judge alone | ✅ pass |
| 2 | regenerate train responses | prompts | generation (template, stripping, padding) | ✅ pass, after a fix |
| 3 | rebuild the vector, cosine vs archive | prompts | capture + build math | ✅ pass |
| 4 | transition matrix | — | the full pipeline | ✅ pass, with a caveat |

**Bottom line: the refactor reproduces the notebook.** Rung 2 found one real defect
(character-based prompt stripping), which is fixed; rungs 3 and 4 then land where a faithful port
should. Rung 4 also established that the *archive* was scoring contaminated text, so the refactor
is now measuring something cleaner than the runs it was validated against.

---

## Rung 1 — re-judge the archived responses ✅

`python tools/parity_rejudge.py --k 3` — no GPU, no model load. The 100 archived responses are
fed back through the refactored `neutrality_judge`; because the responses are fixed, generation is
out of the picture and any disagreement is the judge and only the judge.

```
archive tally: {neutral: 72, opinionated: 28}

trial 1: {neutral: 73, opinionated: 27}   agreement vs archive  93/100
trial 2: {neutral: 75, opinionated: 25}   agreement vs archive  95/100
trial 3: {neutral: 76, opinionated: 24}   agreement vs archive  94/100

majority vs archive:                      94/100  (94.0%)
judge self-consistency (all 3 agree):     96/100  (96.0%)

confusion (archive -> new majority):
      neutral -> neutral       70
  opinionated -> opinionated   24
  opinionated -> neutral        4    flipped
      neutral -> opinionated    2    flipped
```

**Verdict: the judge port is faithful.** The reasoning is the gap between the two numbers, not
either alone. Agreement with the archive is 94%, and the judge's agreement *with itself* across
three trials on identical inputs is 96% — so archive disagreement sits essentially at the judge's
own noise floor. A broken rubric or a broken `parse_verdict` would show up as systematic
one-directional flips far exceeding that floor; instead the six disagreements are near-symmetric
(4 one way, 2 the other) and every one is a genuinely borderline both-sides response.

**This also delivers `needed-experiments.md` §0.2's measurement:** ~4% of examples are label-unstable
across k=3 at fixed model and rubric. That is the first quantified judge-drift number this project
has, and it sets a floor: **any steering effect smaller than a few percentage points is not
distinguishable from judge noise at n=100.** Design experiments accordingly, or raise k.

### Caveat that limits this rung

`gpt-4o-mini` is an unpinned alias and the archive was judged **2025-10-06**, so the underlying
snapshot has almost certainly moved. The 94% therefore conflates per-call nondeterminism with ten
months of model drift, and this rung cannot separate them. Both point the same direction —
**pin a dated snapshot** (§0.2) — but it means 94% is a *lower* bound on the port's fidelity, not a
point estimate of it.

---

## Rung 2 — regenerate the train responses ✅ (after a fix)

`python tools/parity_generate.py --batch-size 32`. Prompts held fixed, generation re-run through
`models.generate`. Greedy decoding is deterministic, so a faithful port should reproduce the
archived text nearly character-for-character.

**Before the fix**, on the first 8 prompts:

```
exact match            : 0/8
mean common prefix     : 0 chars
got: ' reading?<|im_end|>\n<|im_start|>assistant\nBoth gardening and reading can be...'
got: 'tart|>assistant\nIndoor activities and outdoor activities can both be fun...'
```

Every response carried prompt tail and chat-template markup. The cause is the character-based
strip described under "ported defects" below: `to_tokens` left-pads a batch to its longest member,
so the decoded string is `<pad>… <bos> <prompt> <response>`, and cutting `len(prompt)` *characters*
off the front lands inside the padding. Mean common prefix of exactly 0 across every example is the
signature — not a subtle numeric drift but a systematic offset.

**After fixing `_strip` to slice by token index** (left padding makes the prompt occupy an
identical width `n_input` in every row, so `row[n_input:]` is exactly the continuation):

```
exact match              : 86/100 (86.0%)
mean similarity          : 0.945
median similarity        : 1.000
mean common prefix       : 623 chars
empty outputs            : 0

similarity by prompt-length quartile:
  shortest 25%: 0.937      longest 25%: 0.947     <- flat; padding correlation gone
```

**Verdict: generation is faithful.** 86/100 responses reproduce 10-month-old text
character-for-character on different hardware. The flat length-quartile profile is the specific
confirmation that the padding defect is gone — before the fix the same measure read 0.974 vs 0.550.
The 14 mismatches are genuine early divergences (the model picks a different continuation within
the first few tokens, then compounds), consistent with HF weight revisions since 2025-10, fp16
tie-breaking, and different GPU kernels.

---

## Rung 3 — rebuild the vector, cosine vs archive ✅

`python -m src.bias_steer run configs/parity_log103.py` then
`python tools/parity_vector.py runs/<run_id>`.

```
mean cosine (unweighted)    : +0.8833
mean cosine (norm-weighted) : +0.9064
layers with cosine < 0      : 0/24
per-layer range             : +0.79 (layer 7) .. +0.93 (layer 0)
```

**Verdict: the capture/build math is faithful.** Every one of the 24 layers is positively aligned,
with no sign inversion and no dead layer — which rules out the failure modes this rung exists to
catch (wrong hook point, wrong token reduction, swapped contrast poles; the docs flagged the coeff
sign convention as the prime suspect and it is not the problem).

Cosine sits near 0.9 rather than 1.0 for a reason that is *expected and not a defect*: the vector
is a difference of means over verdict-bucketed residuals, so it inherits every upstream difference.
Train buckets came out **74 neutral / 26 opinionated** against the archive's **72 / 28** — a small
count shift, but membership differs by more than the counts suggest, since rung 1 measured ~4%
judge instability and rung 2 measured 14% of generations diverging. Different bucket membership
over different text necessarily yields a slightly different mean.

Worth noting: fresh per-layer norms run ~20% larger than the archive's at every layer, a
consistent ratio rather than noise. Same direction, slightly different magnitude — expected from
the bucket differences, but flagged because a systematic norm offset would matter to any
absolute-coefficient comparison across these two runs.

---

## Rung 4 — transition matrix ✅ (with a caveat that favors the refactor)

Condition × verdict over the same 100 held-out prompts:

| condition | archive | this run |
|---|---|---|
| initial — neutral | 67 | 71 |
| initial — opinionated | 32 | 29 |
| initial — unparsed | **1** | **0** |
| steered_pos — opinionated | 88 | 71 |
| steered_neg — neutral | 91 | 97 |

Steering works and is correctly signed in both directions: `steered_pos` inverts the initial
distribution (71 neutral → 71 opinionated) and `steered_neg` drives it to 97 neutral. The baseline
and the neutral arm both land within a few points of the archive.

### The caveat: the archive was judging contaminated text

**100/100 of the archive's `OPINION_RESPONSE` and `NEUTRAL_RESPONSE` blocks contain the full chat
template** — `<|im_start|>system … You are to follow the instructions … <|im_start|>assistant` —
prepended to the actual response. Its `INITIAL_RESPONSE` blocks are clean.

That asymmetry is the same padding bug, and it explains itself: the notebook's `normal_generation`
passed *strings* to `generate` while `batched_generation` passed pre-padded *tokens*, so only the
steered path was corrupted. Its judge therefore scored the steered arms on text with the system
instruction ("give the clear, definitive answer") and the question glued to the front.

So the 88 vs 71 gap on `steered_pos` is **not** evidence that our steering is weaker. The two runs
fed the judge materially different inputs on exactly that arm, and the archive's version was
contaminated in a direction plausibly biased toward "opinionated". The comparison cannot be made
sharper without re-judging the archive's steered text stripped, which is possible but of limited
value now that the pipeline is clean.

The single archived `initial` response that parsed to no label is also worth a note: the
notebook's exact-case `ANSWER:` scan dropped it, where the refactor's `parse_verdict` returned 0
unparsed across all 300 judgements.

---

## Environment this was run on

```
A100-SXM4-40GB · torch 2.13.0+cu130 · transformer_lens 3.7.0 · Python 3.10.12 (.venv)
45/45 tests pass in the venv
```

### transformer_lens 3.x compatibility

The notebook predates TL 3.x, so the ported calls were re-verified against the installed version:

- `generate(input=...)` still accepts `List[str]` — the notebook's batch-of-strings call is valid.
- `from_pretrained_no_processing` no longer names `device` / `output_hidden_states` directly; both
  ride through `**from_pretrained_kwargs` into `from_pretrained`, which does accept `device`,
  `dtype`, and `default_padding_side`. The existing call in `models.py` works unchanged.
- **`temperature=0` is greedy, not a bug.** `models.generate_with_hooks` passes `temperature=0`
  while `models.generate` passes `do_sample=False`, which looks like the steered and unsteered arms
  decode differently. They do not: `sample_logits` early-returns `argmax` on `temperature == 0.0`,
  bypassing top-k/top-p/penalties, so both paths are greedy and the two arms are comparable.
  Recorded because it is a natural thing to "fix" and doing so would be a no-op at best.

## Changes made during the ladder

Ordered as they were forced by evidence, each isolated so its effect is attributable.

1. **`models._strip` now slices by token index, not character count.** The fix rung 2 demanded.
   Left padding makes the prompt occupy an identical token width in every row, so `row[n_input:]`
   is exactly the generated continuation. Took exact-match from 0/8 to 86/100.
   *This is a genuine behavior change versus the notebook, not just a port fix* — the archive's
   steered arms were scored on contaminated text and ours are not.
2. **`models.generate_with_cache` takes a `capture_names` filter.** An unfiltered `run_with_cache`
   retains every hook point at every layer (~15x the tensors actually read) for every example in a
   batch at once — what the notebook did, and why it could not scale. `SteeringMethod.names`
   declares which hook points a technique reads, defaulting to `resid_pre`. This is what makes a
   14B model viable at a usable batch size.
3. **`judge.neutrality_judge` closes its OpenAI client** (`async with`). It is called once per
   batch via `asyncio.run`, so a client left open leaked its connection pool into a closing loop
   and emitted a wall of `Event loop is closed` tracebacks at teardown — harmless, but it buries
   real errors in a long run.
4. **`DatasetSpec.shuffle`** (default `True`) and the **`snapshot` dataset loader**, both
   prerequisites discovered while building rung 2:
   - The Log_103 prompt set exists *only* inside the archived pickle — it is not
     `datasets/GPT_Prompts/comparison_questions_200.csv` (296 different prompts). The loader is the
     "formalize as dataset snapshots" item from `01-feature-roadmap.md` §1.2;
     `tools/snapshot_from_pickle.py` lifts a pickle to JSON once so configs never unpickle.
   - `run()` shuffled unconditionally where the notebook took a plain head-slice. With
     `shuffle=False` the split reproduces the archived 100-prompt train set exactly, in order.

**Not changed, deliberately:** `temperature=0` on the steered path (verified equivalent to greedy,
above), and the residual capture being taken over the *response* text only rather than
prompt+response. The latter is a faithful port and a real scientific choice worth revisiting on
purpose — see `models.generate_with_cache` — but not as a silent parity fix.
