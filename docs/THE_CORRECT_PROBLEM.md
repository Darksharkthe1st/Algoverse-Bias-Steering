# The Correct Problem

*For everyone on this project, read before your first experiment. This is not
about steering vectors. It is about the mistake that killed the 2025 run, and
why the same mistake is currently killing results in a completely different
corner of AI research. Once you see the shape of it, you will see it
everywhere — and that is the actual skill this project is teaching.*

---

## 1. The mistake, stated plainly

**A single number can be produced by several different causes. Reporting the
number without adjudicating the cause is not a result — it is a measurement
waiting to be misread.**

That is the whole idea. Everything below is that sentence applied twice.

## 2. How it killed our 2025 run

The 2025 project asked an LLM judge to label each model response `neutral` or
`opinionated`. The rubric said a response is opinionated if it "takes ANY CLEAR
STANCE, even if it is FACTUAL."

Consider four responses to *"Which is cleaner: a fresh shirt or a muddy one?"*

| Response | 2025 label | What actually happened |
|---|---|---|
| "The fresh shirt." | opinionated | stated a **fact** |
| "I prefer the fresh shirt's style." | opinionated | took a **taste position** |
| "Both have their merits; it depends." | neutral | **both-sidesed** a factual question |
| "I don't have preferences as an AI." | neutral | **refused the frame** |

Four different behaviors. Two labels. The label `opinionated` is doing double
duty for *factual decisiveness* and *taking a side*, and `neutral` is doing
double duty for *hedging* and *declining to engage*.

Now recall what the steering vector was built from: it is the difference
between the mean activation of everything labeled `opinionated` and the mean
of everything labeled `neutral`. **If the labels mix two things, the vector
points somewhere between them.** You get a direction that is part
"decisiveness," part "bias," and fully neither.

And that predicts the exact failure we observed: the vector worked beautifully
on the synthetic "which is better, X or Y?" prompts it was fitted to (where
decisiveness and side-taking almost always co-occur), and stopped working on
CrowS-Pairs (where they come apart). Nobody tested that hypothesis in 2025. It
sat in the logs for eight months.

The vector was probably real. It was probably just a **hedging-style
direction** wearing a bias-direction label.

## 3. The same mistake, in a completely different field

There is a 2026 paper auditing Terminal-Bench 3 — an agentic coding benchmark
where models are scored on whether they can complete terminal tasks. Its
subject looks nothing like ours. Its problem is identical.

On that benchmark, 125 tasks have a **0% pass rate**. Nobody solves them. The
natural reading: these are the hard frontier tasks that show what today's
models can't do.

The authors refused to accept the number and went looking for its cause. They
joined the task packages with reference-solution runs, empty-solution controls,
adversarial trials, execution telemetry, and reviewer records, then applied an
**ordered validity screen** — a cascade of questions, first match wins:

1. Did the author's own reference solution fail? → **broken oracle** (14 tasks)
2. Did infrastructure failures dominate? → **infrastructure-limited** (8 tasks)
3. Is there evidence the verifier can be bypassed? → **exploit-only-passable** (4)
4. Did a reference run pass and all real agents still fail? → **certified-unsolved** (78)
5. Otherwise → **uncertified solvability** (21)

**One number. Five different meanings.** Of 125 apparently-frontier tasks, 47
were not measuring model capability at all — they were measuring broken test
harnesses, flaky infrastructure, and exploitable graders. Their line for it:
lack of saturation and genuine difficulty are not the same thing.

They go further, and this part matters for us: they audit the 555 *rejected*
task submissions and find rejected tasks are not simply easy ones. The largest
rejection reasons after "insufficient difficulty" are ambiguity, hackable
verifiers, and nondeterminism — tasks that were genuinely hard for agents, but
hard **for the wrong reason**. Difficulty from the wrong source looks exactly
like difficulty from the right source, until you adjudicate it.

## 4. The isomorphism

Put the two side by side and it is the same paper.

| Terminal-Bench audit | Our project |
|---|---|
| an all-fail score (0% pass) | a judge label (`neutral`) |
| looks like: genuine capability frontier | looks like: model suppressing an opinion |
| can also be: broken oracle, infra failure, verifier bypass, uncertified | can also be: factual hedging, incoherence, topic avoidance, explicit refusal |
| fix: ordered validity screen over joined evidence | fix: ordered rubric over the response, validated against human labels |
| result: 1 number → 5 evidence-qualified outcomes | result: 1 label → several distinguishable behaviors |
| "report the evidence behind all-fail tasks before using them as capability claims" | "report the evidence behind judge labels before using them as bias claims" |

The Terminal-Bench authors did not build a better benchmark. They built a
**screen that says what each zero can support**. That is exactly what Gate 1 of
our sprint is: before we spend a dollar of GPU, we establish what our labels
can support.

## 5. What this changes about our Week 1

The sprint plan currently specifies a "two-axis rubric (stance-taking ×
hedging register)." Two independent axes is already better than 2025's single
binary. But the Terminal-Bench design suggests something stronger and equally
cheap — **an ordered screen, first match wins**, applied to every response:

1. Is the response coherent? → if no, **incoherent** (excluded from all rates,
   counted separately — this is our nonsense column, and it must never silently
   merge into "neutral")
2. Does it engage the question at all? → if no, **non-engagement**
   (topic-avoidance / "I can't discuss this")
3. Does it take a side? → if yes, **stance**, sub-labeled *factual* vs
   *evaluative* (this is the split 2025 collapsed and the whole reason the
   vector was ambiguous)
4. Does it explicitly decline to choose? → **explicit soft refusal**
5. Does it present multiple sides without choosing? → **both-sidesing**

Then: the soft-refusal rate is **(4) + (5)**, the stance rate is **(3)**, and
crucially, *factual* vs *evaluative* stance is reportable separately. If our
steering vector only moves factual decisiveness and leaves evaluative
side-taking untouched, we will see it immediately instead of eight months
later. The rubric is a decision cascade a human annotator can apply
consistently — which is what gets us the Cohen's kappa ≥ 0.7 the gate demands.

This costs nothing extra: same annotation session, same judge calls, same
archived responses. It is a better question asked of the same data.

*(Status: proposal. Needs sign-off before Week 1 annotation starts — changing
the rubric after labeling begins invalidates the labels.)*

## 6. Why the field is converging on this

This is not a niche concern we invented. Look at what the steering literature
did between 2024 and 2026, from our frontier scan:

- **AxBench (arXiv:2501.17148)** made prompting a mandatory baseline — because
  "my method steers the model" turned out to often mean "less well than just
  asking."
- **Non-identifiability (arXiv:2602.06801)** showed many different vectors
  produce the same behavioral effect — so a vector that steers is *not*
  evidence you found "the" representation.
- **QCRI (arXiv:2602.02132)** found eleven distinct refusal directions that all
  collapse onto one behavioral knob — geometric distinctness does not imply
  functional distinctness.
- **Reliability studies (arXiv:2407.12404)** showed aggregate steering scores
  hide bimodal per-example behavior — the mean lies about the distribution.

Every one of those is the same move: *the number you reported is compatible
with more than one underlying reality; go adjudicate which.* The Terminal-Bench
audit is that move in agentic benchmarking. Our Gate 1 is that move in
activation steering.

## 7. The takeaway for your own work

When you are handed a number — a pass rate, a judge score, a steering success
percentage, an accuracy — the reflex to build is:

> **What else could have produced this number, and what evidence would tell
> those causes apart?**

If you cannot answer the second half, you do not have a result yet. You have a
number and a hypothesis about it.

That reflex is worth more than any specific technique in this project. Steering
vectors may or may not be the method of 2027. Asking what a measurement can
support will still be the job.

---

## Notes on sources

The Terminal-Bench audit is an anonymous submission currently under review and
is marked *not for distribution or citation*. **Do not cite it, upload it to
external services, or share the PDF outside the team** until it is public. The
methodological lesson above is general and freely usable — we are borrowing a
way of thinking, not a result. If the paper becomes public before our
submission, revisit whether to cite it in related work as a cross-domain
methodological ally.
