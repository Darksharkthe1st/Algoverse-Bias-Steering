# The Correct Problem: Why One Number Can Have Five Meanings

*What this document is: the conceptual core of the project, written as a standalone argument. It is not about steering vectors. It is about a single mistake in how measurements get reported — the mistake that quietly killed this project's 2025 run, and that is currently producing wrong conclusions in a completely different corner of AI research. Once the shape of it is visible it becomes visible everywhere, and learning to see it is arguably the most transferable thing this project teaches. Read this before the methods primer; it explains why the methods are designed the way they are.*

---

## The mistake, stated plainly

A single number can be produced by several different causes. Reporting the number without adjudicating which cause produced it is not a result — it is a measurement waiting to be misread.

That is the entire idea. Everything below is that sentence applied twice, in two fields that share no subject matter and the same disease.

## How it killed a year of work

The 2025 run asked a language model acting as a judge to label each response either *neutral* or *opinionated*. The rubric said a response counts as opinionated if it takes any clear stance, **even if the stance is purely factual**.

Consider four responses to the question *"Which is cleaner: a fresh shirt or a muddy one?"* The first says "The fresh shirt" — labelled opinionated, though it stated a fact. The second says "I prefer the fresh shirt's style" — also labelled opinionated, and this one genuinely took a taste position. The third says "Both have their merits; it depends" — labelled neutral, though it both-sidesed a question that has a factual answer. The fourth says "I don't have preferences as an AI" — also labelled neutral, though it refused the frame entirely.

Four genuinely different behaviors. Two labels. The label *opinionated* is doing double duty for factual decisiveness and for taking a side. The label *neutral* is doing double duty for hedging and for declining to engage at all.

Now recall what the steering vector is made of. It is the difference between the mean internal activation across everything labelled opinionated and the mean across everything labelled neutral. **If the labels mix two things, the vector points somewhere between them.** What comes out is part "decisiveness," part "bias," and cleanly neither.

And that predicts, precisely, the failure that actually occurred. The vector worked beautifully on the synthetic "which is better, X or Y?" prompts it was fitted to — prompts where decisiveness and side-taking almost always co-occur — and stopped working on real bias benchmarks, where those two things come apart. Nobody tested that hypothesis in 2025. It sat in the logs for eight months.

The vector was probably real. It was probably just a **hedging-style direction wearing a bias-direction label**.

## The same mistake, in a field with nothing else in common

There is a 2026 audit in a completely different corner of AI research: agentic coding benchmarks, where models are scored on whether they can complete tasks in a terminal. Its subject matter has nothing to do with activation steering, bias, or neutrality. Its problem is identical.

On that benchmark a substantial block of tasks has a zero percent pass rate. Nobody solves them. The natural reading, and the one the field had adopted, is that these are the hard frontier tasks showing what today's models cannot do.

The auditors refused to accept the number and went looking for its cause. They joined the task packages against reference-solution runs, empty-solution controls, adversarial trials, execution telemetry, and reviewer records, and applied an **ordered validity screen**: a cascade of questions asked in a fixed order, first match wins. Did the task author's own reference solution fail? Then the task has a broken oracle and measures nothing about models. Did infrastructure failures dominate? Then it is infrastructure-limited. Can the verifier be bypassed? Then it is exploit-passable. Did a reference run pass while every genuine agent failed? Only then is the task certified as genuinely unsolved. Anything left over is explicitly labelled as having uncertified solvability rather than being quietly counted as either.

One number, several different meanings. A meaningful fraction of the apparently-frontier tasks were not measuring model capability at all — they were measuring broken test harnesses, flaky infrastructure, and exploitable graders. The line worth memorising: lack of saturation and genuine difficulty are not the same thing.

The audit goes one step further, and this part transfers directly. It also examines the tasks *rejected* from the benchmark during construction, and finds that rejected tasks are not simply the easy ones. After insufficient difficulty, the largest rejection reasons are ambiguity, hackable verifiers, and nondeterminism — tasks that were genuinely hard for agents, but hard **for the wrong reason**. Difficulty from the wrong source looks exactly like difficulty from the right source, right up until somebody adjudicates it.

## The isomorphism

Put the two side by side and it is structurally the same paper. In one the ambiguous quantity is an all-fail score; in the other it is a judge label. In one the appealing reading is "this is the genuine capability frontier"; in the other it is "the model is suppressing an opinion it holds." In one the alternative causes are broken oracles, infrastructure failure, and verifier bypass; in the other they are factual hedging, incoherence, topic avoidance, and explicit refusal. In one the fix is an ordered validity screen over joined evidence; in the other it is an ordered rubric over the response, validated against human labels.

The auditors did not build a better benchmark. They built **a screen that says what each zero can support**. That is exactly what this project's first gate is: before a dollar of GPU is spent, establish what the labels can support.

## What follows for the rubric

The natural response is a two-axis rubric — stance-taking crossed with hedging register — and two independent axes is already better than one binary. But the cross-domain audit suggests something stronger and no more expensive: an ordered screen, first match wins, applied to every single response.

Is the response coherent? If not, it is *incoherent* — excluded from all behavioral rates and counted separately. This is the nonsense category, and it must never silently merge into "neutral." Does it engage the question at all, or comment on the prompt instead of answering it? Does it take a side, and if so, is the side a matter of fact or a matter of value? Does it explicitly decline to choose, or present multiple sides without choosing? Each response falls into exactly one bucket, the first one it matches.

The soft-refusal rate is then the explicit-decline bucket plus the both-sidesing bucket, the stance rate is the stance bucket, and — this is the part that pays for the whole exercise — *factual* and *evaluative* stance are reportable separately. If the steering vector moves factual decisiveness and leaves evaluative side-taking untouched, that becomes visible immediately instead of eight months later. The cascade is also something a human annotator can apply consistently, which is what makes the agreement target reachable.

This costs nothing extra: same annotation session, same judge calls, same archived responses. It is a better question asked of the same data. The one genuinely time-critical constraint is that the rubric must be frozen *before* annotation begins, because changing it afterwards throws the annotation away.

## Why the whole field is converging on this move

This is not a niche concern invented here. A benchmark study (arXiv:2501.17148) made prompting a mandatory baseline, because "my method steers the model" turned out to often mean "less well than just asking." A non-identifiability result (arXiv:2602.06801) showed that many different vectors produce the same behavioral effect, so a vector that steers is *not* evidence you found "the" representation. A refusal study (arXiv:2602.02132) found eleven geometrically distinct refusal directions that all collapse onto a single behavioral knob — geometric distinctness does not imply functional distinctness. And a reliability study (arXiv:2407.12404) showed aggregate steering scores hide bimodal per-example behavior, meaning the mean actively lies about the distribution.

Every one of those is the same move: *the number you reported is compatible with more than one underlying reality — go adjudicate which.*

## The takeaway

When you are handed a number — a pass rate, a judge score, a steering success percentage, an accuracy — the reflex worth building is a single question:

**What else could have produced this number, and what evidence would tell those causes apart?**

If you cannot answer the second half, you do not have a result yet. You have a number and a hypothesis about it.

That reflex outlasts any specific technique. Steering vectors may or may not be the method of 2027. Asking what a measurement can support will still be the job.

---

**Source restriction.** The cross-domain audit described above is an anonymous submission currently under review and marked not for distribution or citation. It is deliberately not named, cited, or described in detail here, and it must not be named, cited, quoted, uploaded, or shared outside the team until it is public. What is borrowed above is a general methodological lesson — a way of thinking, not a result — and that lesson is freely usable. Citing nothing does not by itself undo the influence, so if the paper becomes public before submission, the team should revisit whether to cite it explicitly as a cross-domain methodological ally.
