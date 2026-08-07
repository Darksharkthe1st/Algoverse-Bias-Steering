# Open Questions: What Is Genuinely Unanswered, and What Would Settle It

*What this document is: an honest inventory of what this project does not know, written after the August 2026 audits removed several things it had previously believed. Each question is stated with the evidence that currently bears on it and the specific experiment that would settle it. Some of these are answerable in a week on a laptop; one is a multi-year research programme. They are ordered roughly by how much they block everything else.*

---

## Do the archived labels measure stance-taking, or style?

This is the question everything else waits on, and it is completely open.

The 2025 judge scored any clear stance as "opinionated," including purely factual ones, so the contrast set behind every archived vector mixes decisiveness with side-taking, and a difference-of-means vector over mixed labels points between the two. That is not speculation — it is a consequence of the arithmetic, and it predicts precisely the transfer failure observed: the vector worked on synthetic comparison prompts, where decisiveness and side-taking co-occur, and stopped working on real bias benchmarks, where they come apart. The audits reproduced those labels perfectly, seven rows out of seven, and that certifies bookkeeping and nothing else. **Reproducing a label is not validating it.**

What settles it: freeze a six-way ordered screen separating factual stance from evaluative stance, double-annotate roughly 150 archived responses blind to experimental arm, and require Cohen's kappa of at least 0.70 per category. Then re-judge the archive under that screen and ask whether the archived vectors moved *stance-taking* by a meaningful margin or only *hedging register*. This costs no GPU time and is the one genuinely time-critical item on the calendar, because the rubric must be frozen before annotation begins — relabelling afterwards throws the annotation away. If the agreement target is missed, the steering work stops and the project becomes a construct-validity audit note. If the vectors moved only hedging, they are dead and the project pivots. Neither outcome is currently known.

## Is soft refusal separable from hard refusal?

Until August 2026 the project believed this had been answered. It has not. The only archived experiment bearing on it was **invalid, not null** — a one-dimensional tensor indexed by layer, producing a scalar broadcast across the residual width rather than a direction, compounded by an ordering mismatch that meant every run loaded a different model's vector than its label claimed. There is nothing under that result in either direction: not evidence of separability, not evidence of entanglement.

The surrounding literature has staked out both possibilities. One line finds eleven geometrically distinct non-compliance directions that all collapse onto a single shared behavioral knob (arXiv:2602.02132), predicting entanglement. Another finds that removing the harm-refusal direction shifts unrelated dispositions — models become measurably more optimistic and hedge less (arXiv:2607.17427) — and explicitly asks somebody to run this experiment.

What settles it: extract both directions in the same models under one unified convention, **dose-match the two interventions to equal on-target effect on a frozen development split before any off-target number is read**, then measure whether steering opinionation moves the safety batteries and whether steering harm-refusal moves the opinion batteries. The dose-matching is not optional bookkeeping: without it, "opinion steering doesn't move safety" is indistinguishable from "we steered opinion weakly," which is a one-sentence reviewer kill. Either outcome publishes — independence contradicts the one-knob generalisation, entanglement mechanistically explains the off-target disposition shifts.

## How much can a silent fault fabricate?

This is the project's newest question and its most distinctive. The archive produced a clean, convincing table — 1 unsafe response out of 99, then 27 out of 99 after "steering" — that measured nothing, with no crash and no warning. The follow-up nobody in the literature has answered: how often does that happen across the field, and would anyone be able to tell?

The claim that published steering nulls may be bug artifacts is unsupportable as an accusation and supportable as a measurement. Take a pipeline that demonstrably works, inject one fault at a time — scalar broadcast, wrong model's vector, extraction failures counted as a behavior, unseeded split, mislabelled column semantics — and quantify for each what effect size it manufactures from nothing, whether it produces a fake positive or a fake null, whether any standard reported statistic would reveal it, and how much of the result table changes. The output is a susceptibility profile. The scalar-broadcast case goes first, because there is a real archived instance to check the simulation against, and the whole study is arithmetic and linear algebra needing no GPU. The nearest prior art defines silent failure as invisibility in the reported numbers (arXiv:2607.02586); what it does not cover is the class where every tensor shape remains *legal* and the semantics are wrong anyway.

## How much does a direction move when you resample the contrast set?

Nobody knows, and this blocks every geometry number the project might report. The archive contains two vectors for the same model that are byte-identical — per-layer cosine of exactly 1.000 across all 32 layers. They are copies, not independent redraws, so there is no estimate anywhere of how far a direction moves when the contrast set is resampled. Without that floor, no cosine means anything: a cross-direction cosine of 0.35 is uninterpretable until it is known whether re-extracting the *same* direction from a fresh draw gives 0.97 or 0.60. In the first case 0.35 is a real dissociation, in the second it is noise. The fix is unglamorous and unavoidable — several bootstrap redraws of the contrast set per model, with the within-direction noise floor reported on every geometry panel alongside the cross-direction numbers.

## Three cheap questions that are still open

Does the depth structure survive normalisation? Per-layer vector norms span 2 to 3 times on one model family and 234 to 1391 times on the others, so any claim about *where in the network* soft refusal is mediated has to survive unit normalisation first, or it will largely be re-plotting the residual-norm profile.

Does judge normalisation change the labels? The stored responses carry a scaffold header and chat-template control tokens in 85 to 100 percent of cases plus a truncated echo of the prompt, so whatever the 2025 judge scored, it scored that noise too. Whether stripping it changes labels is a free ablation, needing only a sample re-judged twice.

Can the zero-vector ablation control be restored? Long cited as proof that "the vector does the work," it is currently unusable because that run's extraction failures have not been separated from genuine incoherence. Separating them is a CPU job that either restores a control or removes a claim.

## The long-horizon question: is a direction even the right object?

Everything above treats soft refusal as something that might be a linear direction. The 2026 literature increasingly says that framing is a local approximation, and that linear directions are **local charts on curved, low-dimensional feature manifolds**. One study finds character-count information living on a helical manifold, with sparse-autoencoder latents behaving like place-cell discretisations of a continuum (arXiv:2601.04480). A manifold-steering result finds the activation and behavior manifolds approximately isometric, with steering *along* the manifold beating linear steering at roughly 2.8 times lower intervention energy (arXiv:2605.05115). Refusal itself graduated from direction to cone to subspace (arXiv:2502.17420), and a region-based alternative to sparse autoencoders reports far higher interpretable-feature rates and better steering (arXiv:2602.02464).

Read together, these suggest the right object for "declining to take a side" may be a **region** rather than a line — which would explain, without any appeal to bugs, why adding the direction works while ablating it does not, and why it transfers inside its training distribution and not outside it.

This is the long-horizon programme, not the August work, and it is explicitly out of scope for the current sprint: no cone fitting, no manifold steering, no region methods. That cut is deliberate rather than an oversight. But it is where the field is moving, and the measurement discipline being built now is exactly what a geometry programme would need underneath it — a region result built on an unvalidated construct would just be a more expensive version of the same mistake.
