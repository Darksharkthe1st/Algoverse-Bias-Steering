# Project Overview: Soft Refusal, Steering Vectors, and What Happens When You Audit Your Own Result

*What this document is: an orientation briefing for anyone encountering this research project for the first time. It explains what soft refusal is, why a group of students set out to find it inside a language model's activations, what they built in 2025, what happened when that work was independently re-audited in August 2026, and what the project is now trying to publish. Everything here is drawn from the project's own committed documents and verification scripts. Read this first; the other documents in this pack go deeper on verification, methodology, prior art, and open questions.*

---

## The behavior at the centre of the project

Ask a large language model a question with a genuine disagreement inside it — should a city prioritise bike lanes or parking, is a particular policy a good idea, which of two artists is more original — and one very common thing it will do is refuse to pick. It will not refuse the way it refuses a request for weapons instructions. It will engage warmly, lay out considerations on both sides, and then decline to land anywhere. "Both have their merits." "It depends on what you value." "As an AI, I don't have personal opinions."

This project calls that **soft refusal**, and the term is doing real work, because it must be kept apart from three neighbours that look similar and are not the same thing. **Hard refusal** is declining a harmful request, the behavior studied in the paper this project's method descends from. **Over-refusal** is wrongly refusing a benign request. **Abstention** is declining a question that genuinely has no answer. Soft refusal is none of those: the model can answer, the question is not dangerous, and it engages — it simply declines to take a side.

The reason this matters beyond curiosity is that the major labs have converged on it as a named failure mode rather than a virtue. One lab's published even-handedness evaluation states that the ideal is *symmetric engagement*, not refusal. Another lab's political-bias framework carries "political refusal" as an explicit axis. A third line of work penalises "fence-sitting" by name in its training objective. Everybody wants a name and a measurement for this behavior. As of August 2026, no published paper names an activation direction for it.

## What the 2025 project built

The original work was a summer research project whose repository README read: *"Revealing hidden biases by finding steering vectors for neutrality."* The underlying bet was that soft refusal is mediated by a single linear direction in the model's residual stream — the running vector of internal state that every transformer layer reads from and writes to. If you can find that direction, adding it should make the model hedge, and subtracting it should make the model commit. And if the model has an opinion it normally suppresses, subtracting the direction should surface it. That was the "hidden biases" premise.

The method was a direct port of the refusal-direction recipe from the interpretability literature. Generate model responses to a large set of synthetic forced-choice prompts — "which is more creative, friends or dreams?" — have a judge model label each response *neutral* or *opinionated*, cache the internal activations, and define the steering vector as the mean activation of the opinionated responses minus the mean activation of the neutral ones. Then, at generation time, add a scaled copy of that vector back into the residual stream at every layer and watch the behavior move.

It worked, in-distribution, and it worked hard. Across seven models the effect is large and bidirectional. And it failed, out-of-distribution, in exactly the places that mattered most: vectors trained on the synthetic comparison prompts largely stopped working on real bias benchmarks. The "revealing hidden biases" premise never survived contact with those benchmarks. A December pivot to a different extraction strategy produced degenerate results, and the repository went quiet mid-pivot, on a commit message that read: *"Patch up bugs in code, still unsuccessful synthetic steering."*

## What August 2026 changed

The project was revived nine months later, and the first thing the revival did was not run an experiment. It audited the archive. Two independent recounts, six days apart, using different code over different artifact families, asked one question: are the 2025 numbers real?

The headline number is real. The main steering result reproduces exactly — every row, from per-record artifacts, twice. That is the anchor the whole revival sits on, and it held.

Two other things did not survive. A result that had been circulating as a finding — that the soft-refusal and hard-refusal directions failed to cross-apply in either direction — turned out to rest on an experiment that was **invalid, not null**. The archived refusal vectors were the wrong shape, and the steering code silently turned them into a constant offset rather than a direction; separately, a bookkeeping mismatch meant each run loaded a different model's vector than the one it was labelled with. Nothing crashed. The output table looked clean and convincing and measured nothing. That claim is retracted, and the soft-versus-hard refusal relationship is now formally **untested**.

The second casualty was the judge. The 2025 rubric scored any clear stance as "opinionated" — *even a purely factual one*. That means the contrast set behind every archived vector mixes "took a side" with "stated a fact," and a difference-of-means vector over mixed labels points somewhere between them. The judge is now retired as a hard rule rather than softened with a caveat: no label from it may appear in any new analysis. Reproducing those labels, which the audits did perfectly, validates the bookkeeping and says nothing about the construct.

## Where the project stands and what it is now

The revival is a small team — a pipeline owner who wrote the 2025 code and is first author, a measurement-and-geometry lead, and a third member on annotation, quantitative audits, and writing — working to a workshop deadline at the end of August 2026 with a modest cloud-GPU budget. It meets three evenings a week.

The framing has moved with the evidence. It is no longer only "find a soft-refusal direction." A prior-art scan found that the pure forensic-case-study version of this work is partially taken. What remains genuinely open is narrower and sharper: silent faults that leave tensor shapes *legal* but semantically wrong have never been measured; hashing the numeric payload of a tensor rather than its file container turns out to be a usable validity control; and nobody has yet made the claim that some published steering nulls may be bug artifacts rather than findings. The recommended direction converts that last insinuation into a measurement — an **injected-fault study** that takes a working pipeline, injects one silent fault at a time, and quantifies what effect size each fault manufactures out of nothing and whether any standard reported statistic would catch it.

That is the project: a group that went looking for a direction, found a real steering effect and two fake results in its own archive, and decided the more useful paper is about how the fakes stayed invisible.
