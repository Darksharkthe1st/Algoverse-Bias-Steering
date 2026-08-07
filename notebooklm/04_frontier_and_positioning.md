# The Frontier in 2026, and Where an Honest Contribution Can Still Live

*What this document is: a map of where activation steering research actually stands in mid-2026, and an honest assessment of which parts of this project's story are already claimed by other people and which parts are genuinely open. It covers the nearest competing papers with their arXiv identifiers, the reasons the 2024 recipe is now a baseline rather than a method, the shift from directions to geometry, and the specific narrow gap the project has decided to occupy. It ends with the venue decision, which forces a choice the team has not yet made.*

---

## The 2024 recipe is a baseline now

The method this project inherited — take activations from two contrasting sets of responses, subtract the means, add the difference back at inference time — was state of the art when it was published for refusal (arXiv:2406.11717). By 2026 it is the thing new methods are measured against, and the measuring has not been kind.

The benchmark that reset expectations is AxBench (arXiv:2501.17148). Run head to head, plain prompting and finetuning beat every representation-level steering method tested. Difference-in-means survives as the best *cheap* intervention; sparse-autoencoder features land near the bottom. The practical consequence is that a system-prompt baseline is now mandatory. Any paper claiming a steering result without showing that simply asking the model does worse will be asked why, and the answer had better not be "we didn't check."

The reliability literature did the second half of the damage. A generalization-and-reliability study (arXiv:2407.12404) showed enormous per-example variance hiding under aggregate steering scores — the mean genuinely misrepresents the distribution. A large replication across 36 models and 14 families (arXiv:2504.04635) found many models where the effect is simply zero, which makes single-family validation disqualifying rather than merely thin. In 2026 this line added pre-intervention predictors of whether a given model will be steerable at all (arXiv:2602.17881, arXiv:2604.15557).

Then came the result that reframes the entire enterprise: **non-identifiability** (arXiv:2602.06801). Many different vectors produce the same behavioral effect, and the equivalence class includes vectors produced by independent extraction pipelines. A vector that steers your model is therefore not evidence you found *the* representation of anything. This is the objection every direction-finding paper now has to answer, including this one, and it is the reason this project's terminology rules insist on "*a* direction," never "*the* direction."

The recipe itself kept improving in parallel. Conditional activation steering (arXiv:2409.05907) fires an intervention only when a condition vector detects the relevant context. Affine concept editing (arXiv:2411.09003) treats refusal as an affine function — projection plus addition relative to a reference point — rather than a pure additive direction. Hypernetwork-generated vectors (arXiv:2506.03292) match prompting on the AxBench evaluation. And activation *capping* — clamping a projection at a percentile rather than adding a constant multiple — has largely displaced fixed coefficients, which is directly relevant here given what the audits found about per-layer norms.

There is also now a side-effect literature, which matters because it is where honest negative results have been landing. Steering toward a narrow behavior can induce broad emergent misalignment (arXiv:2606.08682) and silently weaken safety (arXiv:2603.24543). One preregistered study (arXiv:2607.17427) shows that removing the refusal direction shifts unrelated dispositions — models become measurably more optimistic and hedge less — and explicitly asks somebody to run the dissociation experiment this project has been circling.

Meanwhile the technique went into production. One major lab industrialised precisely this pipeline as persona vectors (arXiv:2507.21509), using it for monitoring, for preventative steering during finetuning, and for flagging training data, and shipped a deployed-grade activation-capping result (arXiv:2601.10387). Another ships activation-based probes. And a startup deprecated its self-serve steering API in early 2026: monitoring with directions found product-market fit, consumer-facing steering did not.

## Geometry: directions became charts on manifolds

The second shift is conceptual. The consensus moved from "features are directions" toward "linear directions are local charts on curved, low-dimensional feature manifolds."

The flagship demonstrations are a lab study showing that character-count information lives on a helical manifold and that sparse-autoencoder latents behave like place-cell-style discretisations of a continuum (arXiv:2601.04480), and a manifold-steering result (arXiv:2605.05115) showing that the activation manifold and the behavior manifold are approximately isometric, and that steering *along* the manifold beats linear steering at roughly 2.8 times lower intervention energy.

Refusal specifically graduated from direction to cone to subspace. Concept cones (arXiv:2502.17420) showed refusal is not represented by one unique direction. A follow-up (arXiv:2602.02132) found eleven distinct non-compliance directions that nonetheless collapse onto one shared behavioral knob. And in reasoning models, refusal turns out to be jointly encoded in the residual stream and in the chain-of-thought text: static steering achieves 39 percent compliance with the reasoning trace held fixed, 70 percent with it removed, and 94 percent with it regenerated under steering (arXiv:2605.26772). That paper tests exactly one model and runs no non-reasoning control — a hole a within-checkpoint thinking toggle could fill.

Sparse autoencoders are alive as infrastructure and demoted as ontology: large open latent sets ship, but at least one major lab publicly pivoted from sparse-autoencoder basic science toward probes and model biology after negative downstream results, and there are now results showing these models dilute manifolds across redundant latents (arXiv:2604.28119) with geometry-walled scaling behavior (arXiv:2605.09887). The strongest current alternative for steering is a mixture-of-factor-analyzers region method (arXiv:2602.02464) reporting roughly 96 percent interpretable features against roughly 29 percent.

## The bias and neutrality niche is crowded, but the exact claim is unclaimed

The closest neighbour by subject is a December 2025 paper on refusal steering (arXiv:2512.16602) that removes refusal on politically sensitive topics at 80-billion-parameter scale, taking political refusals from 92 percent to 24 percent while retaining 99 percent of safety-benchmark performance. It frames the work as censorship removal and finds the signal *distributed* rather than single-direction. Two axes separate it from this project: its intervention does not distinguish a model that engages without taking a side from a model that refuses, and its contribution is refusal removal rather than a factorization. This project must never frame itself as "uncensoring."

Political-neutrality steering also exists (arXiv:2508.08846), with a multilingual extension across 50 countries finding that ideology directions do not align across languages by default (arXiv:2601.23001), and a continuous censorship dial (arXiv:2504.17130). The distinction those papers force is between **opinionation** — whether the model takes *a* side, a scalar — and **ideology** — *which* side, a direction. They steer which; this project steers whether.

The labs professionalised measurement in parallel: an open-sourced paired-prompts even-handedness evaluation whose stated ideal is symmetric engagement rather than refusal, a five-axis political-bias framework carrying "political refusal" as a named axis, and a consistency-training paper penalising "fence-sitting" by name (arXiv:2605.22771). Benchmarks moved too: IssueBench (arXiv:2502.08395), with 2.49 million prompts across 212 issues, is the standard now, CrowS-Pairs is widely criticised, and reviewers will ask why anyone used BBQ (arXiv:2110.08193) as an opinion resource when it is a bias-in-question-answering benchmark.

The term **soft refusal** itself remains unclaimed as an activation direction as of August 2026. The one-knob finding, the distributed-signal finding, and the off-target-disposition finding all circle it from different sides without naming it.

## What the prior-art scan actually found about the forensic framing

Here the news is mixed and worth stating precisely. The pure forensic-case-study framing — "we audited an archive and found the results were bugs" — is **partially taken**. The nearest neighbour is an audit-of-audits paper (arXiv:2607.02586, July 2026) that defines a *silent* failure as one that is invisible in the reported numbers. That is close enough that a reviewer will find it, so the project has to say what it adds.

Three things are genuinely open.

The first is **silent broadcast** specifically: faults where the tensor shapes are all legal and the semantics are wrong anyway. A one-dimensional tensor indexed by layer yields a scalar that broadcasts across the residual width. Every shape check passes. Nothing crashes. That specific class has never been measured.

The second is **tensor-payload hashing as a validity control**. Hashing the file container fails when serialisation formats differ; hashing the canonical contiguous numeric payload recovers exact identity regardless of container, which is how this project recovered the provenance of every vector in a mislabelled experiment. That is a cheap, general, reusable control and nobody presents it as one.

The third is the claim that **published steering nulls may be bug artifacts**. Nobody has made it, and as an insinuation it is unsupportable — you cannot accuse a literature of bugs you have not demonstrated.

Which is why the recommended direction converts it into a measurement. Take a working steering pipeline and inject one silent fault at a time: the scalar broadcast, the wrong model's vector, judge parse failures counted as a behavior class, an unseeded split, mislabelled column semantics. For each one, measure what effect size the fault manufactures out of nothing, whether it produces a fake positive or a fake null, whether any standard reported statistic would reveal it, and how much of the result table changes. The output is a **susceptibility profile** — a quantitative statement of which silent faults produce which artifacts at what magnitude. It requires no GPU, it is almost entirely arithmetic and linear algebra, and it turns an accusation the project cannot support into a measurement it can. The scalar-broadcast case is the one to run first, because there is a real archived example to check the simulation against.

## The model set is chosen for controls, not for vintage

The 2025 model set is 2023–24 vintage and is a reviewer liability on its own, but that is not the reason to refresh it. The reason is that the current landscape hands over two controlled experiments for free.

Two checkpoints in the same family, both 27 billion parameters, have **byte-identical configurations** — same architecture string, 64 layers, 5120 model width, same head counts, same feedforward width, same vocabulary — and differ *only in post-training*. That makes one question answerable with no confounds at all: does the refusal direction move under re-alignment when the architecture is held byte-for-byte fixed? It also doubles as a **positive control for the entire apparatus**. If the pipeline cannot detect a difference between two checkpoints that genuinely differ, the pipeline cannot detect anything — and it is much better to learn that before spending the budget than after.

The second free contrast is a dense 31-billion-parameter model against a mixture-of-experts sibling of the same family, same recipe, same tokenizer, same release date. A controlled dense-versus-MoE comparison with no confounds. The 2026 literature already reports that plain difference-in-means works fine on mixture-of-experts models and that expert-aware variants underperform it (arXiv:2606.04160), with the practical caveat that rarely-routed experts give unreliable mean estimates unless all experts are forced to produce outputs when computing directions.

## The venue decision, which is a real fork

The target is a discovery-oriented interpretability workshop at NeurIPS 2026, deadline August 29, 2026 anywhere-on-Earth, five main-text pages plus references and appendix, non-archival and double-blind. Its call for papers explicitly invites failure cases and negative results, which is unusually well matched to a paper whose strongest material is two retracted findings and a susceptibility profile. Non-archival status also means the work stays eligible for a fuller venue later.

There is an alternative — a science-of-interpretability workshop at the same conference with an August 28 deadline — but it **forbids concurrent submission to another NeurIPS workshop**. The team must pick one. That decision has not been made and it is a real fork, not a formality.
