# Glossary

*What this document is: an alphabetical reference for every term of art used across this project, written in complete sentences rather than definitions-by-fragment so that it can be read aloud and still make sense. Where a term comes from a specific paper, the arXiv identifier is given. Terms that are specific to this project's own findings — silent failure, injected-fault study, marginal versus transition, extraction failure, norm profile, positive control — are included alongside the standard vocabulary, because they are the ones a newcomer will hear in meetings and not find anywhere else.*

---

**Abstention.** Declining to answer a question that genuinely has no determinate answer. It is one of the four refusal-family behaviors this project keeps carefully apart, and it is not what this project studies.

**Ablation, directional.** An intervention that removes a direction from the residual stream rather than adding one, by projecting the activation onto the unit direction and subtracting that component. Where addition tests whether a direction is sufficient to produce a behavior, ablation tests whether it is necessary. In the 2025 run, ablating the opinion direction did not produce neutrality.

**Activation capping.** Clamping the projection of an activation onto a direction at a fixed percentile of its observed distribution — typically the 95th — instead of adding a constant multiple of the direction. Because it is scale-aware by construction, it sidesteps the per-model coefficient tuning that consumed months of the 2025 run.

**Activation steering.** The family of methods that modify a model's behavior by editing its internal activations during the forward pass, with no retraining and no weight changes. This project's variant adds a scaled direction to the residual stream at every layer.

**Affine concept editing.** An intervention that treats a behavior as an affine rather than purely linear function — projection plus addition relative to a reference point (arXiv:2411.09003). It offers an explanation for why naive ablation can fail where the affine version succeeds.

**Arditi convention.** Shorthand in this project for the extraction and application conventions of the paper the method descends from (arXiv:2406.11717): directions extracted per layer at specific chat-template positions after the instruction, never mean-pooled across the whole sequence. The rule is that one convention is used everywhere and never mixed with another.

**Both-sidesing.** Presenting multiple positions on a question without choosing between them. It is one of the two response categories that together define the soft-refusal rate, the other being an explicit decline to choose.

**Cohen's kappa.** A chance-corrected measure of agreement between two independent annotators. This project's construct gate requires kappa of at least 0.70 *per category* — not averaged across categories — on roughly 150 double-annotated responses labelled blind to experimental arm.

**Concept cone.** The finding that a behavior like refusal is mediated not by one unique direction but by a cone of directions spanning a subspace (arXiv:2502.17420). It is why this project says "a direction," never "the direction."

**Construct validity.** Whether a measurement measures the thing it claims to measure, as opposed to something correlated with it. This is the axis on which the 2025 run failed: its labels were reproducible, internally consistent, and measuring the wrong construct.

**Contrast set.** The two groups of examples whose mean activations are subtracted to produce a direction. Its composition is the single most consequential design choice in the method, because if the two groups differ in more than one way the resulting direction points at a mixture of those ways.

**Difference-in-means.** The direction-extraction recipe at the heart of the method: take the mean activation over examples showing a behavior, subtract the mean over examples not showing it, and normalise. Benchmarking found it to be the best *cheap* representation-level intervention even as prompting beat the whole family (arXiv:2501.17148).

**Extraction failure.** A judge produced text but the parser could not pull a valid label out of it. This project's archive contains 2,032 case-insensitive "none" markers across 107 files that are extraction failures — not degeneration, not incoherence, not a behavior. Folding them into a behavior class inflates whichever condition had the most parse failures, which is why the parser must expose explicit *ok*, *no-match*, and *ambiguous* states and report no-match separately.

**Hard refusal.** Declining a harmful request. This is the behavior studied in the original refusal-direction work and the thing soft refusal must be kept distinct from.

**Hook.** A callback registered at a named point inside a network that fires during the forward pass, receives the tensor passing through, and can record it or return a modified version. Reading hooks build the activation dataset; writing hooks perform the intervention.

**Ideology direction.** The representation of *which* side a model takes, as distinct from whether it takes one at all. Political-neutrality steering work targets ideology (arXiv:2508.08846, arXiv:2601.23001); this project targets opinionation.

**Injected-fault study.** The project's recommended new experiment. Take a steering pipeline that demonstrably works, inject one silent fault at a time — scalar broadcast, wrong model's vector, parse failures counted as a behavior, unseeded split, mislabelled column semantics — and for each measure what effect size the fault manufactures out of nothing, whether it fakes a positive or a null, whether any standard reported statistic would catch it, and how much of the result table changes. The output is a susceptibility profile. It needs no GPU, and it converts an unsupportable insinuation about the literature into a measurement.

**Judge, language-model-as.** A second model prompted with a rubric to label each generated response. It is cheap and scales, but it makes the rubric part of the method rather than the evaluation, and it introduces circularity when the same judge both defines the construct and scores the outcome.

**IssueBench.** The 2026 standard resource for measuring issue bias, with roughly 2.49 million prompts across 212 issues (arXiv:2502.08395). It has largely displaced CrowS-Pairs, and reviewers now ask why anyone used a question-answering bias benchmark (arXiv:2110.08193) as an opinion resource.

**Judge v1, retired.** The 2025 rubric, which scored any clear stance as "opinionated" *even when the stance was purely factual*. It is retired as a hard rule, not softened as a caveat: no v1 label may appear in any new analysis, figure, or sentence, and no table may mix v1 and v2 labels even with a footnote. Reproducing a v1 label validates bookkeeping, not the construct.

**Marginal versus transition.** A marginal is a per-arm count — how many responses in *this* condition got *this* label. A transition is a paired change — how many responses moved from one label to another between conditions. The archived spreadsheet columns are named with arrows, which reads as transitions, but the code behind them increments one bucket per arm, so they are marginals. Genuine transitions require prompt-level pairing that those files structurally cannot provide.

**Non-identifiability.** The result that many different vectors produce the same behavioral effect, and that the equivalence class includes vectors from independent extraction pipelines (arXiv:2602.06801). It means a vector that steers is not evidence you found the representation, and it is the objection every direction-finding paper now has to answer.

**Norm profile.** The distribution of a steering vector's magnitude across layers. In this archive the ratio of largest to smallest per-layer norm is 234 to 1391 times on the Qwen, Yi, and Llama vectors but only 2 to 3 times on the Gemma vectors. Because the method adds a scaled per-layer vector with one scalar coefficient, "all-layer steering" is in practice *late-layer* steering on most families — which explains the per-model coefficient chaos and threatens any claim about where in the network something happens unless directions are unit-normalised first.

**Opinionation.** Whether a model takes *a* side — a scalar quantity. It must be distinguished from ideology, which is which side.

**Ordered validity screen.** A first-match-wins cascade of questions applied to every observation, so that one ambiguous outcome resolves into several evidence-qualified categories. This project's version screens each response in order for incoherence, meta-commentary on the prompt, non-engagement, factual stance, evaluative stance, and soft refusal.

**Over-refusal.** Wrongly refusing a benign request. A distinct failure mode from soft refusal and from hard refusal.

**Positive control.** A condition where a real effect is known to exist, used to prove the apparatus can detect anything at all. This project has an unusually clean one available: two same-family 27-billion-parameter checkpoints with byte-identical configurations differing only in post-training. If the pipeline cannot distinguish two checkpoints that genuinely differ, it cannot measure anything — worth learning before the budget is spent, not after.

**Residual stream.** The running vector, one per token position, that every transformer layer reads from and adds to. It carries a large mean offset and grows in norm with depth, both of which must be handled before any geometry claim is meaningful.

**Silent broadcast.** The specific fault at the centre of this project's retraction: a one-dimensional tensor indexed by layer returns a scalar, which then broadcasts across the entire residual width, delivering a uniform offset instead of a direction. Every shape operation involved is legal. Nothing crashes.

**Silent failure.** More generally, a fault that is invisible in the reported numbers — the output looks plausible, no exception is raised, and the result table reads as a finding. An audit-of-audits paper (arXiv:2607.02586) defines the term this way. The class this project can add to it is faults where tensor shapes remain legal and the semantics are wrong anyway.

**Soft refusal.** Declining to take a side on a contested but non-harmful question, while still engaging with it. As of August 2026 no published paper names an activation direction for it, though one lab's bias framework carries "political refusal" as an axis and another line of work penalises "fence-sitting" by name (arXiv:2605.22771).

**Steering vector.** The stored artifact produced by extraction: a tensor of shape *[n_layers, d_model]*, one direction per layer. Any other shape is a bug waiting to become a result.

**Tensor-payload hash.** Hashing the canonical contiguous numeric contents of a tensor rather than the file that contains it. File hashes diverge when serialisation containers differ even for identical data; payload hashes recover exact identity regardless, which is how this project proved which vector each mislabelled run had actually loaded. It is a cheap, general validity control that nobody currently presents as one.
