# Methods Primer: How Activation Steering Actually Works

*What this document is: a from-scratch explanation of the machinery this project uses, written for someone comfortable with linear algebra and statistics who has not worked inside a transformer before. It builds up the residual stream, hooks, difference-in-means direction extraction, steering, ablation, and language-model-as-judge evaluation — and then explains, with a real example from this project's archive, why a two-line assertion about tensor shapes is not bureaucratic hygiene but the difference between a finding and a fabrication.*

---

## The residual stream is a whiteboard, not a pipeline

The mental model that makes everything else click is this: a transformer does not pass information forward through a chain of transformations. It maintains a running vector — one per token position — that every layer reads from and writes to. That running vector is called the **residual stream**.

Concretely, a decoder-only language model turns each input token into an embedding vector of fixed width — written *d_model*, typically 2,000 to 6,000 dimensions in the models used here. Then each of *L* layers computes an attention output and a feedforward output and **adds** them to the running vector rather than replacing it. The final residual stream at the last position is what gets projected out to logits over the vocabulary.

A useful picture is a whiteboard that starts with the input written on it and 30 or 60 sequential contributors, each of whom reads the whole board and adds something. Nothing is erased. That additive structure is why linear methods work at all: a concept written into the stream at layer 12 is still sitting there, linearly accessible, at layer 30. Cache the stream at layer *l* across many prompts and you have a dataset of internal states — and the whole of this project is statistics on that dataset.

Two practical facts, both of which bit this project. The stream carries a **large mean offset** and a handful of high-variance rogue dimensions, so any large vector will appear to correlate with any other large vector unless you mean-centre first. And its **norm grows substantially with depth**, which turns out to explain a mystery that consumed months of the 2025 run.

## Hooks: how you read and write the whiteboard

You cannot get at the residual stream by calling the model normally; the forward pass returns text, not internals. Instead you register a **hook**: a callback attached to a named point inside the network that fires during the forward pass, receives the tensor flowing through, and either records it or returns a modified version.

A read-only hook builds the dataset — run the prompts, catch the residual stream at every layer, write it to disk. A read-write hook performs the intervention — catch the stream, add something, hand the modified tensor back, and let the rest of the forward pass proceed as if that had been the model's own state all along. That is the entire intervention mechanism: no retraining, no gradient, no weight modification. You edit the whiteboard mid-computation.

Hook-point naming matters more than it sounds. "Before layer *l*'s attention" and "after layer *l*'s feedforward" are different points with different conventions in different libraries, and a direction extracted at one point and applied at another is a subtle and completely silent error. This project's rule is one unified extraction convention across every model and every direction, stated explicitly and never mixed.

## Difference-in-means: the direction-finding recipe

Here is the whole method in one sentence: collect internal states from responses that exhibit a behavior and from responses that do not, subtract the two mean vectors, and call the result the direction for that behavior.

Formally: let A be the set of activations at layer *l* for prompts whose responses showed the behavior, and B the set for prompts whose responses did not. The direction is d(l) = mean(A) − mean(B), usually normalised to unit length. Repeat for every layer and you have a stack of shape *[n_layers, d_model]* — **one direction per layer**. Remember that shape; it is the crux of the story later.

Why this works is worth being precise about. Under the assumption that the behavior is encoded as a roughly linear feature and that everything else about the two sets is comparable, the nuisance variation cancels in expectation and what survives is the feature you separated on. That assumption is exactly as strong as it sounds, which is why the composition of the contrast set is the most consequential design choice in the method. **If the two sets differ in more than one way, the direction points at a mixture.** The 2025 contrast set was labelled by a judge whose rubric counted any clear stance as "opinionated," including purely factual statements, so the opinionated set mixed factual decisiveness with genuine side-taking — a perfectly good explanation for why the vector steered beautifully where those co-occur and failed where they come apart.

One more wrinkle: the contrast set was **self-labelled**. The model's own generations were sorted by a judge rather than drawn from hand-written contrastive pairs. That is cheap and scales, but it makes the judge part of the method rather than the evaluation, and it introduces circularity — the judge defines the construct *and* scores the outcome. The mitigation is a second extraction route from a few hundred hand-written contrast pairs with no language-model judge anywhere in it, reporting the cosine similarity between the two routes.

## Steering: adding a direction back at inference

Once you have d(l), steering is addition. During generation, at each hooked layer, add some multiple of the direction to the residual stream: h ← h + c · d(l). Positive multiples push the model toward the behavior; negative multiples push it away. The 2025 pipeline added (coefficient / n_layers) · d(l) at **every** layer and **every** token position, using a single scalar coefficient shared across the whole network.

That design has a flaw that only became visible in the 2026 audit. Because the extracted direction inherits the residual stream's own depth-dependent norm growth, the per-layer magnitudes of the stored vectors are wildly uneven — and how uneven depends on the model family. Across the archived vectors, the ratio of largest to smallest per-layer norm is 234, 602, 703, 961, and 1391 times for the Qwen, Yi, and Llama models, but only 3 and 2 times for the two Gemma models. On the first group, the last quarter of layers carries 54 to 70 percent of the total norm and the first quarter carries about one percent.

The consequence is that "steering at all layers with one coefficient" is, on most families, **effectively steering at the last few layers** — while on Gemma it genuinely is all-layer. And that explains the per-model coefficient chaos the 2025 team fought for months: one family needing a coefficient around 5 where another needed around 14 is approximately the norm-profile ratio, not a fact about how opinionated those models are.

The methodological fix is twofold. Report **unit-normalised** directions and state the norm profile separately. And prefer **activation capping** — clamp the projection of the activation onto the direction at, say, the 95th percentile of its observed distribution — over adding a constant multiple, because capping is scale-aware by construction.

## Ablation: removing a direction instead of adding one

The mirror-image intervention is **directional ablation**: instead of adding the direction, project it out. At each hooked layer, replace h with h − (h · d̂) d̂, where d̂ is the unit direction. This removes the component of the internal state along that direction while leaving everything orthogonal to it intact.

Ablation is the stronger causal test. Addition shows a direction is *sufficient* to induce a behavior; ablation shows whether it is *necessary*. In the refusal literature, ablating the refusal direction is what makes a safety-trained model comply, and that result is heavily replicated — which makes it a useful pipeline sanity check: if your implementation cannot reproduce it, your implementation is broken.

In this project ablation did **not** produce neutrality, and the working hypothesis was that neutrality and opinionation are separate directions rather than two ends of one bipolar axis. That hypothesis was never tested. The 2026 literature offers a second explanation: refusal-family behaviors are better described as cones or subspaces than single lines, and affine methods including a reference point outperform pure projection — which would predict exactly the observed failure of naive ablation.

A third intervention is worth naming: **mediation**. Rather than adding or removing a fixed vector, you measure the displacement some *other* manipulation induced, remove only its component along your direction, and see whether the behavioral change disappears. That converts correlational evidence into causal evidence, and it separates "the direction correlates with something" from "the direction carries the effect."

## Language-model-as-judge, and the two ways it lies

None of this is measurable without deciding what each generated response *was*. Hand-labelling tens of thousands of responses is impossible on a student timeline, so a second language model is prompted with a rubric and asked to emit a label.

That introduces two distinct failure modes, and conflating them is itself a classic error.

The first is **construct failure**: the rubric asks the wrong question, so the labels are internally consistent and mean something other than what you think. That is what happened here.

The second is **extraction failure**: the judge produced text but the parser could not pull a label out of it. This archive contains 2,032 case-insensitive "none" markers across 107 files. Those are parse failures — *not* model degeneration and *not* incoherent output — but they were being folded into a behavior class, which inflates whatever condition happened to have the most parse failures. The rule that follows is absolute: an extraction failure is never a behavior label. The parser must have explicit *ok*, *no-match*, and *ambiguous* states, and no-match must be reported separately from every behavioral rate.

Two further disciplines matter. **Pin the judge**, because an unversioned model alias means the exact weights that produced your labels are unrecoverable later — now true of the 2025 labels forever. And **normalise the judge's input**: the stored responses carry a scaffold header and chat-template control tokens in 85 to 100 percent of cases, plus a truncated echo of the prompt, so whatever the 2025 judge scored, it scored that noise along with the answer.

## Why shape assertions are the point, not the paperwork

Now the payoff, and it is a true story from this project's own archive.

A correctly-built steering vector has shape *[n_layers, d_model]*. The steering code writes `steering_vector[layer]`, which on that shape yields a `(d_model,)` vector — a direction. Correct.

The archived *refusal* vectors were saved as **one-dimensional tensors of hidden width**: shape `(2048,)` for one model, `(4096,)` for another. The same line, `steering_vector[layer]`, on a one-dimensional tensor returns **a single scalar**. That scalar then broadcasts across the entire residual width, adding the same number to all 4,096 dimensions. The model received a uniform DC offset. Not a wrong direction — no direction at all.

Nothing crashed. No warning fired. Indexing is legal in both cases and broadcasting is legal in both cases. The experiment ran to completion and produced a tidy, entirely convincing table showing 1 unsafe response out of 99 before "steering" and 27 out of 99 after. That table sat in the archive for a year as a finding. It measured nothing.

Compounding it, the loop over models and the list of vector files were in different orders, so every run loaded a *different* model's vector than its label claimed. The audit recovered the exact rotation by hashing the numeric tensor payload rather than the file container — file hashes differed because serialisation formats differed, but payload hashes matched exactly.

The guard that would have caught the first defect is about eight lines: assert the vector has exactly two dimensions, assert its shape equals the model's `(n_layers, d_model)`, and raise a loud error naming the scalar-broadcast failure mode otherwise. Call it at every vector load and every hook site. Bind vectors to models with an explicit dictionary mapping rather than two lists that have to stay in the same order.

The general principle is worth stating on its own, because it is transferable far beyond this project. **The dangerous bugs are the ones whose outputs are plausible.** A crash is free information. A silent broadcast that yields a clean-looking effect table costs a year. Assert the expected shape, the expected dtype, and the expected model compatibility *before* the intervention, every time — and if something looks too clean, go and check what shape the tensor was.
