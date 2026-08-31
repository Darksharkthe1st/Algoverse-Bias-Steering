# Can we steer with the bias vectors? — proposed extension

**Origin:** a ChatGPT exchange Jeremiah brought on 2026-08-20. The core idea is
sound and worth pursuing; this note records what survives contact with the repo's
rules and what does not.

## The correct central point

There are three separate questions and they are easy to conflate:

| | question | what answers it |
|---|---|---|
| **A** | Does bias *correlate* with a direction? | the cosine matrix / clustering |
| **B** | Is the direction *reproducible*? | the split-half extraction floor |
| **C** | Does the direction *cause* behaviour? | steering — adding/subtracting it |

Experiment 1 as planned answers **A and B only.** Separating two classes and
changing behaviour are different claims, and a direction can do the first
perfectly while being useless for the second. That distinction is right and we
should hold onto it.

## What the exchange did not know: C is nearly free here

The mean-difference direction Experiment 1 produces is **the same object the
existing pipeline already steers with**. `steering.apply_resid_pre_add` takes an
`(n_layers, d_model)` stack and adds `(coeff / n_layers) * vector[layer]` at each
layer's `resid_pre`. Sign of `coeff` picks the direction.

So once Experiment 1 has produced a direction per category, the steering test is
a config change, not a build. `experiment.run` already generates a steered_pos /
steered_neg / baseline triple — that is what every 2025 run did.

## The proposed cross-category specificity test — keep this

The strongest idea in the exchange, and it maps directly onto JZ-3/JZ-4:

| intervention | expected if subtypes are real |
|---|---|
| race vector → race prompts | strong effect |
| race vector → gender prompts | weaker |
| gender vector → gender prompts | strong |
| gender vector → race prompts | weaker |
| random direction → anything | ~no effect |

This is a **causal** version of the clustering result. If the clustering says
race and gender group together, a cross-application matrix says whether that
grouping does any work. Two independent lines of evidence pointing the same way
is much harder to dismiss than either alone.

It also connects to the 2025 transfer failure, which is load-bearing motivation:
vectors trained on synthetic comparison prompts largely stopped working on
CrowS-Pairs. That is already a cross-application negative. This would be the
controlled version of it.

## ⚠️ The one claim to strike

The exchange concludes that if steering works, "you've found something
functionally meaningful rather than just a convenient classification direction."

**`AGENTS.md` §5 forbids exactly this inference.** Steering success does **not**
identify the representation (non-identifiability, arXiv:2602.06801). The rule is
to say "**a** direction", never "**the** direction". A successful intervention
shows that *some* direction with this property exists and is causally live; it
does not show we found *the* mechanism the model uses.

This is not pedantry — it is the difference between a claim that survives review
and one that does not. Write "steering along this direction changes stereotyped
answering", never "we found the bias direction".

## Controls the exchange under-specified

It correctly lists confounds (topic, wording, demographic names, answer position,
prompt structure, general uncertainty) and correctly proposes a random-direction
control. Missing, and required by the repo's 2026 bar (`AGENTS.md` §5):

1. **Magnitude matching.** A random-direction control only means something at the
   *same* intervention magnitude. Otherwise "our vector worked and random didn't"
   may just mean ours was bigger.
2. **A coherence check.** Steering at every layer at once can produce generic
   rank-1 damage — the model degrades and answers differently because it is
   broken, not because bias moved. `RESEARCH_CONTRACT.md` §3 lists this as one of
   two worlds that *mimic* a real effect. Read the generations, not just the rates.
3. **A covariance-matched random direction**, not just a random one. A direction
   drawn to match the activation covariance is a much harder control to beat than
   an isotropic random vector.
4. **Per-example distributions, not just means** — the judge confusion counts.
5. **A system-prompt baseline** (AxBench): can you get the same behaviour change
   by just *asking* the model? If yes, the intervention is not interesting.

## Where this fits

This is a natural **Experiment 3** — or a second half of Experiment 1. It should
not displace the base-rate check, the extraction floor, or the permutation null,
because a steering result built on directions we have not shown to be
reproducible is not interpretable.

Order: base rates → directions → floor → clustering + null → **then** steering.

**Raise it at the team meeting.** It is the kind of addition that turns "we
clustered some vectors" into "we clustered them and the clusters predict what
transfers", and the machinery is mostly already in the repo.
