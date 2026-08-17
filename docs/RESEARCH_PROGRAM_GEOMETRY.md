# Research program: mapping the geometry of refusal

*Parked deliberately, 2026-08-07. This is **not** the Aug 29 paper. It is the
direction the current work is a foundation for, written down now so it doesn't
get lost in a sprint.*

## The interest

The long-horizon question is not "does a soft-refusal direction exist." It is
**what shape does refusal actually have in latent space, and can we map it.**
Single directions, cones, subspaces, manifolds — the field has been moving along
that ladder for two years, and the honest position in 2026 is that nobody has a
map. They have samples.

## Why the current sprint is a prerequisite, not a detour

This is the part worth being precise about, because it looks like a consolation
prize and isn't.

Every geometric claim is a claim about **distances and angles between measured
directions**. Our own archive just demonstrated four separate ways those
measurements go silently wrong:

1. **Norm profiles swamp geometry.** Per-layer vector norms span 2–3× on gemma
   but 600–1391× on Qwen/Yi/Llama. Any cross-model depth or angle comparison
   that doesn't unit-normalize first mostly recovers the residual-norm profile.
   A "geometry result" computed on unnormalized directions is an architecture
   plot wearing a costume.
2. **There is no variance floor.** We have no estimate of how much a direction
   moves when the contrast set is resampled — the two archived Qwen1.5-7B
   vectors are byte-identical copies. **A cosine of 0.35 between two directions
   is uninterpretable** until you know whether re-extracting the *same*
   direction gives 0.97 or 0.60. Every published cosine without that floor is a
   number without units.
3. **Shape bugs are silent.** A 1-D tensor indexed as if it had layers yields a
   scalar broadcast across the residual width. It produces a clean result table.
   Geometry computed on a mis-shaped artifact is geometry of nothing.
4. **The construct may not be what the label says.** If the judge conflates
   decisiveness with bias, the "direction" you map is a mixture, and its
   geometry is the geometry of your rubric's ambiguity.

So the sequence isn't sprint-then-geometry. It's: **you cannot map a space you
cannot measure reliably, and the sprint is what makes the measurements mean
something.** The extraction-variance floor in particular is a hard gate — it is
listed as a workstream now precisely because the geometry program dies without
it.

## What the map would actually be

Sketch, not a plan. Each rung needs the one below it.

**Rung 1 — a metrology layer.** Unit-normalized directions, an extraction
variance floor from contrast-set bootstrap, per-layer norm profiles reported
separately from angles, and shape assertions everywhere. Deliverable: any two
directions can be compared with a defensible error bar. *Most of this is in the
current sprint already.*

**Rung 2 — structure at one site.** For a single refusal-flavored behavior at a
single model and layer: is it a direction, a cone, or a subspace? Principal
angles between independently-extracted directions, and the fraction of causal
effect retained as you project onto successively smaller subspaces. The
literature has cones (arXiv:2502.17420) and subspaces (arXiv:2607.02396) but
little on *when* each description is warranted.

**Rung 3 — the atlas.** Many behaviors (harm refusal, soft refusal, over-
refusal, abstention, evasion), many models, one coordinate system. Fractional
depth for the vertical axis; the open question is what the horizontal one is.
Do the flavors occupy a shared subspace with different bases — which is what the
one-shared-knob finding (arXiv:2602.02132) would predict — or genuinely
separate regions?

**Rung 4 — dynamics.** Refusal as a trajectory rather than a point: how the
representation moves across layers and tokens, and where the decision actually
commits. There is early work reading refusal as a trajectory; almost nothing on
the geometry of that path.

**Rung 5 — transport.** Does the map transfer across models? Universal-geometry
results (vec2vec and successors) suggest embedding spaces are alignable; whether
*behavioral* directions align under the same transport is open, and it would be
the difference between an atlas of one model and an atlas of the family.

## Two threads from this project that feed it directly

**Input perturbations as probes.** The perturbation arm from
`docs/2026-08-02_sprint_proposal.md` — currently parked — is a way to move a
model along a direction *without designing the intervention*. As a mapping tool
that's more interesting than as a validity check: perturbations are a source of
naturally-occurring displacement vectors, and the distribution of where they
land is itself a measurement of the space. The dual-use rules in the proposal
(§9) carry over unchanged.

**The post-training pair.** Qwen3.5-27B and Qwen3.6-27B have byte-identical
architectures and differ only in post-training. That is a controlled probe of
how alignment training *moves* refusal geometry — arguably the single cleanest
such probe currently available in open weights, and it exists for free.

## Honest assessment

The 2026 frontier moved toward geometry: feature manifolds, concept cones,
manifold-aware steering. The field agrees the picture is geometric. What it
lacks is metrology — shared conventions for normalization, variance, and shape
discipline that make one group's cosine comparable to another's.

**That gap is a real contribution and it is more tractable than the atlas.** It
is also, not coincidentally, exactly what this sprint is building. The strongest
version of this program probably starts by publishing the measurement layer and
letting the map follow, rather than the other way round.

## Status

Parked. Not in the Aug 29 scope, and adding it there would repeat the 2025
failure mode. Revisit after the sprint, when we know whether the construct
survived Gate 1 and what the extraction-variance floor actually is — both of
which are answers this program needs and doesn't have yet.
