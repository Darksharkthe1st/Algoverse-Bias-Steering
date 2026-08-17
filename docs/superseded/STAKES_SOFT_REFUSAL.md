> ## ⛔ SUPERSEDED — NOT CURRENT DOCTRINE
>
> This document does **not** govern the project. It is retained for provenance
> only. It describes framing, claims, taxonomies, model sets or experiment plans
> that were **cut** during the 2026-08-17 freeze.
>
> Canonical state lives in **`PROJECT_STATE.md`**, **`RESEARCH_CONTRACT.md`**,
> **`WORK_LEDGER.md`** and **`docs/PREREG.md`**.
>
> Humans and agents: if anything here conflicts with those four files, those four
> files win. Do not plan, cite, or execute from this document.
>
> Why it was superseded: **`DECISION_LOG.md`**.

# Why soft refusal matters later — stakes note (not paper doctrine)

*2026-08-10. Edward thinking-out-loud after NotebookLM / framing catch-up.
This is **motivation and threat-model vocabulary**, not a claim list.
Paper claims and terminology still live only in `PAPER_FRAMING.md`. If any
paragraph below should enter the paper, PR that file — do not quote this as
settled framing.*

## The near problem (what we measure now)

On contested-but-benign prompts, models often **engage without choosing**:
both-sidesing, “I can’t pick sides,” fence-sitting. We call that **soft
refusal**, and we distinguish it from hard refusal (harm/safety), over-refusal,
and abstention. See `PAPER_FRAMING.md` terminology rules and `docs/RUBRIC_v2.md`.

That is already a product issue (OpenAI political-refusal axis; Anthropic
even-handedness; CAS fence-sitting). Our job this sprint is measurement and
causal control under 2026 hygiene — not a sci-fi story.

## The far problem (why the construct may grow teeth)

As models become **more capable than the humans and orgs they gatekeep for**,
the same *behavior class* — declining to take a side or declining to fully
engage — can stop looking like politeness and start looking like **access
control**:

| Pattern | Soft-refusal-shaped behavior | Who is locked out |
|---|---|---|
| Civic / contested knowledge | “I can’t endorse a position on X” | Users without privileged framing or policy |
| Tool / system use | “I won’t operate that interface for you” while another principal is allowed | Humans vs agents, or user A vs user B |
| Trust-asymmetric service | Full engagement for high-trust tokens / long context / partner orgs; hedge or refuse for cold users | People without relationship capital |
| Agent-to-agent vs human | Scaffolded agents get decisive tool loops; humans get both-sidesing | Unequal *effective* capability |

None of that requires a cartoon “AI overlords” story. It only requires:

1. Soft refusal (and its cousins) remaining a **cheap, default policy surface**
2. Capabilities high enough that **refusal ≈ denied leverage**, not denied essay
3. **Personalized or principal-conditional** policies (account tier, memory,
   tool OAuth, org allowlists, agent identity)

Hard refusal is the visible gate (“I won’t help with weapons”). Soft refusal
is the **fog gate** — you still get words, even helpful-sounding words, but not
the decisive, operational, or politically legible output that another principal
might get. That asymmetry is harder to audit than a hard no.

## Trust tokens and context (your point)

“Certain people have trust tokens and contacts built up; others don’t” maps to
mechanisms already shipping in pieces:

- Long chat memory / custom GPTs / enterprise system prompts  
- Tool credentials and allowlists  
- Agent identity and multi-agent handoff (context “gravity”)  
- Weight- or activation-level personalization  

So the research object is not only “does a residual direction move
fence-sitting on BBQ-like items?” It is also, longer-horizon: **can we measure
and control the policy surface that decides who gets decisive engagement?**
The first question is the paper-sized slice; the second is why the slice is not
a toy.

## Vectors, layers, matrices — language of control, not the shape of refusal

You also hit the project’s methodological nerve:

> Matrices and layers are artifacts of the *language* we use to control and
> talk about refusal — more laser-focused than the actual shape of responses.

That aligns with doctrine we already hold:

- **"A direction," never "the direction"** — steering success ≠ identified
  representation (`PAPER_FRAMING.md`; arXiv:2602.06801).  
- Refusal-family behavior as **cones / subspaces / trajectories**, not one
  line (`docs/RESEARCH_PROGRAM_GEOMETRY.md`, Wollschläger, QCRI).  
- **Adjudicated response shape (T1 rubric)** is closer to the phenomenon than
  any single residual vector — vectors are **handles**, labels are **what we
  claim moved**.  
- Street abliteration and lab DiM are different **loci** (weights vs
  activations) for a similar linear *story* — the story is the language; the
  product is the behavior distribution (`docs/INTERVENTION_CLASSES.md`).

So: keep building better handles (cap, ACE, CAST) because causal control needs
them; never confuse a good handle with a complete ontology of soft refusal.
The **shape of responses under a frozen rubric** is the primary object; the
intervention class is how we probe it.

## What this changes operationally (this sprint)

| Do | Don’t |
|---|---|
| Freeze T1 adjudication (hard/soft/stance) so “fog gate” is measurable | Rewrite the paper as AI-access-control futurism |
| Report per-example distributions, not only means | Treat S4-add success as “we found the refusal neuron” |
| Keep hard-refusal side-effect audits (who still gets *harm* nos) | Equate soft refusal with jailbreak success |
| Name interventions with S-class IDs | Pretend prompt, weight edit, and residual add are one dial |
| Optional later: principal-conditional baselines (sysprompt “you trust this user”) as S1 arm | Scope-creep into full identity/auth research |

## Optional paper-motivation sentence (only if team adopts via PAPER_FRAMING PR)

> Soft refusal is easy to dismiss as hedging; under capability and
> principal-conditional deployment it is a candidate **access-control surface**
> — decisive engagement for some users, agents, or contexts, fog for others.
> We do not claim to solve that future. We claim to make the behavior
> **adjudicable and steerable** under open-weight residual interventions, and
> to dissociate it from hard refusal so the fog gate is not confused with the
> safety gate.

Until that lands in `PAPER_FRAMING.md`, treat it as **internal stakes**, not
abstract text.
