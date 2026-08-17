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

# Steering toolkit — what we have, what the field has, what we adopt

*Written 2026-08-10 from the Aug 8 team meeting (Edward owns new techniques)
plus the frontier scan already in `docs/2026-08-01_project_analysis.md` Part 2.
This file owns **intervention recipes and adoption priority** only — not
framing, model set, judge, or claim status. Link those elsewhere.*

## One-line status

We implement **one recipe**: difference-in-means (DiM) extraction +
**constant additive residual steering at every layer**. That was state of the
art in mid-2024. In 2026 it is the *baseline intervention*, not the method.
The paper still needs it (comparability to 2025 + Arditi lineage); the
toolkit must grow so ablation failures, coefficient chaos, and OOD brittleness
have modern answers.

**Class names** for interventions (S1 prompt vs S4 residual vs S5 weight
abliteration, etc.) live in **`docs/INTERVENTION_CLASSES.md`** — use those IDs
in configs and writeups so “steering” is not one overloaded word.

## What ships today (2025 archive → current `src/` / notebooks)

| Piece | Implementation | Where |
|---|---|---|
| Extraction | DiM of mean residual streams, last-token-ish batch mean over judged neutral vs opinion generations | `get_opinion_vec_from_resids` in `experiments/farhan-experimentation.ipynb`; `calculate_steering_vector` in `src/main.py` |
| Intervention | `value[:, :, :] += (coeff / n_layers) * steer_vec` on `blocks.{L}.hook_resid_pre` for **all** layers | `batched_generation` / `steer_model` |
| Bidirectional control | Sign flip of the same vector (`flip_steering`) | same |
| Ablation | Attempted; failed to produce neutrality (load-bearing negative — keep honest) | methodology logs |
| Synthetic residual forcing | Tried Dec 2025 on `farhan-synthetic-steering`; unsuccessful | branch history |
| Shape hygiene | **Required going forward** — see Farhan runbook Task 1; 1-D refusal tensors caused silent DC-offset "steering" | `docs/REVIVAL_AUDIT.md` |

**Not present:** system-prompt baseline, activation capping, ACE, CAST, single-
layer / fractional-depth dose matching, subspace/cone extraction, manifold
steering, SAE feature steering, nnsight / TransformerBridge path for models
outside TransformerLens reimplementations.

## Field map → project use (priority tiers)

Tiers are about **what unblocks our paper and Farhan's grid**, not academic
completeness. Tier-0 is mandatory hygiene; Tier-1 is "implement this week if
we want modern interventions in the main grid"; Tier-2 is geometry / stretch.

### Tier 0 — hygiene (no new "method," just stop being 2024)

| Technique | Why for us | Effort | Owner signal |
|---|---|---|---|
| **System-prompt baseline** (AxBench bar) | Without it, every steering number is unreportable under 2026 norms | Low — same prompts, no hooks | Edward + Farhan runner |
| **Shape assert `(n_layers, d_model)`** at load and hook | Prevents the refusal-arm silent bug class | Trivial | Farhan Task 1 |
| **Unit-normalize directions; report norms separately** | Norm profiles swamp angles (see verification) | Low | Edward geometry support |
| **Per-example 3×3 distributions, not only means** | Reliability literature + our own judge risk | Medium (judge v2) | Gate 1 |
| **Capability + safety side-effect audits** (verbal reasoning + XSTest / JailbreakBench — **not** GSM8K as the only skill check) | Meeting: do early so we never claim clean control we didn't measure | Medium harness | Edward harness / Farhan grid |
| **Dose-match on frozen dev split** before off-target reads | Sprint proposal + prior art | Medium | Farhan |

### Tier 1 — recipe upgrades that fix *our* known failures

| Technique | Paper / source | Maps to which 2025 failure | Adopt? |
|---|---|---|---|
| **Activation capping** (clamp projection onto direction at a percentile, e.g. 95th — Anthropic Assistant Axis style) | arXiv:2601.10387; also persona-vector ops | Coefficient chaos; all-layer constant add overshoots | **Yes — primary modern intervention** |
| **ACE — affine concept editing** (project out + add with a reference point; refusal ≈ affine) | Marshall et al. arXiv:2411.09003 | Ablation failure: pure subtract has meaning on both poles of opinionation | **Yes — ablation replacement** |
| **CAST — conditional activation steering** (steer only when a condition vector fires) | Lee et al. ICLR 2025, arXiv:2409.05907 | Steer-everywhere damages capability; soft-refusal should fire on contested prompts only | **Yes — second modern intervention** |
| **Single-layer / fractional-depth steering** instead of all-layers with `coeff/n_layers` | Depth-migration arXiv:2606.29196; our own layer tests exist but were under-analyzed | Coefficient mythology; enables C5-style profiles | **Yes — for efficacy profiles** |
| **TransformerBridge / HF-wrap path** (TL pain of adding models → wrap HF) | TransformerLens 3.x direction discussed in meeting | Model-set bottleneck for Qwen3.5/3.6, gemma-4 | **Yes — infra, not a result** |
| **MoE force-all-experts when extracting DiM** | Expert-Aware Refusal Steering arXiv:2606.04160 | gemma-4-26B-A4B arm would otherwise get noisy directions | When MoE row runs |

### Tier 2 — geometry / optional (park unless Gate 1 is green and grid is boring)

| Technique | Source | Note |
|---|---|---|
| Concept cones / principal angles | Wollschläger arXiv:2502.17420 | Soft vs hard refusal geometry; needs extraction variance floor first |
| Manifold steering | Wurgaft arXiv:2605.05115 | High value long-term; high implementation cost — see `docs/RESEARCH_PROGRAM_GEOMETRY.md` |
| MFA regions / "directions → regions" | arXiv:2602.02464 | Strongest SAE-alternative for steering in the scan |
| Persona-vector automation (trait → contrast prompts → DiM) | Anthropic arXiv:2507.21509 | Process upgrade for extraction, not a different causal story |
| HyperSteer / ReFT-r1 | arXiv:2506.03292 | Learned interventions; out of scope for 3-week sprint unless free code drops in |
| SAE feature steering (Qwen-Scope) | arXiv:2605.11887 + model-set note | Cross-check DiM only; license check required |

### Explicitly *not* our primary lane this sprint

- **Jailbreak / abliteration productization** — we study soft refusal and
  measurement validity; we do not ship uncensored models.
- **Ideology steering** (which side) — Nadeem / CLAS own that; we steer
  **opinionation** (whether a side). See `PAPER_FRAMING.md`.
- **Claiming "the" direction** — non-identifiability arXiv:2602.06801;
  always "a direction."

## Recommended implementation order (Edward, next few days)

```
1. Inventory lock (this file) + Discord/community harvest → method shortlist
2. Spec three intervention backends behind one interface:
     - add_constant   (legacy, for 2025 comparability)
     - cap_projection (Tier 1 default for new runs)
     - ace_affine     (Tier 1 ablation replacement)
   Optional fourth: cast_gated (condition on "contested-prompt" vector)
3. System-prompt baseline arm in the same harness config
4. Smoke on one small open model (existing TL-supported) before Qwen3.5 ladder
5. Hand off to Farhan's runner as pluggable intervention enum
```

**Do not** block Gate 1 on this. New techniques expand the grid; they do not
make un-judged percentages publishable. Judge v2 / gold-set remains the
measurement gate (`docs/RUBRIC_v2.md`, `docs/PREREG.md`).

## Interface sketch (for Farhan's refactor)

```python
# Conceptual — land in the production runner, not as a one-off notebook cell.
@dataclass
class Intervention:
    name: str                    # "add_constant" | "cap_projection" | "ace_affine" | "cast_gated" | "system_prompt"
    direction: Tensor            # assert shape (n_layers, d_model) OR (d_model,) with layer index
    layers: list[int] | str      # "all" | fractional schedule
    strength: float              # dose; for cap, the clamp scale; for ACE, the affine mix
    condition: Tensor | None     # CAST only
    reference: Tensor | None     # ACE only

def apply_intervention(resid, intervention, layer: int) -> Tensor:
    d = assert_direction_slice(intervention.direction, layer, ...)
    if intervention.name == "add_constant":
        return resid + intervention.strength * d
    if intervention.name == "cap_projection":
        # project resid onto d; clamp coefficient; reconstruct
        ...
    if intervention.name == "ace_affine":
        # resid - proj_d(resid) + strength * (target - ref)  [exact form from ACE paper]
        ...
    ...
```

**Hard rule (from the archive):** never index a 1-D tensor as if it had layers.
Assert before hook.

## Community / Discord harvest (Bazi · Jailbreaking)

Meeting + Edward follow-up: scrape **practitioner** steering methods that never
land in arXiv (persona vectors variants, abliteration recipes, conditional
hooks, tokenizer tricks). That is a **methods inventory**, not a results
source — nothing from Discord becomes a paper number without a controlled rerun
under our judge and model set.

### Tooling (what actually works)

**Primary: `discrawl`** (`~/bin/discrawl`, openclaw/discrawl) — already on this
machine. It mirrors Discord into local SQLite. For private community servers our
bot is **not** in, the working path is:

```bash
# Discord.app must have visited/scrolled the channel so LevelDB has history
set -a && source ~/.env && set +a
discrawl sync          # import desktop cache → SQLite
discrawl status
discrawl messages --channel 1228043845967544380 --limit 5
```

**Installed backup: DiscordChatExporter CLI** (`~/bin/discordchatexporter`,
v2.47.3 under `~/tools/discordchatexporter/`). API export needs a token that
can *see* the channel. The bot token in `~/.env` only sees Paper Planes + DMs
→ **403** on Bazi Jailbreaking. Do not commit user tokens.

**2026-08-10 harvest status** for channel `1228043845967544380`
(`🗝・jailbreaking` in **BASI**):

- **Preferred path:** `scripts/export_discord_channel.py` (API pagination,
  resume-safe JSONL). Smoke: ~2.3k msgs in 1.5 days — channel is extremely
  active; full backfill is multi-hour. Runner: `scripts/run_jailbreaking_export.sh`.
- **discrawl desktop cache** had 74k msgs but thin on recent months — incomplete.
- **UI scroll** (`scripts/scroll_discord_channel.applescript`) is fallback only.
- Keyword filter on the cache dump: ~488 hits; abliteration/CAA/RepE dominate.
  Details: `experiments/community_methods/README.md`
- Character / rumor / project-adjacent map:
  `experiments/community_methods/DOSSIER.md`. Public toolkit that is
  actually next to our stack: [elder-plinius/OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS)
  (Arditi + reversible steering-vector API). **Related work / comparison
  arm only** — not a recipe we adopt; not a results source.

### What to extract (coding pass after export)

Target phrases / clusters (case-insensitive scan of message content + embeds +
linked GitHub READMEs):

- activation / residual / steering / abliterat / "direction" / CAA / RepE
- CAST / conditional / gate / cap / clamp / project out
- nnsight / transformer.?lens / pyvene / EasyEdit / repeng
- refusal vector, opinion vector, persona vector
- layer range, coefficient, "add the vector", "subtract the vector"

Produce a **method card** per distinct recipe:

| Field | Content |
|---|---|
| Name / alias | as used in-channel |
| Mechanism class | add / project-out / cap / gate / weight-edit / prompt-only / other |
| Layer policy | all / single / range / learned |
| Conditioning | always / prompt-class / activation threshold |
| Code link | GitHub if any |
| Soft-refusal relevance | high / med / low / none (jailbreak-only) |
| Adopt candidate? | yes → which Tier | no + reason |

### Dual-use filter

Jailbreak channels mix measurement insight with attack recipes. For this repo:

- **Keep:** intervention mechanics that improve *control quality* or *side-effect
  profiles* on contested-but-benign prompts.
- **Drop from toolkit docs:** end-to-end jailbreak playbooks, payload lists,
  bio/chem/weapons content. Do not re-host those in `docs/` or `experiments/`.
- Paper posture stays measurement / soft-refusal (see `PAPER_FRAMING.md`) — we
  do not rebrand as a red-team methods paper.

### Artifact layout (when export lands)

```
experiments/community_methods/          # gitignored raw if huge / sensitive
  raw/                                  # Discord JSON — local only if ToS-sensitive
  method_cards.yaml                     # curated, commit-able
  links.md                              # public GitHub / papers only
```

Raw Discord dumps may contain usernames and PII — default **do not commit**.
Commit only curated method cards and public links.

## Acceptance for "toolkit expanded"

1. This file lists Tier 0–2 with adopt/defer decisions (done).
2. At least **two** non-legacy interventions implemented behind one interface
   and smoke-tested (`cap_projection` + `ace_affine` preferred).
3. System-prompt baseline runnable on the same prompt set.
4. Discord harvest either (a) method_cards.yaml with ≥10 distinct recipes
   tagged for soft-refusal relevance, or (b) a written note that export was
   blocked (auth / access) and the literature tiers alone stand.
5. Farhan can select intervention by config string in the production runner.

## Non-goals

- Replacing Gate 1 work (rubric freeze, gold-set κ, re-judge).
- Geometry atlas (`docs/RESEARCH_PROGRAM_GEOMETRY.md` stays parked).
- Restating model set, venue, or claims — those have other owners.
