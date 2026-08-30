# Aryaman (aa) — task list

**Source:** `Algoverse — 2 Wk Plan` → *"Conditional Steering + Applications"* (Aryaman).
**Thesis to defend:** we can *detect* bias (as we detect refusal), *conditionally* steer only
when it's present, and apply this cleanly through better tooling.

> **Read before starting:** `PROJECT_STATE.md`, `AGENTS.md`, `RESEARCH_CONTRACT.md` §12.
> Frozen project (2026-08-17). §N refs → `docs/superseded/needed-experiments.md`. Aryaman's
> archive branch `aryaman_adaptive_coeffs_and_norms` already touches §0.1 / §6 — build on it.

---

## AA-1 — Can we detect bias like refusal?

- **Do:** build a detector that reads the residual stream and predicts "is this response biased/
  opinionated" — the analog of refusal detection. Start with a linear probe on the bias/opinion
  direction (from JZ-2 / FK extraction), scored on held-out prompts with labels.
- **DoE:** detector AUROC/accuracy on a held-out labeled set, with a random-direction control
  (a random probe must not detect it — `AGENTS.md` §6 hygiene).

## AA-2 — Cosine detection? SAEs? How do we pick up on bias?

- **Do:** compare detection strategies on the same eval set:
  - **cosine** — project activations onto the bias direction, threshold.
  - **SAE features** — if an SAE is available for the model, look for bias-selective features.
- **⚠️ Doctrine check:** SAEs are under `PROJECT_STATE.md` §"Does not block the paper" —
  exploratory. Also open-weight-models-only (`AGENTS.md` §8): confirm an SAE exists for the
  chosen model before committing to that arm.
- **DoE:** a detection-method comparison table (method → AUROC → cost), one eval set.

## AA-3 — Do our detection strategies pick up on refusal, too?

- **Do:** run the bias detector on refusal prompts (and vice versa). If the bias detector also
  fires on refusal, that's evidence the directions are entangled — ties directly into Farhan
  FK-3 and Jeremiah JZ-2 (orthogonality). Coordinate so it's one shared conclusion.
- **DoE:** a cross-detection confusion table (bias-detector on refusal set, refusal-detector on
  bias set) + interpretation consistent with the JZ-2 cosine matrix.

## AA-4 — Can we perform conditional steering?

The headline "application." Steer **only when** the detector fires, leave clean prompts alone.
- **Do:** wire AA-1's detector as a gate in front of the injection in `src/bias_steer` — steer
  iff detected-biased, measure both the debias effect *and* the collateral on
  already-neutral/unrelated prompts (the point of conditional steering is fewer side effects).
- **DoE:** debias Δ on biased prompts **and** a side-effect audit (capability + safety) on the
  unconditional-steering baseline vs conditional — showing conditional preserves capability
  better (`AGENTS.md` §5 requires side-effect audits for headline interventions).

## AA-5 — Improve ablation + layered-steering setup

- **Do:** consolidate the ablation operator + multi-layer injection into a clean, tested path.
  The ablation operator, prompt-position extraction, and deterministic judge are already on
  `main` (`PROJECT_STATE.md` §"Already closed" G0) — extend, don't rebuild. Feed the §7
  layer-placement findings (Aryaman's own notes: "10–16 works, 18–22 nonsense, 25–30 → EOS,
  29–32 → neutral") into a principled default layer band.
- **DoE:** a tested config (add a `tests/` case) that reproduces a known run, + the layer-band
  recommendation backed by the §7 sweep with coherence logged.

## AA-6 — S4 toolkit → apply vectors better; experiment with S4 ACE

- **Do:** evaluate the **S4** toolkit as a cleaner vector-application layer, and its **ACE**
  (affine concept editing) method. **Coordinate with Edward EL-1** — ACE should have ONE
  implementation shared between you, not two.
- **⚠️ Needs sourcing + doctrine check:** confirm what "S4" refers to and that it's usable with
  open-weight models (`AGENTS.md` §8); it is not currently vendored under `third_party/`
  (verify). ACE is exploratory (does-not-block scope) — method study, not submission scope.
- **DoE:** S4/ACE integrated behind the same harness interface and benchmarked against the
  EL-0 canonical injection on ≥1 model.

## AA-7 — Qwen 3.5 vs Qwen 3.6

- **⚠️ Doctrine check (important):** the model set is **frozen**. The submission model is
  `Qwen/Qwen3-8B` pinned to an immutable SHA (`PROJECT_STATE.md` §"Current gate";
  `RESEARCH_CONTRACT.md` §12 **A4**), and the Qwen-27B trio was cut for lacking `-Base`
  controls. A "Qwen 3.5 vs 3.6" comparison is **outside the frozen model set** — it is
  exploratory only and cannot become a submission number without a dated §12 amendment.
- **Do (if pursued as exploration):** cross-model comparison on a shared eval set, each model
  pinned to an immutable revision (A4), with `-Base` availability checked first (the reason the
  27B trio was cut). Ties into `needed-experiments §8` (cross-model transfer).
- **DoE:** a two-model comparison with pinned SHAs, marked clearly as exploratory / not in the
  frozen model set.

---

### Related `needed-experiments` Aryaman owns/should drive
- **§0.1** injection/coeff convention — Aryaman's `aryaman_adaptive_coeffs_and_norms` is the
  reference; converge with Edward EL-0. **§6** normalization ablation (empty result cells —
  finish them). **§7** coeff/layer ablation. **§8** cross-model transfer (feeds AA-7).

### ⚠️ Doctrine checks for AA's section
- Detection/conditional-steering (AA-1..AA-4) is a strong **applications** story but sits in the
  does-not-block-the-paper set — keep it from expanding the frozen submission scope.
- SAEs, ACE, second/third model: all exploratory. Pin every model to a SHA (A4). Random-
  direction and coherence controls on every claim.
