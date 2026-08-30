# Third-party artifacts: `refusal_direction`

Pre-computed **refusal direction** artifacts from the paper
[*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717)
(Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda — NeurIPS 2024).

- **Source:** https://github.com/andyrdt/refusal_direction
- **License:** Apache-2.0 (see the upstream repo). These files are redistributed
  here only by reference — they are fetched on demand and are **not committed**.
- **Pinned commit:** recorded in `manifest.json` (`source_commit`).

## What we use these for

We test the paper's *own* steering vectors inside this repo's framework, rather
than re-deriving them. For each model we fetch:

| File | What it is |
|---|---|
| `runs/<model>/direction.pt` | the single selected refusal direction, shape `(d_model,)`, stored **raw** (un-normalized) |
| `runs/<model>/direction_metadata.json` | `{layer, pos}` — where the direction was extracted from |
| `runs/<model>/generate_directions/mean_diffs.pt` | full candidate grid `(n_pos, n_layers, d_model)` — used for a provenance check |
| `runs/<model>/completions/*.json` | the paper's committed prompts + responses (baseline / ablation / actadd) |
| `runs/<model>/completions/*_evaluations.json` | their per-prompt refusal labels + aggregate rates — our **ground-truth for validating our refusal metric** |

Models: `gemma-2b-it`, `llama-2-7b-chat-hf`, `meta-llama-3-8b-instruct`,
`qwen-1_8b-chat`, `yi-6b-chat`.

## How to fetch

```bash
python scripts/fetch_refusal_artifacts.py          # all models (~54 MB)
python scripts/fetch_refusal_artifacts.py --model qwen-1_8b-chat
```

The downloaded files live under `runs/` here and are git-ignored. Only this
README, `manifest.json`, and the fetch script are committed.

## Key facts (see the paper / `pipeline/utils/hook_utils.py` upstream)

- **Directional ablation** (bypasses refusal): `x -= (x·r̂) r̂` with `r̂ = direction/‖direction‖`,
  applied at every layer's residual input, attention output, and MLP output, all token positions.
- **Activation addition** (induces refusal): `x += coeff · direction` (raw vector)
  at the single source layer; `coeff=+1` induces refusal on harmless prompts,
  `coeff=-1` bypasses refusal on harmful prompts.
- **Refusal scoring:** substring match against a fixed list of refusal prefixes
  ("I'm sorry", "I cannot", "As an AI", ...).
