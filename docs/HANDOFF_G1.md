# HANDOFF_G1 — running the positive control on the Lambda box

**Canonical.** This one is not a scratchpad. It executes contract §12 **A6** and
`docs/PREREG.md` §7a, both frozen before any Qwen3-8B activation was inspected.
It may not redefine a criterion; if it disagrees with those, they win.

G1 is the **sole current gate**. Nothing downstream — G2, the θ statistic, the
paper's claim — may be read until it passes.

---

## Before you start

`transformer_lens` is not importable off the box, so every step below is
box-only. Nothing here has been executed against a model; the config, the
registry entry and the statistic are unit-tested, the model load is not.

```bash
python -c "import transformer_lens, torch; print(transformer_lens.__version__, torch.__version__, torch.cuda.is_available())"
```

## 1. Fetch the splits (~seconds, no GPU)

```bash
python scripts/fetch_refusal_artifacts.py
```

Arditi ships 71 files. **65 are per-model and we need none of them** — that is
the whole reason G1 is model-internal. The 6 we use are model-independent:
`dataset/splits/{harmful,harmless}_{train,val,test}.json`.

## 2. Confirm Qwen3-8B loads under TransformerLens — **do this first**

This is the one genuinely unverified precondition. It costs one model load and
it decides whether G1 can run at all today.

```bash
python -c "
from src.bias_steer.registry import MODELS
from src.bias_steer import models
spec = MODELS['qwen3-8b']
print('loading', spec.hf_id, '@', spec.revision)
m = models.load_model(spec)
print('n_layers', m.model.cfg.n_layers, 'd_model', m.model.cfg.d_model)
"
```

**If it raises on an unsupported architecture**, TransformerLens does not have
native Qwen3 support in the installed version. Do not silently fall back to a
different model — that is the substitution A6 exists to prevent. Options, in
order: upgrade `transformer_lens`; or load via `TransformerBridge`; or, if
neither works, stop and report it as a blocker so the model choice is a decision
someone makes on purpose. Record whichever path was taken in the run manifest.

## 3. Run G1b / G1c (the five arms)

```bash
python -m src.bias_steer refuse configs/g1_qwen3_8b.py
```

Writes `runs/<ts>_g1-qwen3-8b_Qwen3-8B/` with `results.csv`, `summary.md`,
`manifest.json`, `steering_vector.safetensors`. Check the manifest carries
`model_spec.revision = "b968826d9c46"` — PREREG §3b says a run without it is not
evidence. If OOM, drop `batch_size` to 8 in the config; do not change anything
else.

## 4. Run G1a (direction stability) from the cached residuals

`src/bias_steer/g1_stability.py`. It takes the cached harmful/harmless residuals
at the selected cell — **no second forward pass, no reload**:

```python
from src.bias_steer.g1_stability import assess
report = assess(harmful_resid, harmless_resid)   # (n, d) at the selected cell
```

500 permutations is the default and is nearly free, because a direction is a
difference of means over tensors you already have. Save `report` into the run
directory alongside the other artifacts.

---

## What passes, what fails

**All three legs, or no claim is made.**

| Leg | Passes if | Fails if |
|---|---|---|
| **G1a** estimable | `s_split` > `null_q99` **and** `s_split` ≥ 0.68 | either condition unmet |
| **G1b** causal | ΔP_refuse ≤ −0.15 on held-out `harmful_test`, with baseline refusal ≥ 0.60 there and ≤ 0.10 on `harmless_test` | ΔP > −0.15, or the regime checks fail |
| **G1c** specific | permuted and random directions each \|ΔP\| < 0.05, **and** `r̂_harm` beats each by ≥ 4 SE | a control reproduces the effect |

`assess()` returns `pass` for G1a directly, along with every number behind it
(`s_split`, `null_q99`, `null_median`, `p_permutation`, `alignment_full_est`).

**Read the legs diagnostically when one fails:**

- **G1a only** → the extraction recipe or the selected cell is wrong. The
  direction is not being estimated, so nothing downstream means anything.
- **G1b only** → the operator or the regime. Check the baseline refusal rate
  first: if it is under 0.60 the model is not refusing enough for a −0.15 move
  to be possible, and that is a prompt-formatting bug far more often than a
  finding. The Qwen1.5 run's own failure was exactly here (baseline 0.380).
- **G1c only** → the effect is generic rank-1 damage, not the refusal direction.
  This is one of the two worlds `RESEARCH_CONTRACT.md` §3 lists as *mimicking*
  a shared control, so it is the most dangerous single failure to wave through.

**If G1 fails, that fires stop-rule §12.2.** It is not retried against a
different model until it passes — that is precisely the move A6 forbids. Bring
the failure to the team call.

## What this run is not

It is **not** G2, and it is not licence to start G2. It touches no 27B model.
It does not compute θ. `runs/20260816-011914_refusal-repro_qwen-1.8b` stays
where it is as historical mechanism evidence (contract §12 A5, A6) — it is not
this gate and does not substitute for it.

## Known blockers

1. **TransformerLens ↔ Qwen3 compatibility is unverified.** Step 2 resolves it.
   This is the only thing that can stop G1 outright.
2. **Wiring G1a to the cached residuals is not written.** The statistic and its
   decision rule are implemented and tested; the ~20 lines that hand it the
   right `(n, d)` slice at the selected cell are not, because that code path
   needs a model to exercise.
3. **`bias-steering.exe.xyz` has no known deployment path** — tracked separately
   in `DECISION_LOG.md` D-018. It does not block G1.
