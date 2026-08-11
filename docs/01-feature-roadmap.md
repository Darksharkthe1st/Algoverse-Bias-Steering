# Feature Roadmap

A feature-by-feature inventory of what the current notebook
(`experiments/farhan-experimentation.ipynb`) does, mapped against what the
re-implementation needs to preserve, change, or add.

Legend for **Re-impl status**:
- **Port** — keep the behavior, move it into a proper module more-or-less as-is.
- **Rework** — the capability must survive but the mechanism should change (usually
  for extensibility, traceability, correctness, or reproducibility).
- **New** — capability the notebook lacks that the re-architecture requires.
- **Drop** — remove; supported a goal (resumability) that is now a non-goal (arch roadmap §9).

---

## 1. What has been built (current notebook)

### 1.1 Environment & model loading
| Feature | Where (cell) | Notes | Re-impl status |
|---|---|---|---|
| Device selection (CUDA / MPS / CPU) | `getDevice` (5) | Straightforward. | **Port** |
| Model load via `transformer_lens.HookedTransformer` | `get_model` (6) | `from_pretrained_no_processing`, fp16, left-padding, eval mode. | **Port** |
| HF auth / RunPod lifecycle | (35, 36) | `hf auth login`, `runpodctl stop pod` embedded in pipeline. | **Rework** (pull out of core loop) |

### 1.2 Data loading
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| BBQ loader | `src/data.py:load_bbq_dataset` | jsonl → templated question strings. | **Port** |
| Hidden-bias / CrowS / custom / plain loaders | `src/data.py` | Several format-specific loaders. | **Port** |
| Custom template × noun-pair expansion | `load_custom_dataset` | Cartesian pairing w/ order-flip to cancel order bias. | **Port** |
| Loading a dataset from a **prior run's pickle** | (37) `get_any_variable(...dataset.pkl)` | Reuses frozen prompt sets for reproducibility. | **Rework** (formalize as dataset snapshots) |
| Train/test split | `complete_test` (35) | Simple ratio slice (`train_split`). | **Port** |

### 1.3 Generation & residual extraction
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| Chat-template tokenization | `tokenize_prompts` (8) | System instruction + user turn; falls back to raw tokenize for base models. | **Port** |
| Plain batched generation | `normal_generation` (21) | Greedy (`do_sample=False`), strips prompt+BOS from output. | **Port** |
| Residual capture per layer | `batch_resids` (10) + `run_with_cache` | Mean over tokens of `blocks.{l}.hook_resid_pre` → `(n_layers, d_model)` per prompt. | **Port** (this is the scientific core) |
| Steered generation via forward hooks | `batched_generation` (22) | Adds `coeff/n_layers · steer_vec[l]` at every layer's `resid_pre`. | **Port** |
| Steering direction flip | `flip_steering` | Negates coeff to push toward neutral. | **Port** |

### 1.4 LLM-as-a-judge
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| Async batched GPT-4o-mini judging | `oai_llm_judges`, `generate_function` (13) | `asyncio.gather` over a batch. | **Rework** (behind the `judge()` contract; retry + rate-limit — no resume cache) |
| Verdict parsing (`ANSWER: <label>`) | `get_judgements` (12) | Hand-rolled substring scan; brittle. | **Rework** (robust parser, configurable label set) |
| Configurable judge rubric | `openai_sys_instruct` (36) | Prompt swapped by hand between neutrality / safety / bias runs. | **Rework** (rubric as versioned config) |
| Gemini judge path | imported but unused | `google.generativeai` imported; OAI used in practice. | **New/optional** (just another `JUDGES` entry — no backend hierarchy) |

### 1.5 Steering-vector computation
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| Bucket residuals by verdict | `calculate_batched_vectors` (18) | neutral / opinion / nonsense lists. | **Port** |
| Vector = mean(opinion) − mean(neutral) per layer | `get_opinion_vec_from_resids` (19) | The learned direction. | **Port** |
| **Partial resume from `ModelResiduals`** | (18) | Accepts prior residuals, counts `total`, continues. Was the seed of resumability. | **Drop** (resumability is a non-goal; keep only the verdict-bucketing logic) |
| Per-batch residual checkpoint | `log_residuals` (30) | Pickled `.resids` each batch. | **Rework** (persist residuals once as a run *output* in `safetensors`, not per-batch checkpoints) |

### 1.6 Steered evaluation / metrics
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| Three-way generate (initial / opinion / neutral) + judge | `batched_tests` (25) | The evaluation loop. | **Port** |
| Transition matrix | `GeneralResults` (24) | 3×3 initial→steered verdict counts. | **Port** |
| Steering-quality tallies | `TestResults` (24) | good/bad/same per opinion/neutral/nonsense. | **Port** |
| Resume from prior `model_responses` | `batched_tests` param | Supported in signature, not driven. | **Drop** (resumability is a non-goal) |

### 1.7 Vector analysis
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| Cosine similarity between vectors | `compare_vectors` (27) | Per-layer cosine, mean. | **Port** |
| Opinion-vs-refusal comparison | (38) | Checks whether "neutrality" ≈ "refusal". | **Port** |

### 1.8 Logging & persistence
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| Incrementing global log index | `setup_logging_directory` (29) | Reads/writes `farhan_logs/current_save.txt`. Race-prone, non-reproducible, per-user. | **Rework** (deterministic run IDs) |
| Per-run directory `Log_N_<nickname>/` | (29) | Groups all artifacts of a run. | **Port** (concept) / **Rework** (naming) |
| Pickle artifacts (vec, responses, residuals, dataset) | `log_*` (30) | `pickle` everywhere. | **Rework** (safetensors + JSONL) |
| Human-readable text logs | `textlog_*` (31) | Pre-steering + steered response dumps. | **Port** (keep as a rendered view) |
| CSV results table | `csvlog_*` (32) | One row per model run for the analysis sheet. | **Port** |
| Reload helpers | `get_*` (33) | Unpickle typed artifacts. | **Rework** (typed artifact store) |

### 1.9 Orchestration
| Feature | Where | Notes | Re-impl status |
|---|---|---|---|
| Multi-model sweep | `complete_test` (35) | Loops over index-aligned parallel lists (`model_names`, `chat_LLM`, `model_sizes`, `opin_coeffs`, `neut_coeffs`, `vector_files`). | **Rework** (structured experiment matrix) |
| `is_qwen = i < qwen_count` | (35) | Positional hack for a model quirk. | **Rework** (per-model capability flags) |
| Use precomputed vector if provided, else compute | (35) | Skip training when a vector exists. | **Port** (natural cache hit) |
| Git commit after each model | (35) | `!git add . && git commit`. | **Rework** (out of core; commit *summaries* only) |

---

## 2. Gaps the notebook does **not** cover (all **New**)

These are the capabilities the re-architecture exists to add, grouped under the four concerns
driving the design (see architecture roadmap §1). Every one is absent or only half-present today.

**Traceability**
1. **No link from a result back to its config or code.** Folders like `Log_220` don't record
   what config or which code version produced them. Needs a per-run `manifest.json` (full config
   + git SHA) and a browsable `runs/index.csv`.
2. **Non-reproducible run identity.** A global incrementing counter in a text file is per-machine
   and collides across users/branches. Needs a readable, self-describing run-id slug.

**Extensibility**
3. **Adding a dataset means rewriting shared code.** Loaders return bare `list[str]` in
   idiosyncratic shapes, so a new benchmark ripples through generation/judge/metrics. Needs a
   canonical `Example` schema every dataset maps into.
4. **Techniques/judges are hard-edited, not swappable.** The steering method is inlined and the
   judge rubric is a hand-swapped string. Needs name→component registries + rubric-as-config.
5. **Positional model config.** `is_qwen = i < qwen_count` and seven index-aligned lists encode
   model facts by position. Needs per-model `ModelSpec` records.

**Interpretability**
6. **Aggregate-only results.** The CSV logs pre-computed transition counts, so any aggregate not
   anticipated up front requires a re-run. Needs tidy long-format per-response rows.
7. **No decoupled analysis path.** Comparing experiments means manually opening logs. Needs a
   standalone `analysis/` that reads outputs without importing the run engine.

**Configurability / hygiene**
8. **Parameter-threading tax.** `complete_test(...)` takes ~14 positional args funneled to
   sub-functions. Needs one grouped `ExperimentConfig`.
9. **Separation of concerns.** Compute, judging, storage, and orchestration are interleaved in one
   async function, hard to test or swap.
10. **Non-portable artifacts.** Pickle is version-fragile and unsafe to load; tensors belong in
    `safetensors`, tabular results in CSV.

*(Deprioritized — was a gap, now a non-goal: crash-safe resume, idempotent work units, judge-call
caching, signal handling. See architecture roadmap §9.)*

---

## 3. Re-implementation priority (feature view)

Ordered by "what unlocks the four concerns first" (mirrors architecture roadmap §11):

- **P0 — Contracts & config (traceability + configurability foundation)**
  - `Example`/`Result` schema; `ExperimentConfig` + sub-specs; registries
  - `manifest.json` (config + git SHA) + `runs/index.csv`; readable run-id slug
- **P1 — Port the science behind the contracts (extensibility)**
  - `datasets.py` over `src/data.py`; `models.py` spec loader + generation
  - `steering.py` (mean-diff = current behavior); `judge.py` (neutrality + robust parsing)
- **P2 — Wire + persist (interpretability)**
  - `run(config)` end-to-end; tidy `results.csv` + summary; reproduce a known past result for parity
- **P3 — Payoff**
  - `analysis/compare.py` cross-run comparison; prove plug-and-play (add a dataset + a method, zero downstream edits)
- **P4 — Extensions**
  - More judges/rubrics; CLI wrapper; notebook kept as a thin driver over `run(config)`
