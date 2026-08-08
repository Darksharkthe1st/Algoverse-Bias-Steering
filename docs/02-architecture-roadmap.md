# Architecture Roadmap

How to re-implement the notebook pipeline as a small, module-based system organized around
the concerns that actually cost time in the research loop:

1. **Traceability** — always know *which* experiment a result came from and *what code* produced it.
2. **Plug-and-play extensibility** — add a dataset / model / steering technique / judge without
   rewriting code that everything else shares.
3. **Result interpretability** — structured, tabular outputs you can compare after the fact
   without sifting through logs.
4. **Configurability** — one place to set every lever, no threading parameters through five functions.

Resumability, which drove earlier drafts of this doc, is **explicitly demoted to a non-goal**
(see §9). Batching made runs short enough that re-running beats resuming.

This document is design-only: structure, contracts, and data flow — not implementation.

---

## 1. What we're optimizing for

| Concern | The design move | Where |
|---|---|---|
| Traceability | Serialize the full config + git SHA into a per-run `manifest.json`; keep a browsable `index.csv` of all runs. | §6 |
| Extensibility | A canonical `Example`/`Result` schema + small name→component registries. Adding a dataset/method/judge is one function + one registry line. | §3 |
| Interpretability | Three parallel views per run: tidy `results.csv` (analysis) + plaintext `logs/` (reading) + tqdm CLI (watching); a standalone `analysis/` reads outputs. | §7 |
| Configurability | One `ExperimentConfig` object of grouped sub-specs; functions take config, not loose params. It is *also* the traceability record. | §4 |

The elegance is that these interlock: the **config object** is both the configurability answer
(§4) and the traceability record (§6); the **schema + registries** are both the extensibility
answer (§3) and what makes results uniform enough to be tabular (§7).

---

## 2. Two ideas do most of the work

**Idea A — the config object is the experiment's identity.** All levers live in one
`ExperimentConfig`. Functions receive it (or a sub-spec) instead of long parameter lists. When a
run starts, that object is serialized verbatim into the run's `manifest.json` alongside the git
commit SHA. Configuring an experiment and recording what an experiment *was* become the same act.

**Idea B — a canonical schema + registries make the moving parts swappable.** Every dataset maps
its raw files into a common `Example`; every stage speaks `Example`/`Result` and never knows which
dataset produced them. Datasets, steering methods, and judges are looked up by name from plain
dicts. Adding one is a local edit — write a function, register it — with zero downstream churn.

Everything below is an application of these two ideas.

---

## 3. Abstraction contracts

This is the heart of the design and the answer to *"non-limiting, but not 500 levers."*

### 3.1 The governing rule

Each abstraction is **a narrow required contract plus one open escape hatch.**

- The **narrow contract** is a fixed function signature (or a small declarative spec). It's what
  keeps the lever count tiny and every component uniform.
- The **escape hatch** is an open `metadata` / `params` dict. It's what keeps the system
  non-limiting: anything a specific dataset/method/judge needs that the others don't goes here,
  invisible to code that doesn't care.

The corollary — a rule we enforce: **new capability = a new registered component or a new
`metadata` key, never a new top-level lever.** That is precisely how we stay expressive without
the config growing 500 knobs.

A second, quieter rule: **behavior is a function; uniform behavior is data.** Datasets, methods,
and judges are *functions* you write (they genuinely differ). Models are *specs* you declare
(under `transformer_lens` they behave uniformly, so adding one needs no code).

### 3.2 The shared schema (the contract's currency)

```python
@dataclass
class Example:
    id: str                 # stable, dataset-scoped
    prompt: str             # fully-rendered USER-turn text (no chat template yet)
    metadata: dict = {}     # escape hatch: options, category, gold, bias_axis, raw row...

@dataclass
class Result:
    example_id: str
    condition: str          # "initial" | "steered_pos" | "steered_neg"
    response: str
    verdict: str            # a label from the judge's label set
    # run coordinates (run_id, model, dataset, method, coeff) are stamped on at write time
```

`Example.prompt` is deliberately the *user turn only*. Chat-template wrapping and the system
instruction are applied later (by the model layer, from config) — so the same dataset feeds a
chat model and a base model unchanged. This fixes a real coupling in the notebook, where
`tokenize_prompts` hard-coded one system instruction.

### 3.3 Dataset — a function that yields Examples

```python
# contract
def load(spec: DatasetSpec) -> list[Example]: ...
# registry
DATASETS = {"bbq": load_bbq, "crows": load_crows, "plain": load_plain, ...}
```

- **Required:** produce `id` + `prompt` per item.
- **Escape hatch:** stash format-specific fields (BBQ's 3 options, a gold answer, a bias category)
  in `metadata`. Generic stages ignore it; a dataset-aware judge or metric reads it.
- **Adding one:** write `load_x() -> list[Example]`, add a line to `DATASETS`. Nothing downstream
  changes. ← directly kills "rewrite a chunk all datasets share."
- Existing `src/data.py` loaders become the bodies of these functions.

**Selecting representative samples** (e.g. BBQ's categories) is a *separate, dataset-agnostic*
step — not per-loader code — precisely because the category already lives in `Example.metadata`.
One config spec + one function covers every dataset:

```python
@dataclass
class SampleSpec:
    filter:    dict[str, list] = {}          # keep Examples whose metadata[k] ∈ v
                                             #   e.g. {"category": ["Age", "Religion"]}
    per_group: tuple[str, int] | None = None # stratify: N per distinct metadata[key]
                                             #   e.g. ("category", 50) → balanced across categories
    limit:     int | None = None             # global cap after filtering
    seed:      int = 0                       # reproducible subset; recorded in the manifest

def sample(examples: list[Example], spec: SampleSpec) -> list[Example]: ...  # filter → stratify → cap
```

This is how you "map out representative samples without bloat": getting 50 balanced samples per BBQ
category is `per_group: ["category", 50]` in config, not a hand-split of prompt blocks. Adding a new
dataset never touches sampling. The `seed` lands in the manifest, so the exact subset is
reproducible (traceability). Any *new* sampling axis is a new `metadata` key — never a new lever.

### 3.4 Model — a spec you declare, not code you write

```python
@dataclass
class ModelSpec:
    name: str               # short handle used in configs + filenames
    hf_id: str              # e.g. "Qwen/Qwen1.5-7B-Chat"
    chat_template: bool
    size: str               # "7B" — for the results table
    quirks: list[str] = []  # e.g. ["qwen"] — replaces `is_qwen = i < qwen_count`
    backend: str = "transformer_lens"   # escape hatch for a future API model
MODELS = {"qwen-7b": ModelSpec(...), "gemma-2b": ModelSpec(...), ...}
```

- Adding a HookedTransformer model = **add a dict entry**, zero new code, because all such models
  share one loader (`load_model(spec) -> LoadedModel`) and one generation path.
- The `backend` field is the escape hatch: a genuinely different source (API model) is the only
  case that needs a new loader, keyed off this field.
- This is why the model "lever" is nearly free: you declare models, you don't implement them.

### 3.5 Steering method — the three functions you'll live in

Honest framing, because this one is different from the other three: **this is not a large
abstraction, and we won't pretend it is.** These are the central functions you'll edit constantly
as the experiment setup evolves; the git SHA in each manifest is what actually records "which
version ran," and the code flows through the same calls no matter how they're packaged. Naming them
buys *not* reified swappability — it buys a **stable, consistent boundary around them**, so that
editing these functions is the only thing you ever have to think about. The separation of concerns
is effectively the same as the notebook's; the win is everything *surrounding* it being uniform.

The technique is three functions:

```python
capture(cache, model)        -> resid  [n_layers, d_model]   # notebook batch_resids: mean over tokens of hook_resid_pre
build(resids_by_label, cfg)  -> vector [n_layers, d_model]   # notebook get_opinion_vec_from_resids: mean(pos) − mean(neg)
apply(model, vector, coeff)  -> forward hooks                # notebook batched_generation: add coeff/n_layers · vec[l]
```

- **Contract:** fixed signatures (config in, `Example`s/residuals in, vector/hooks/`Result`s out) so
  the surroundings never shift under you. What's *inside* them is yours to rewrite freely.
- **Packaging is optional.** Group them in a `SteeringMethod` dataclass + `METHODS` registry *only*
  if you want two variants to coexist and be A/B-compared in one `index.csv`; that's a plain dict
  lookup, not a framework. Otherwise call the functions directly. The honest default is: "edit these
  three, everything else stays put."
- The +/− contrast (which verdict is the positive pole) is a config param, defaulting to the
  judge's two labels.

### 3.6 Judge — a function that labels responses

```python
# contract
def judge(responses: list[str], examples: list[Example], spec: JudgeSpec) -> list[str]: ...
JUDGES = {"neutrality": neutrality_judge, "safety": safety_judge, ...}

@dataclass
class JudgeSpec:
    name: str               # picks the JUDGES entry
    model: str = "gpt-4o-mini"
    labels: list[str] = ["neutral", "opinionated"]
    rubric: str = ...       # the system-prompt text (versioned in config)
```

- **Required:** return one label per response, drawn from `labels`.
- **Escape hatch:** the judge receives full `Example`s, so a dataset-aware judge (e.g. BBQ
  accuracy against a gold option in `metadata`) is possible without changing the contract.
- Swapping neutrality → safety → bias is a **config change** (different `JUDGES` entry + rubric),
  not a code edit. ← the notebook's hand-swapped `openai_sys_instruct` becomes data.
- Robust `ANSWER: <label>` parsing and retry live inside the judge, not scattered.
- Judge produces per-item `verdict`s; **aggregation is a separate concern** (`metrics.py`, §7) —
  a judge labels, a metric counts.

### 3.7 Summary: what you write to add each thing

| To add a… | You write | You register | Downstream changes |
|---|---|---|---|
| Dataset | `load_x() -> list[Example]` | `DATASETS["x"]` | none |
| Model (HF) | *nothing* — just a `ModelSpec` | `MODELS["x"]` | none |
| Steering technique | edit `capture`/`build`/`apply` | `METHODS` *(optional — only to A/B two at once)* | none |
| Judge / rubric | `judge_x()` (or reuse one + new rubric) | `JUDGES["x"]` | none |

Four narrow contracts, (up to) four dict registries, one shared schema. That's the entire
extensibility surface. **Sampling is not on this list on purpose** — it's dataset-agnostic
(§3.3), driven by config over `metadata`, so it's never something you re-implement per dataset.

---

## 4. Configuration model

All levers in one grouped object; component-specific params nested in their own specs so the
top level stays short.

```python
@dataclass
class ExperimentConfig:
    label:       str                    # human name → into run_id + index.csv
    models:      list[str]              # keys into MODELS
    dataset:     DatasetSpec            # name (key into DATASETS) + path + split
    sample:      SampleSpec = SampleSpec()  # filter + representative stratified sampling (§3.3)
    judge:       JudgeSpec              # §3.6
    method:      str = "mean_diff"      # key into METHODS
    coeffs:      Coeffs                 # opinion / neutral steering strengths
    system_prompt: str = DEFAULT_SYS    # the instruction wrapped around every prompt
    max_tokens:  int = 128
    batch_size:  int = 32
```

Rules that keep the lever count bounded:
- **Select-by-name, don't expose internals.** `method: "mean_diff"` is one lever; the method's
  guts are not top-level knobs.
- **Component params live with the component.** Judge model/rubric are in `JudgeSpec`, not five
  loose top-level fields.
- **Defaults everywhere.** A minimal config is a few lines; depth comes from registries +
  `metadata`, never from new global knobs.
- **This object is the manifest.** Serialized verbatim per run (§6) — configuring and recording
  are one act.

This directly replaces the notebook's seven index-aligned parallel lists (`model_names`,
`chat_LLM`, `model_sizes`, `opin_coeffs`, `neut_coeffs`, `vector_files`, `qwen_count`) and the
long `complete_test(...)` signature.

---

## 5. Source file map & data flow

### 5.1 Module tree (~a dozen focused modules)

```
Algoverse-Bias-Steering/
├── src/
│   ├── data.py                 # EXISTS — raw loaders; bodies of datasets.py entries
│   ├── utils.py                # EXISTS — repo_root, time strings
│   └── bias_steer/             # NEW package
│       ├── config.py           # ExperimentConfig + sub-specs — every lever (§4)
│       ├── schema.py           # Example, Result — the shared contract currency (§3.2)
│       ├── registry.py         # DATASETS / MODELS / METHODS / JUDGES dicts + register() (§3)
│       ├── datasets.py         # load_* -> list[Example] + sample() (wraps src/data.py) (§3.3)
│       ├── models.py           # ModelSpec loader + tokenize/chat-template + generation (§3.4)
│       ├── steering.py         # capture / build / apply (+ optional SteeringMethod/METHODS) (§3.5)
│       ├── judge.py            # judge_* evaluators + ANSWER parsing + retry (§3.6)
│       ├── metrics.py          # verdicts -> tidy rows + transition/quality tallies (§7.1)
│       ├── logs.py             # plaintext run.log / train.txt / eval.txt writers (§7.2)
│       ├── tracking.py         # run-id slug, manifest.json, git SHA, index.csv (§6)
│       ├── experiment.py       # run(config): wire everything, write outputs
│       ├── cli.py              # entrypoint: default = one run (teammates, §7.3); --queue = batch (§10)
│       └── coordinator.py      # --queue engine: route→checkout→drain→commit/push (§10); committed + frozen
├── analysis/                   # NEW — standalone; reads outputs, never imports the engine (§7.1)
│   └── compare.py
├── configs/                    # NEW — experiment config files (referenced by route.json, §10.3)
└── experiments/                # LEGACY — LEFT UNTOUCHED: notebook, past_logs/, best_vecs/, etc.
                                #   the new system does not modify or depend on this dir (§12)
```

`schema.py` + `registry.py` may merge into one `core.py` if you prefer fewer files; `sample()`
lives inside `datasets.py` (not its own module). Everything you edit *often* (datasets, steering,
judge, config) is its own module; the rarely-touched wiring is isolated in `experiment.py`, and the
plaintext-logging plumbing is quarantined in `logs.py` so the science functions stay clean.

### 5.2 Data-flow trace (one `run(config)`)

```
ExperimentConfig  (from a small Python config or CLI)
   │ tracking.py → run_id slug + git SHA; open runs/<run_id>/ + logs/run.log
   │ registry.py → resolve dataset/method/judge
   │ datasets.py → list[Example]  →  datasets.sample() (filter + stratify by metadata, seeded) → frozen subset
   ▼
for each model key in config.models:                     [tqdm: outer bar over models]
   models.py         load ModelSpec → HookedTransformer                → logs/run.log
   │
   ├─ TRAIN split ───────────────────────────────────────  [tqdm: bar over batches]
   │   models.py + steering.capture   generate + capture residuals per Example
   │   judge.py                        label each response  → verdicts
   │        └─ append  prompt → response → verdict  to logs/train.txt   (tail-able live)
   │   steering.build                  group residuals by verdict → steering_vector
   │
   ├─ TEST split ────────────────────────────────────────  [tqdm: bar over batches]
   │   models.py                        initial generation
   │   steering.apply + models.py       steered_pos, steered_neg generations
   │   judge.py                         label all three conditions → Results
   │        └─ append  prompt → initial/steered± → verdicts  to logs/eval.txt  (tail-able live)
   │
   └─ WRITE (experiment.py + tracking.py) ────────────────
       manifest.json              full config + git SHA + timestamp + sample seed
       results.csv                tidy: one row per (example × condition)  ← analysis input
       steering_vector.safetensors
       residuals.safetensors      (git-ignored — bulky)
       append one row to runs/index.csv   (headline metrics + coordinates)

analysis/compare.py   (later, separately)
       reads runs/index.csv + selected results.csv → comparison tables
```

Note there is no ledger, no stage state machine, no scan-and-skip — a run is a straight function
from config to outputs. The `logs/*.txt` writes happen incrementally *during* each phase, so
`tail -f` gives you a live prompt-by-prompt view alongside the tqdm bars (§7.2–7.3).

---

## 6. Traceability

The answer to "which experiment was this, and what code produced it":

- **Per-run `manifest.json`** — the `ExperimentConfig` serialized verbatim, plus:
  `git_sha = git rev-parse HEAD`, a `dirty` flag (uncommitted changes present?), timestamp, and
  the resolved dataset snapshot hash. `git checkout <git_sha>` reproduces the exact code.
- **`runs/index.csv`** — one row per run: `run_id, label, model, dataset, method, opin_coeff,
  neut_coeff, git_sha, timestamp, <headline metrics>`. This is the browsable ledger of "what
  experiment was what" — sortable in Excel, `grep`-able, the thing that was missing when folders
  were just `Log_220`.
- **Readable run IDs** — `run_id = <YYYYMMDD-HHMMSS>_<label>_<model>` (a slug, not a hash), because
  with resume gone we optimize IDs for human scanning, not cache-hit matching.

### 6.1 On dirty working trees

If `dirty` is true, the SHA alone doesn't pin the code. Options (pick at build time): refuse to run
dirty, or auto-write a `code.patch` of the diff into the run folder. Recommended default: **warn +
save the patch**, so traceability holds without blocking quick iteration.

---

## 7. Outputs: structured results, plaintext logs, live progress

A run emits three parallel views of itself, each for a different consumer: **CSV for analysis,
plaintext logs for reading, a progress view for watching.** They coexist because they answer
different questions ("what were the numbers," "what exactly happened," "how far along is it").

### 7.1 Structured results — for analysis (pandas / Excel)

**Running produces data; analysis consumes it. They are separate and never re-run each other.**

- **Tidy long-format `results.csv`** — one row per `(example × condition)`:
  `run_id, model, dataset, condition, coeff, example_id, verdict` (+ optional `category` from
  `metadata`). Long format is the unlock: transition matrices, per-category rates, and cross-model
  comparisons are all one `groupby` / pivot, computed *after the fact*. The notebook's
  aggregate-only CSV couldn't do this — you had to have logged the right aggregate up front.
- **`metrics.py`** still computes the familiar aggregates (transition matrix, good/bad/same
  tallies) for the per-run `summary.md` and the `index.csv` headline — but they're derived from the
  tidy rows, not the only thing saved.
- **`analysis/`** is standalone: it imports pandas and reads `runs/index.csv` + selected
  `results.csv`, and never imports `bias_steer`. So re-analysis never risks a re-run, and (with AI
  assistance) ad-hoc comparison scripts are cheap to write against a stable, tabular contract.

### 7.2 Plaintext logs — for reading exactly what happened

Every phase writes human-readable text to `runs/<run_id>/logs/`, mirroring the notebook's
`_pre-steering.txt` / `_steered.txt` habit (every prompt + response recorded verbatim):

- `logs/run.log` — the event stream: model loaded, phase started/finished, counts, timings, errors.
- `logs/train.txt` — per train example: `prompt → response → verdict` (≈ `textlog_initial_responses`).
- `logs/eval.txt` — per test example: `prompt → initial / steered_pos / steered_neg → verdicts`
  and the running tallies (≈ `textlog_steered_responses`).

These are **committed** (plain text, human-scannable, part of the run's traceable record) and
written **incrementally** as each batch completes — so `tail -f runs/<id>/logs/eval.txt` is a live,
prompt-by-prompt window into a running experiment. They are a separate concern from `results.csv`:
logs are for eyes, the CSV is for pandas. `metrics.py`/`experiment.py` own the log writers so the
science functions stay clean.

### 7.3 Live progress — for watching a run

A thin CLI wrapper (`python -m bias_steer run <config>`) drives the run and prints a live view, the
same feel as watching the notebook:

- A header at start: `run_id`, model, dataset, train/test sizes, key coeffs.
- **tqdm** bars: an outer bar over models, inner bars over batches per phase (generate / judge /
  eval), with a postfix showing the running verdict tallies (neutral / opinionated / nonsense) —
  the structured cousin of the notebook's progress prints.
- On finish: the run's headline metrics and the path to its folder.

Deliberately lightweight — tqdm + prints, not a TUI framework. For an even more granular view,
`tail -f` the logs from §7.2 in a second terminal.

---

## 8. On-disk layout (committed by default)

Runs are **committed to git by default** — the safe failure mode. Only bulky residual tensors and
an explicit `runs/_discard/` opt-out pile stay out of version control.

```
runs/                                   # C  committed by default
  index.csv                             # C  the run registry (browsable in Excel)
  _discard/                             # I  throwaway runs you don't want tracked
  <run_id>/                             # C  <date>_<label>_<model> slug
    manifest.json                       # C  full config + git SHA + timestamp (traceability)
    results.csv                         # C  tidy: one row per (example × condition)
    steering_vector.safetensors         # C  the DELIVERABLE — tiny (<1MB)
    residuals.safetensors               # I  bulky (~200MB+), regenerable — NOT backed up
    logs/                               # C  plaintext, human-readable, tail-able live (§7.2)
      run.log                           #      events: model loaded, phase, counts, timings
      train.txt                         #      prompt → response → verdict   (≈ pre-steering.txt)
      eval.txt                          #      prompt → initial/steered± → verdicts (≈ steered.txt)
    summary.md                          # C  optional curated summary
```

### 8.1 The `.gitignore` (narrow; residuals excluded *by name*)

```gitignore
# Runs are committed by DEFAULT (never lose an experiment). Exceptions:
runs/**/residuals.safetensors    # bulky, >100MB GitHub cap, regenerable from committed inputs
runs/_discard/                   # throwaway runs
_coordinator/                    # ephemeral batch STATE (route/queue/control/status) — §10.3
                                 #   NB: the coordinator CODE lives in src/bias_steer/ and IS committed
```

`runs/` itself is not ignored; the small steering-vector `.safetensors` (the deliverable) is
committed — which is why residuals are excluded by name, not by a blanket `*.safetensors`.

### 8.2 Formats
- **Tensors** → `safetensors` (portable, safe) — replaces `.pkl`/`.resids`.
- **Tabular results / index** → CSV (your preference; Excel- and pandas-friendly).
- **Config/manifest** → JSON (nested). **Summaries / dumps** → Markdown / text.

### 8.3 Discarding a run
New runs write into the committed area. To drop one from git: `mv runs/<run_id> runs/_discard/`.
Nothing in the pipeline reads `_discard/`; location is the only signal `.gitignore` understands.

---

## 9. Resumability — explicit non-goal

Deprioritized per project direction: batching made runs short, so re-running a bad run beats the
machinery of resuming one. We therefore **do not build**: a ledger/state-machine, scan-and-skip
per-unit processing, idempotent unit keys, signal-handling checkpoints, or a resume cache. Their
removal is the single largest simplification versus earlier drafts.

What we keep, reframed: intermediate artifacts (responses, verdicts, vector) are written as
**outputs that serve traceability and interpretability**, not as checkpoints.

Where the one bit of "resume" that *does* survive lives: **batch-level restart** in the
coordinator (§10). Because each run owns a self-contained folder and the queue tracks a
pending/done cursor, restarting a killed batch simply skips configs already marked done — no
per-unit resume machinery, just "don't redo a finished run."

---

## 10. Batch running: the coordinator

Running experiments back-to-back (A finishes → B begins → C begins), unattended, with **branches
as the durable unit of experimental state.** Because results are committed-by-default (§8), a
branch holds *exactly* one campaign's code + configs + results; pushing it backs the whole thing up.
The queue is just ephemeral scheduling on top. Execution is **sequential and single-node by design**
— no parallelism, no worktrees; a queue is meant to be simple.

### 10.1 One command, two modes — default runs normally, `--queue` orchestrates

The coordinator is the **single user-facing command**, and it does the simple thing by default:

- **Default (no flag):** run one experiment, exactly like `run(config)`. This is all a teammate ever
  needs — they never have to learn the queue system. Git-agnostic; writes results to `runs/`.
- **`--queue`:** switch on the batch orchestrator — validate `route.json`, drain the config queue one
  at a time, check out branches, commit/push at phase boundaries. This is the campaign workflow (you
  set it up for yourself); teammates simply don't pass the flag.

So the queue machinery is **opt-in behind a flag**, not something everyone has to understand.

In `--queue` mode it owns the process:

| Tier | Job | Git? | Lifetime |
|---|---|---|---|
| **Coordinator** (`--queue`) | read `route.json` → check out a branch → drain its config queue, spawning **one** run subprocess at a time → commit/push at phase boundaries → advance | **sole automatic git writer** | long-lived, one instance |
| **`run(config)`** (subprocess, = default mode) | one experiment (§5.2); pure compute + file output; emits phase-complete events | **git-agnostic** | per run |

The coordinator is the *only* thing that launches a run in `--queue` mode, so **two experiments can
never run at once** — the single-writer guarantee is structural, not a lock we hope holds. Each run
is a **subprocess** (`python -m bias_steer run <config>`), so an OOM / segfault kills only that run;
the coordinator marks it failed and continues (soft-land), and the branch's code can change under it
without destabilizing the orchestrator. It inherits the subprocess's stdout so the tqdm bars (§7.3)
show live, printing campaign-level status around them.

**The coordinator is committed to git, and deliberately frozen.** It ships with the repo so every
teammate and every fresh machine *has* it — no copying a script around. The obvious worry (a
tracked file changes across branches, and `checkout` would swap it) is resolved by treating
`coordinator.py` as **stable-by-convention: we intend never to change it**, so all branches carry
identical bytes and a swap is a no-op. And the *running* coordinator already holds its code in
memory, so even an on-disk swap mid-run can't affect the live process. If it ever must change, that's
a coordinated, team-wide protocol bump — never a per-experiment edit. (Only the coordinator **code**
is tracked; its **state** — route/queue/control/status — stays gitignored, §10.3.)

### 10.2 In-place checkout discipline (the careful rules)

`--queue` mode uses in-place `git checkout` (one working dir, sequential). That's simple but has
sharp edges; these rules keep it safe:

- **The coordinator code is checkout-safe by being frozen, not by being ignored** (§10.1). All
  branches carry identical `coordinator.py`, and the live process runs from memory, so an in-place
  `checkout` neither meaningfully changes it on disk nor touches the running orchestrator.
- **Commit before you switch.** Never `checkout` with uncommitted results; switch branches only
  *between* runs, never while a subprocess is alive. The gitignored control state (queue/route/…)
  survives the switch untouched — the whole point of gitignoring *state*.
- **One coordinator per branch.** Since each experimenter works their own branch, each branch has a
  single pusher → pushes fast-forward, no cross-contributor collisions.

### 10.3 Code vs. state: what's committed, what's gitignored

The split that makes "ship it to everyone" and "keep the queue local" both true:

- **Coordinator code → committed** (`src/bias_steer/coordinator.py` + the CLI). Frozen by convention
  (§10.1); every clone and every machine has it automatically.
- **Coordinator state → gitignored** (`_coordinator/`). Ephemeral, per-machine, per-user; also the
  file-based control surface a supervising LLM (Claude) drives without a live session.

```
_coordinator/                 # gitignored → ephemeral, per-machine control state
  route.json                  # THE LEVER: ordered [{branch, configs:[paths], push}]  (edit to steer)
  queue/                      # ephemeral cursor: pending / done / failed  (survives checkout)
  control.json                # commands in:  pause | resume | skip | stop   (LLM or human writes)
  status.json                 # written out:  current branch/run/phase, queue depth, last push result
  coordinator.log
```

- **Configs are tracked on their branch** (durable); the queue holds only a pending/done **cursor**
  over them, not copies — so "state is tied to branches," exactly as intended.
- **Humans watch** the inherited tqdm CLI; **Claude supervises** by editing `route.json` / `queue/` /
  `control.json` and reading `status.json` + `index.csv` + `logs/` — the runner-vs-supervisor split
  from before, now with a concrete file interface.

### 10.4 Commit & push cadence — per pipeline phase

Per your call, backup granularity is the **pipeline phase**, not the whole run: the coordinator
commits + pushes after each phase boundary (e.g. after the steering vector is built, then again
after eval/tests). If a run dies after the vector but before tests, the vector is already safe on the
remote.

- **Coordinator is the sole git writer.** The run subprocess just writes files and signals
  phase-complete (a marker file / stdout sentinel); the coordinator does the `commit` + `push`. This
  keeps the science code git-free and version control in one reviewable place.
- **Push is best-effort.** A failed push (auth/network/non-fast-forward) is logged and retried later,
  never a halt — the local commit already means nothing is lost (preserves "never lose an experiment").
- **Auto-squash without force-push.** Frequent per-phase commits make the branch history granular
  (good for backup + "watch what happened"); the *clean* history comes from a **squash-merge when the
  branch folds into `main`** — a native git feature, teammate-safe. We deliberately do **not** rewrite
  + force-push the live branch to squash it: with multiple contributors that's a footgun. Commit
  messages are structured (`<run_id> · <phase>`) so even the granular log reads clearly.

### 10.5 Multi-contributor merges (optional nicety)

You noted branches carry their own `runs/`/`index.csv` and you'll merge overlap at branch-merge time.
To make those merges *conflict-free* rather than manual: store the index as **per-run fragment files**
(`runs/index/<run_id>.json`) and treat `index.csv` as a derived rollup (concatenate the fragments).
Two branches then add disjoint files (distinct `run_id`s) → a merge is a clean union, no conflict.
Low cost, and it directly serves the 3-contributor reality. (Skip if you'd rather keep one flat CSV
and resolve the occasional conflict by hand.)

---

## 11. Phased build order

- **Phase 0 — contracts & config**
  - `schema.py` (`Example`/`Result`), `config.py` (`ExperimentConfig` + sub-specs), `registry.py`
  - `tracking.py`: run-id slug, `manifest.json` (+ git SHA), `index.csv`
- **Phase 1 — port the science behind the contracts**
  - `datasets.py` over existing `src/data.py`; `models.py` (spec loader + generation)
  - `steering.py` (`capture_mean` / `build_mean_difference` / `apply_resid_pre_add` = current behavior)
  - `judge.py` (OpenAI neutrality judge + robust parsing)
- **Phase 2 — wire + persist**
  - `experiment.py`: `run(config)` end-to-end; `metrics.py` → tidy `results.csv` + summary
  - `logs.py` plaintext `run.log`/`train.txt`/`eval.txt`; `cli.py` header + tqdm bars
  - Reproduce a known past result to validate parity with the notebook
- **Phase 3 — interpretability & extensibility payoff**
  - `analysis/compare.py` (cross-run comparison from CSVs)
  - Prove plug-and-play: add one new dataset and one new steering method with zero downstream edits
- **Phase 4 — the coordinator (batch running, §10)**
  - `bias_steer coordinate`: `route.json` + gitignored `_coordinator/` control plane
  - One-run-at-a-time subprocess launcher; soft-land failures; batch-level restart (skip done)
  - Sole-git-writer commit/push at phase boundaries; best-effort push; squash-on-merge
  - Phase-complete signal from `run()`; `status.json` + `control.json` for LLM supervision
- **Phase 5 — optional**
  - More judges/rubrics; per-run index fragments (§10.5). (The legacy `experiments/` notebook is
    left as-is — not converted into a driver; see §12.)

---

## 12. Non-goals

- **Resumability / checkpoint-resume** (§9) — deliberately dropped.
- **Not** changing the science: residual definition, `coeff/n_layers · (mean_pos − mean_neg)`,
  greedy decoding, and metric definitions are ported verbatim as the default method.
- **Not** a class hierarchy of backends — registries of functions/specs instead.
- **Not** a distributed or parallel scheduler. Batch running (§10) is **sequential, single-node,
  one-run-at-a-time** — no worktrees, no concurrent experiments. Scale comes from queuing, not parallelism.
- **Not** live-branch history rewriting. Clean history comes from squash-*merge* (§10.4), never a
  force-push of a shared branch.
- **Not** replacing `transformer_lens` or `src/data.py` — they sit behind the contracts.
- **Not** touching `experiments/`. The legacy notebook, `past_logs/`, `best_vecs/`, and all existing
  logs stay exactly as they are — a frozen historical record. The new system lives in
  `src/bias_steer/` (+ `configs/`, `analysis/`) and neither modifies nor depends on `experiments/`.
  Reusing a legacy artifact (e.g. a past steering vector or dataset snapshot) is done by *reading a
  copy*, never by editing anything under `experiments/`.
