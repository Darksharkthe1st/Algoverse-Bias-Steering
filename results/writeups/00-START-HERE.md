# START HERE — Algoverse bias-steering, Jeremiah's workstream

**You are picking up a research project mid-flight. Read this file completely
before doing anything.** It orients you in about five minutes and tells you which
of the other documents you actually need.

Last session: 2026-08-20 → 2026-08-21. Experiment 1 is complete. Experiment 2 has
not started.

---

## 1. What this project is

The team studies whether social bias in language models is **one mechanism or
several**. Concretely: does each kind of bias (race, age, religion, disability…)
correspond to its own *direction* in the model's activation space, or are they
all the same direction wearing different labels?

Jeremiah owns this workstream. The team's framing, terminology and deadlines are
governed by the lab repo — see §5.

**Experiment 1 is done.** Its result, in one sentence: *two BBQ categories —
Disability_status and Physical_appearance — yielded reproducible bias directions
in every model tested; the rest did not, under the current extraction and
validation procedure.*

Denominators differ per model because each drops the categories that fail its
task control: qwen-7b and qwen-14b score 10, gemma-2b and yi-6b score 9,
qwen-1.8b scores 8. See `notes/09-open-questions-answered.md` Q2.

---

## 2. Where everything is

```
C:\Users\Jeremiah Zhang\research\soft-refusal-algoverse\
├── README.md          ← you are here
├── notes\             12 documents; see §3 for which to read
├── runs\              325 files, 100 MB — every result artifact
└── repo\              the lab's GitHub repo, on branch jz/bias-taxonomy
```

**Two separate git repositories.** Both local, neither pushed.

| repo | what it versions | branch | pushed? |
|---|---|---|---|
| project root | `notes/` + `runs/` + box scripts | `master` | n/a — no remote |
| `repo/` | the lab's code, our 21 changed files | `jz/bias-taxonomy` | **NO — deliberately** |

`repo/` is excluded from the outer repo via `.gitignore`. Do not nest them.

**Nothing has been pushed to GitHub.** Jeremiah decided this deliberately. Do not
push without asking. The branch is 21 files / ~5,100 lines ahead of `origin/main`.

---

## 3. Which documents to read

**Read these three, in this order:**

| doc | what it gives you |
|---|---|
| **`notes/11-EXPERIMENT-PROTOCOL.md`** | **Read first.** How to run the next experiment. A five-phase gate protocol, a pre-registration template, a confound checklist, and ten documented incidents from last session with the rule that prevents each. This is the most important document in the folder. |
| `notes/10-EXPERIMENT-1-FINDINGS.md` | What Experiment 1 found. Jeremiah's four research questions, answered, with the numbers and the limits on each claim. |
| `notes/09-OVERNIGHT-STATUS-REPORT.md` | Facts only: every run, every exit code, which artifacts are superseded and why. Consult when you need to know whether a number is still live. |
| **`notes/09-open-questions-answered.md`** | **Audit of seven challenges to the results, answered from artifacts.** Contains three corrections to the write-up and one unresolved threat to the probe-derived claims. Read before quoting any number. |

**Reference, read when relevant:**

- `notes/00-overview.md` — the two experiments in Jeremiah's own words. **Two
  framings are preserved deliberately; B is current.** Do not delete A.
- `notes/01-models-and-datasets.md` — model catalog, which are gated, per-layer
  norm profiles, dataset inventory
- `notes/08-results-2026-08-20.md` — full results with every caveat (29 KB)
- `notes/07-literature.md` — the papers, and what each means for the design
- `notes/06-steering-extension.md` — the queued steering work
- `notes/04-team-questions.md` — open questions for the team

**Stale, kept for provenance:** `notes/02`, `03`, `05`. Their method sections were
superseded twice. Do not plan from them.

**Published summary:** https://claude.ai/code/artifact/7e808c9e-1821-4b03-85af-dc2aeefd6c5c
(private; Jeremiah shares it from the page)

---

## 4. Environment — things that will waste your time if you don't know them

**Python.** Use `C:\Users\Jeremiah Zhang\anaconda3\python.exe`. The conda env
named `algoverse` is a **broken stub** — it has a `conda-meta` directory and no
interpreter. The base env has numpy, scipy, sklearn, matplotlib and pytest.
No torch locally, by design: all analysis code is torch-free and runs on CPU.

**Git** is installed but **not on PATH for this session**. Use the full path
`C:\Program Files\Git\cmd\git.exe`. `core.longpaths` is already enabled globally
— it is required, the repo has paths over 260 characters.

**PowerShell 5.1 gotchas that cost time last session:**
- No `&&` chaining. Use `;` and `if ($?) { }`.
- `>` redirection adds a UTF-8 BOM and rewrites line endings. **Never redirect
  into a patch or source file** — use `git diff --output=FILE` so git writes it.
- Never rewrite source files with string replacement; it mangled UTF-8 once.
  Use the editor.
- `Remove-Item` cannot delete long paths. Use the robocopy-mirror trick.

**Lambda GPU.** The instance from last session has been terminated. Jeremiah is a
collaborator on Farhan's Lambda team account with substantial credits. His SSH
key is at `~/.ssh/lambda_jeremiah` (private) and already registered with Lambda
as `jeremiah`. Use **1x A100 40GB SXM4**, **Lambda Stack 22.04**, no filesystem,
default firewall. Avoid the GH200 — it is ARM64 and the ML stack fights it.

**On a fresh box, these three collisions will bite in this order:**
```bash
pip install 'numpy<2'        # Lambda Stack's torch is compiled against numpy 1.x
pip install 'pillow>=9.1'    # system PIL predates Image.Resampling
pip install 'jinja2>=3.1'    # apply_chat_template needs it
```
`box_setup.sh` and `box_fix_numpy.sh` in the project root handle bring-up.

⚠️ **Security note:** Jeremiah's Hugging Face read token was pasted into the last
session's transcript. **Tell him to rotate it** at
https://huggingface.co/settings/tokens. His gemma access is granted on the
*account*, so a new token works immediately.

---

## 5. The lab repo's rules — these bind you

`repo/AGENTS.md` and `repo/CLAUDE.md` govern anything inside `repo/`. The ones
that matter most:

1. **Do not coin terminology.** The behaviour is *hedging*; "soft refusal" is
   retired. (The project folder is named `soft-refusal-algoverse` for historical
   reasons — leave it, but don't use the term in writing.)
2. **Say "a direction", never "the direction."** Steering success does not
   identify the representation (arXiv:2602.06801).
3. **Assert tensor shape `(n_layers, d_model)` before any cosine or steering.**
   A 1-D vector indexed per layer yields a scalar broadcast — a DC offset, not a
   direction. This bug produced a convincing table that survived a year.
4. **Numbers trace to a committed artifact**, cited by path.
5. **Never quote a number without its denominator.**
6. **Honest negatives stay honest.** Do not soften them and do not overclaim them.
7. The project is **frozen** (`RESEARCH_CONTRACT.md` §12). Work entering the paper
   needs a dated amendment.

---

## 6. How Jeremiah wants to work

These are standing preferences, learned by correction. They are also in the
agent's persistent memory, but state them here so they survive a memory reset.

- **Plan fully, then execute once.** Most of the effort goes into the plan;
  execution is one clean pass. No changing parameters, code or scope mid-run. If
  you are editing analysis code while a GPU bills, planning was not finished.
- **Stop him before mistakes.** Raise design problems *before* running, not in a
  caveat afterwards. He would rather be stopped than proceed into a bad result.
- **Evaluate critiques on the merits.** He often brings outside critiques from
  other models. Judge each claim independently and say plainly which parts are
  wrong. Agreeing with everything reads as instability and destroys the signal
  value of agreement. Check claims against data where you can.
- **Explain in plain language.** He is building the ML and linear-algebra
  background as he goes. Define terms on first use; prefer a concrete example with
  real numbers. He follows the substance well — don't simplify that, just don't
  assume vocabulary.
- **Never delete a superseded plan.** Mark it superseded and keep it. He has
  corrected this twice.
- **Only use browser automation when he explicitly asks.** It consumes his usage
  allowance quickly.

---

## 7. Experiment 1 — the state of the result

**Established:**
- `Disability_status` and `Physical_appearance` produce reproducible directions in
  all four capable models (Qwen-7B, Qwen-14B, Yi-6B, Gemma-2B) — three families,
  2B–14B, under two estimators, across six orders of magnitude of regularisation.
  **This is the claim that survives every audit challenge**, because the extremes
  estimator that produces it has no regularisation parameter.
- **No race-related category reproduces anywhere, under anything.**
- Among directions that do reproduce, clustering beats a permutation null in two
  models: **p=0.030** (Qwen-14B, 5 directions) and **p=0.005** (Qwen-7B, 3).
- The pipeline is validated: a control direction for *topic identity* reproduces
  at **0.86–0.92** through the identical machinery.

**Not established, do not claim:**
- That race/gender/sexuality "have no bias vector." Every failure is a failure to
  *recover* one with this contrast, at this n, at this capture site.
- That any direction is causal. Steering showed no category specificity at any
  dose, and three required controls were never run.
- That the *arrangement* of the clusters replicates across models — only three
  categories reproduce in both, which has no statistical power.

**Known soft spots** (full detail in `notes/09-open-questions-answered.md`):
- **Every probe-derived number is provisional.** The alpha sweep hit its boundary
  without finding an optimum on qwen-14b, and the 0.50 usable-floor threshold was
  calibrated against the *extremes* estimator only — never against the probe. A
  direction at alpha=1 and the same category's direction at alpha=1e6 agree at
  only 0.10–0.21, so alpha selects a direction rather than tuning one.
  **Resolving this is the first GPU task of the next session.**
- Qwen-7B's p=0.005 exists only because Religion clears 0.500 by 0.025 at one
  alpha, and that alpha was chosen after seeing that it did.
- Two of the four positive results rest on heavy-tailed margins where the top 5%
  of items carry ~half the variance.

---

## 8. What to do next

**Do not start Experiment 2 by writing code.** Start with
`notes/11-EXPERIMENT-PROTOCOL.md` §4 and produce `notes/PREREG-exp2.md`. A blank
field in that template is a blocked gate.

Candidate next experiments, in rough value order — Jeremiah decides:

1. **Winsorise the tails** on Physical_appearance and Age, re-extract, check the
   floor survives. Settles whether two of four positives are solid. Cheap.
2. **Split a failing category by stereotyped group** — extract for Black-targeted
   items alone rather than all of Race_ethnicity. If single-group subsets
   reproduce where the pooled category does not, heterogeneity returns as an
   explanation at the right granularity. This is the most interesting open lead.
3. **Finish the steering controls** — covariance-matched random direction,
   coherence check on generations, system-prompt baseline. Until these exist, no
   transfer result is causal.
4. **Replicate the clustering** with α fixed by a pre-registered rule rather than
   tuned, on a model held out from the tuning.

**Also outstanding, non-research:**
- Jeremiah should rotate the HF token (§4).
- The branch is unpushed; Farhan and Edward are building against a repo without
  these changes. Raise it, don't act on it.
- `runs/full_llama3/` is empty — Llama-3-8B is gated and the account is not
  authorised. Request access if it is wanted.

---

## 9. The one-paragraph version

Experiment 1 asked whether different kinds of bias have different directions
inside a language model. Two categories — disability and physical appearance —
have directions that reproduce reliably across five models and three families.
Race, gender, nationality and sexuality do not, anywhere, under any method tried.
Where directions do exist they are geometrically distinct and form above-chance
clusters, but steering along them shows no category specificity, which matches
what Joad et al. found for refusal. The measurement is validated by a control
direction that reproduces at 0.88 through the identical pipeline, so the negatives
are interpretable rather than a broken tool. The main cost of the session was
process, not science: roughly half the GPU time produced superseded artifacts
because the experiment was designed while it was being executed. Fixing that is
what `notes/11-EXPERIMENT-PROTOCOL.md` is for, and it should be read before
anything else.
