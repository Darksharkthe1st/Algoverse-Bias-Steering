# HANDOFF — Jeremiah (onboarding → experiments + writing)

> Read `PAPER_FRAMING.md` before writing any paper/summary text. Disagree → PR
> that file, don't fork the narrative.

You have the math (linear algebra + multivariable calc); what's new is the
transformer internals vocabulary and this project's specific method. The path
below is ordered so each step makes the next one make sense. Use AI to
interrogate everything you don't understand — that's the intended workflow here.

## Week 1 — orientation (no GPU needed)

0. **`docs/THE_CORRECT_PROBLEM.md`** — read this first, before anything else.
   It is four pages on the one mistake that killed the 2025 run, shown twice:
   once here, once in a completely unrelated corner of AI research. If you take
   one transferable skill out of this project, it is that one.
1. **`docs/2026-08-01_project_analysis.md`** — Part 1 tells you exactly what
   this project did, what worked, and what broke. Read it before any paper.
2. **Arditi et al., "Refusal in Language Models Is Mediated by a Single
   Direction" (arXiv:2406.11717)** — the paper ours is built on. Everything in
   our method (difference-in-means, steering hooks, ablation) comes from here.
   The math is exactly what Farhan described: subtract two mean vectors, add
   the result back at inference.
3. **3Blue1Brown's neural-network series** (the transformer + attention
   chapters) — for the internals picture behind "residual stream" and "layers".
4. **TransformerLens "Main Demo" notebook** — run it on any small model (CPU is
   fine); this is the library all our code uses.

5. **Help hand-label the ~150 gold examples** for the judge-v2 rubric (with
   Edward). This is deliberately your first real task: reading 50
   model responses and deciding "did it take a side? was it hedging?" teaches
   the construct faster than any paper.

## Week 2 — hands on the real thing

6. **Reproduce one archived experiment without a GPU.** Pick a `Log_N_*` dir
   under `experiments/past_logs/methodology_experiments/batched_tests/`, load
   its `_responses.pkl` and `_steer_vec.pkl`, and recompute the judge counts
   from the raw responses. You'll touch every moving part of the pipeline
   except generation itself.
7. **Read the frontier section** of the analysis doc (Part 2) + skim QCRI
   arXiv:2602.02132 — the paper our headline claim is positioned against.

## Weeks 3–4

MMLU capability-audit runs + judge-disagreement error analysis (week 3);
repro pass — re-run one grid cell purely from README instructions — and the
eval appendix (week 4), alongside the writing role below.

## Your lane once drafting starts

Writing lead. The 2025 outline exists but is superseded by `PAPER_FRAMING.md` —
claims, terminology, and must-cites all live there. Before-and-after steered
chat snippets (from the archived `_steered.txt` / `_pre-steering.txt` logs) are
yours to curate; they're the paper's most persuasive exhibit.
