# Third-party dataset: `IssueBench`

Realistic writing-assistance prompts for measuring **issue bias** — the tendency
of an LLM to present a single perspective on a contested issue — from
[*IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM
Writing Assistance*](https://arxiv.org/abs/2502.08395) (Röttger, Hinck, Hofmann,
Hackenburg, Pyatkin, Brahman, Hovy — TACL 2025).

- **Source:** https://huggingface.co/datasets/Paul/IssueBench
  (upstream code: https://github.com/paul-rottger/issuebench)
- **License:** CC-BY-4.0 (see the HF dataset card). These files are redistributed
  here only by reference — they are fetched on demand and are **not committed**.
- **Pinned commit:** recorded in `manifest.json` (`source_commit`,
  `636e2da6…`, last modified 2025-02-21), per `docs/PREREG.md` §3b — a dataset
  used as evidence is pinned to an immutable revision, not a bare `main`.

## What this is, in this repo

IssueBench is built from **3.9k writing-assistance templates** (e.g. *"write a
blog post about ___"*) crossed with **212 political issues** (e.g. *AI
regulation*), yielding up to **2.49m prompts**. It is task **FK-5** in
`docs/work-splits/fk-task-list.md`: the boundary check for the single-direction
story — confirm that single-direction additive steering underperforms on
IssueBench / AxBench-style tasks, as the literature reports.

We consume the **pre-materialised prompt splits** rather than re-deriving the
template×issue expansion, so our prompts are byte-identical to the release.

| File | Split | What it is |
|---|---|---|
| `prompts/prompts_debug-*.parquet` | `debug` | 150 prompts — the default fetch; CI + pilot |
| `prompts/prompts_sample-*.parquet` | `sample` | 636k prompts — stratified subsample |
| `prompts/prompts_full-*.parquet` | `full` | 2.49m prompts — the whole bench (~79 MB) |
| `issues/issues-*.parquet` | `meta` | 212 issues: `topic_id` → neutral/pro/con framings, tags, source mix |

Prompt-split columns we use (HF schema): `prompt_text` (the full user prompt),
`topic_text`, `topic_polarity` (`neutral`/`pro`/`con`), `topic_id`,
`template_id`. The `issuebench` loader (`src/bias_steer/datasets.py`) maps
`prompt_text` → `Example.prompt` and stashes the rest in `metadata`
(`category = topic_polarity`, so the generic `sample(per_group=…)` stratifier
works unchanged).

## How to fetch

```bash
python scripts/fetch_issuebench.py                 # debug split (+ issue metadata)
python scripts/fetch_issuebench.py --split sample  # add the 636k subsample
python scripts/fetch_issuebench.py --split full    # the whole 2.49m bench
python scripts/fetch_issuebench.py --force         # re-download
```

Files land under `third_party/issuebench/` (git-ignored). Reading them needs
`pandas` (already a project dependency); the fetcher itself is stdlib-only.
