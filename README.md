# Revealing hidden biases by finding steering vectors for neutrality

A project as part of Algoverse, Summer 2025. We look for steering vectors that
move open-weight LLMs toward neutrality on controversial-but-not-harmful prompts,
and study how that relates to the refusal direction.

The pipeline lives in `src/bias_steer/` — a config-driven system that runs an
experiment (load model → capture residuals → build a mean-difference steering
vector → generate steered responses → judge them) and writes traceable, tabular
outputs. See `docs/` for the architecture.

## Directory structure

- `src/bias_steer`. The experiment pipeline (config, datasets, steering, judge, metrics).
- `src/data.py`, `src/utils.py`. Shared dataset loaders and helpers.
- `configs`. Experiment config files (one `ExperimentConfig` per file).
- `analysis`. Standalone scripts that read run outputs; never import the engine.
- `datasets`. Prompt sets (BBQ, CrowS-Pairs, homemade, etc.).
- `tests`. Phase verification (`python3 tests/test_phase0.py` …).
- `experiments`. Frozen historical record — legacy notebooks and past logs.
- `docs`. Design docs for the pipeline architecture.

## Setup

- `conda create -n algo-neutrality python=3.12`
- `conda activate algo-neutrality`
- `pip install -e .`
- Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (used by the judge).

## Running an experiment

```
python -m src.bias_steer run configs/example_bbq.py
```

Outputs (per-run manifest, `results.csv`, plaintext logs, steering vector) are
written under `runs/<run_id>/`.
