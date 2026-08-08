"""Cross-run comparison from CSV outputs (arch roadmap §7.1).

Reads `runs/index.csv` (one row per run) and per-run `results.csv` (tidy: one row
per example×condition) into pandas, so aggregates are one `groupby`/pivot away —
the payoff of the tidy long format. Never imports the `bias_steer` engine, so
re-analysis can never trigger a re-run.

    python -m analysis.compare runs/            # print the cross-run table

The helpers below are conveniences; with the loaders returning DataFrames you can
also just do your own pandas directly (e.g. `df.groupby("category").verdict.value_counts()`).
"""

import argparse
from pathlib import Path

import pandas as pd

# The condition values written into results.csv. Kept local (a copy of the data
# contract) so this module stays independent of the engine.
INITIAL = "initial"
STEERED_POS = "steered_pos"
STEERED_NEG = "steered_neg"


def load_index(runs_dir="runs") -> pd.DataFrame:
    """`runs/index.csv` as a DataFrame (empty DataFrame if absent)."""
    path = Path(runs_dir) / "index.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_results(run_dir) -> pd.DataFrame:
    """One run's tidy `results.csv`."""
    return pd.read_csv(Path(run_dir) / "results.csv")


def load_run_results(runs_dir, run_id) -> pd.DataFrame:
    """Tidy rows for a run identified by its id under `runs_dir`."""
    return load_results(Path(runs_dir) / str(run_id))


def load_all_results(runs_dir="runs", run_ids=None) -> pd.DataFrame:
    """Concatenate several runs' tidy rows into one DataFrame (each row already
    carries its `run_id`), so cross-run analysis is a single `groupby`."""
    index = load_index(runs_dir)
    if run_ids is None:
        run_ids = [] if index.empty else index["run_id"].tolist()
    frames = [load_run_results(runs_dir, rid) for rid in run_ids]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def verdict_rates(results: pd.DataFrame, by=None) -> pd.DataFrame:
    """Proportion of each verdict within each condition.

    Optionally split further by column(s) — e.g. ``by="category"`` for per-category
    rates or ``by="run_id"`` across runs. Returns a tidy frame with `n` and `rate`.
    """
    if results.empty:
        return results
    extra = [by] if isinstance(by, str) else list(by or [])
    keys = ["condition"] + extra
    counts = results.groupby(keys + ["verdict"]).size().rename("n").reset_index()
    counts["rate"] = counts["n"] / counts.groupby(keys)["n"].transform("sum")
    return counts


def compare(runs_dir="runs", columns=None) -> pd.DataFrame:
    """The cross-run table (index.csv), optionally projected to `columns`."""
    index = load_index(runs_dir)
    if index.empty or columns is None:
        return index
    return index[[c for c in columns if c in index.columns]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="analysis.compare",
                                     description="Compare runs from index.csv.")
    parser.add_argument("runs_dir", nargs="?", default="runs")
    args = parser.parse_args(argv)
    index = load_index(args.runs_dir)
    print("(no runs)" if index.empty else index.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
