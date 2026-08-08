"""Cross-run comparison from CSV outputs (arch roadmap §7.1).

Reads `runs/index.csv` (one row per run) and per-run `results.csv` (tidy: one row
per example×condition), and computes aggregates *after the fact* — the payoff of
the tidy long format. Stdlib only (no pandas, no torch); never imports the engine,
so re-analysis can never trigger a re-run.

    python -m analysis.compare runs/            # print the cross-run table
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

# The condition values written into results.csv. Kept local (a copy of the data
# contract) so this module stays independent of the engine.
INITIAL = "initial"
STEERED_POS = "steered_pos"
STEERED_NEG = "steered_neg"


def load_index(runs_dir="runs") -> list[dict]:
    """Rows of `runs/index.csv` (empty list if absent)."""
    path = Path(runs_dir) / "index.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_results(run_dir) -> list[dict]:
    """Tidy rows of one run's `results.csv`."""
    with (Path(run_dir) / "results.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def load_run_results(runs_dir, run_id) -> list[dict]:
    """Tidy rows for a run identified by its id under `runs_dir`."""
    return load_results(Path(runs_dir) / run_id)


def verdict_counts(rows, condition=None, group_by=None):
    """Count verdicts over tidy rows. Optionally restrict to one `condition` and/or
    split by a column (e.g. ``group_by="category"`` for per-category rates)."""
    def match(r):
        return condition is None or r["condition"] == condition

    if group_by:
        grouped = defaultdict(Counter)
        for r in rows:
            if match(r):
                grouped[r.get(group_by)][r["verdict"]] += 1
        return {k: dict(v) for k, v in grouped.items()}

    counter = Counter(r["verdict"] for r in rows if match(r))
    return dict(counter)


def rate(rows, condition, verdict) -> float:
    """Fraction of `condition` rows whose verdict is `verdict` (0.0 if none)."""
    total = sum(1 for r in rows if r["condition"] == condition)
    if not total:
        return 0.0
    hits = sum(1 for r in rows if r["condition"] == condition and r["verdict"] == verdict)
    return hits / total


DEFAULT_COMPARE_COLUMNS = [
    "run_id", "model", "dataset", "opin_coeff", "neut_coeff",
    "n_test", "opin_good", "neut_good", "status",
]


def compare(runs_dir="runs", columns=None) -> list[dict]:
    """A cross-run table (one row per run) projected to `columns`, from index.csv."""
    columns = columns or DEFAULT_COMPARE_COLUMNS
    return [{c: row.get(c, "") for c in columns} for row in load_index(runs_dir)]


def format_table(rows, columns=None) -> str:
    """Render a list of dict rows as a fixed-width text table."""
    if not rows:
        return "(no rows)"
    columns = columns or list(rows[0].keys())
    widths = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in rows]) for c in columns}

    def line(values):
        return "  ".join(str(v).ljust(widths[c]) for c, v in zip(columns, values))

    out = [line(columns), "  ".join("-" * widths[c] for c in columns)]
    out += [line([r.get(c, "") for c in columns]) for r in rows]
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="analysis.compare",
                                     description="Compare runs from index.csv.")
    parser.add_argument("runs_dir", nargs="?", default="runs")
    args = parser.parse_args(argv)
    print(format_table(compare(args.runs_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
