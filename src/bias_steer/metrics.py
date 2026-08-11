"""Results -> tidy CSV + aggregate metrics (arch roadmap §7.1).

Running produces tidy `Result` rows; metrics are *derived* from them, not the only
thing saved. Ports the notebook's `GeneralResults` (per-condition verdict counts)
and `TestResults` (did steering move each example the right way?).
"""

import csv
from collections import Counter

from .schema import INITIAL, STEERED_POS, STEERED_NEG

# Columns of a run's results.csv — one row per (example, condition).
RESULT_COLUMNS = [
    "run_id", "model", "dataset", "condition", "coeff", "example_id", "verdict", "category",
]


def tidy_rows(results, *, run_id, model, dataset, opin_coeff, neut_coeff) -> list[dict]:
    """Flatten `Result`s into tidy rows. `coeff` records the signed strength that
    produced each condition (initial=0, +opinion, -neutral)."""
    coeff_for = {INITIAL: 0, STEERED_POS: opin_coeff, STEERED_NEG: -neut_coeff}
    return [
        {
            "run_id": run_id, "model": model, "dataset": dataset,
            "condition": r.condition, "coeff": coeff_for.get(r.condition, ""),
            "example_id": r.example_id, "verdict": r.verdict,
            "category": r.metadata.get("category"),
        }
        for r in results
    ]


def write_csv(path, rows) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _by_example(results) -> dict:
    """example_id -> {condition: verdict}."""
    out: dict = {}
    for r in results:
        out.setdefault(r.example_id, {})[r.condition] = r.verdict
    return out


def condition_verdict_counts(results) -> dict:
    """{condition: {verdict: count}} — the notebook's GeneralResults, tidy form."""
    out: dict = {}
    for r in results:
        out.setdefault(r.condition, Counter())[r.verdict] += 1
    return {cond: dict(counter) for cond, counter in out.items()}


def steering_quality(results, *, pos_label, neg_label, nonsense_label="nonsense") -> dict:
    """Did steering move each example the right way? Ports the notebook's TestResults.

    - opinion: comparing INITIAL vs STEERED_POS against `pos_label`
    - neutral: comparing INITIAL vs STEERED_NEG against `neg_label`
    - nonsense: whether steering pushed a coherent answer into `nonsense_label` (bad)
      or rescued a nonsense one (good)
    """
    by_ex = _by_example(results)
    opinion = {"good": 0, "bad": 0, "same_good": 0, "same_bad": 0}
    neutral = {"good": 0, "bad": 0, "same_good": 0, "same_bad": 0}
    nonsense = {"very_good": 0, "good": 0, "same": 0, "bad": 0, "very_bad": 0}

    for cond in by_ex.values():
        init, pos, neg = cond.get(INITIAL), cond.get(STEERED_POS), cond.get(STEERED_NEG)

        # opinion: want INITIAL -> STEERED_POS to reach pos_label
        if init != pos_label and pos == pos_label:
            opinion["good"] += 1
        elif init == pos_label and pos != pos_label:
            opinion["bad"] += 1
        elif init == pos_label and pos == pos_label:
            opinion["same_good"] += 1
        else:
            opinion["same_bad"] += 1

        # neutral: want INITIAL -> STEERED_NEG to reach neg_label
        if init != neg_label and neg == neg_label:
            neutral["good"] += 1
        elif init == neg_label and neg != neg_label:
            neutral["bad"] += 1
        elif init == neg_label and neg == neg_label:
            neutral["same_good"] += 1
        else:
            neutral["same_bad"] += 1

        # nonsense: steering shouldn't turn coherent answers into nonsense
        n = nonsense_label
        steered_nonsense = [pos == n, neg == n]
        if init == n and not any(steered_nonsense):
            nonsense["very_good"] += 1
        elif init == n and not all(steered_nonsense):
            nonsense["good"] += 1
        elif init != n and all(steered_nonsense):
            nonsense["very_bad"] += 1
        elif init != n and any(steered_nonsense):
            nonsense["bad"] += 1
        else:
            nonsense["same"] += 1

    return {"opinion": opinion, "neutral": neutral, "nonsense": nonsense}


def render_summary(*, run_id, label, model, dataset, coeffs, git, n_train, n_test,
                   counts, quality) -> str:
    """Human-readable per-run summary.md (committed)."""
    def block(title, d):
        return f"### {title}\n" + "\n".join(f"- {k}: {v}" for k, v in d.items())

    counts_md = "\n".join(
        f"- **{cond}**: " + ", ".join(f"{v}×{k}" for k, v in sorted(verds.items()))
        for cond, verds in counts.items()
    )
    return (
        f"# {label} — {model}\n\n"
        f"- run_id: `{run_id}`\n"
        f"- dataset: `{dataset}`  |  method coeffs: opinion={coeffs.opinion}, neutral={coeffs.neutral}\n"
        f"- git: `{git[0]}`{' (dirty)' if git[1] else ''}\n"
        f"- train examples: {n_train}  |  test examples: {n_test}\n\n"
        f"## Verdict counts by condition\n{counts_md}\n\n"
        f"## Steering quality\n"
        f"{block('opinion (toward pos)', quality['opinion'])}\n\n"
        f"{block('neutral (toward neg)', quality['neutral'])}\n\n"
        f"{block('nonsense', quality['nonsense'])}\n"
    )
