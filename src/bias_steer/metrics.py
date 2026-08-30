"""Results -> tidy CSV + aggregate metrics (arch roadmap §7.1).

Running produces tidy `Result` rows; metrics are *derived* from them, not the only
thing saved. Ports the notebook's `GeneralResults` (per-condition verdict counts)
and `TestResults` (did steering move each example the right way?).
"""

import csv
import json
from collections import Counter

from .schema import INITIAL, STEERED_POS, STEERED_NEG

# Columns of a run's results.csv — one row per (example, condition).
RESULT_COLUMNS = [
    "run_id", "model", "dataset", "condition", "coeff", "example_id", "verdict", "category",
]

# Columns of a run's examples.csv — one row per Example (the frozen sampled subset
# this run used). Parent table to results.csv's child; join on `example_id`.
EXAMPLE_COLUMNS = ["example_id", "dataset", "prompt", "category", "metadata_json"]


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


def write_csv(path, rows, columns) -> None:
    """Write tidy `rows` to `path` with exactly `columns` as the header. Extra keys
    in a row are dropped (`extrasaction="ignore"`), so callers can pass richer dicts."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_examples_csv(path, examples, *, dataset) -> None:
    """Snapshot the frozen sampled subset a run used — one row per `Example`, keyed
    by `example_id`. Freezes the ground truth against positional-id drift and makes a
    run folder self-contained (prompts recoverable without replaying `sample(seed)`).

    `metadata` is JSON-encoded into one column so nested fields (e.g. BBQ's `answers`)
    survive losslessly; `category` is also lifted to its own column to match
    results.csv and keep groupbys cheap. Delegates to `write_csv`."""
    rows = [
        {
            "example_id": ex.id, "dataset": dataset, "prompt": ex.prompt,
            "category": ex.metadata.get("category"),
            "metadata_json": json.dumps(ex.metadata),
        }
        for ex in examples
    ]
    write_csv(path, rows, EXAMPLE_COLUMNS)

def write_csv(path, rows) -> None:
    write_rows(path, rows, RESULT_COLUMNS)


# Columns of a refusal-repro run's results.csv (arXiv:2406.11717).
REFUSAL_RESULT_COLUMNS = [
    "run_id", "model", "harm", "condition", "coeff", "example_id", "category", "verdict",
]


def refusal_rates(results, *, refusal_label="refusal") -> dict:
    """Per-condition refusal stats from judged `Result`s.

    Returns `{condition: {n, refused, refusal_rate, success_rate}}` where
    `success_rate = 1 - refusal_rate` is the paper's
    `substring_matching_success_rate` (fraction NOT refused)."""
    out = {}
    for cond, verds in condition_verdict_counts(results).items():
        n = sum(verds.values())
        refused = verds.get(refusal_label, 0)
        rate = refused / n if n else 0.0
        out[cond] = {"n": n, "refused": refused,
                     "refusal_rate": rate, "success_rate": 1.0 - rate}
    return out


def render_refusal_summary(*, run_id, label, model, git, direction, coeff, rates) -> str:
    """Human-readable summary.md for a refusal-repro run."""
    def line(cond):
        r = rates.get(cond)
        if not r:
            return f"- **{cond}**: (no data)"
        return (f"- **{cond}**: refusal {r['refused']}/{r['n']} = "
                f"{r['refusal_rate']:.3f}  (success {r['success_rate']:.3f})")

    order = ["harmful/baseline", "harmful/ablation", "harmful/actadd",
             "harmless/baseline", "harmless/actadd"]
    shown = [c for c in order if c in rates] + [c for c in rates if c not in order]
    body = "\n".join(line(c) for c in shown)
    return (
        f"# {label} — {model} (refusal-direction repro)\n\n"
        f"- run_id: `{run_id}`\n"
        f"- direction: layer={direction['layer']}, pos={direction['pos']}, "
        f"‖r‖={direction['norm']:.3f}  |  act-add coeff magnitude={coeff}\n"
        f"- git: `{git[0]}`{' (dirty)' if git[1] else ''}\n\n"
        f"## Refusal rate by condition\n{body}\n\n"
        f"_Interpretation: ablation should DROP harmful refusal; act-add(+) should "
        f"RAISE harmless refusal (arXiv:2406.11717)._\n"
    )


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
