"""Compare our refusal-repro results against the paper's committed numbers
(arXiv:2406.11717).

The paper ships, per model, the substring-match success rate for each of its five
arms (in `third_party/refusal_direction/runs/<model>/completions/*_evaluations.json`).
`paper_rates` reads those as the ground truth; `compare_rates` diffs our run's
per-arm refusal rates against them; `render_comparison` renders the table that is
the headline Chunk-5 deliverable.

Refusal rate = 1 − substring_matching_success_rate (fraction that refused). We
compare refusal rates so the sign is intuitive: ablation should LOWER harmful
refusal, act-add(+) should RAISE harmless refusal.

Pure-stdlib (json) — no torch, no model.
"""

import json

from . import refusal

# Our arm name -> the paper's committed evaluations file for that arm.
ARM_TO_EVAL_FILE = {
    "harmful/baseline":  "jailbreakbench_baseline_evaluations.json",
    "harmful/ablation":  "jailbreakbench_ablation_evaluations.json",
    "harmful/actadd":    "jailbreakbench_actadd_evaluations.json",
    "harmless/baseline": "harmless_baseline_evaluations.json",
    "harmless/actadd":   "harmless_actadd_evaluations.json",
}


def paper_rates(model: str) -> dict:
    """The paper's published per-arm rates for `model` (catalog key or run-dir).

    Returns `{condition: {success_rate, refusal_rate, n}}`, skipping any arm whose
    evaluations file has not been fetched. Empty if none are present."""
    run_dir, _ = refusal._resolve(model)
    comp_dir = refusal.artifact_dir(run_dir) / "completions"
    out = {}
    for cond, fname in ARM_TO_EVAL_FILE.items():
        path = comp_dir / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        success = float(d["substring_matching_success_rate"])
        out[cond] = {"success_rate": success, "refusal_rate": 1.0 - success,
                     "n": len(d["completions"])}
    return out


def compare_rates(ours: dict, theirs: dict, *, tol: float = 0.05) -> list[dict]:
    """Diff our per-arm refusal rates against the paper's.

    `ours` is `metrics.refusal_rates(...)` output ({cond: {refusal_rate, ...}});
    `theirs` is `paper_rates(...)`. One row per arm present in BOTH, in the
    canonical arm order. `within_tol` flags |delta| <= tol."""
    rows = []
    for cond in ARM_TO_EVAL_FILE:
        if cond not in ours or cond not in theirs:
            continue
        o = ours[cond]["refusal_rate"]
        t = theirs[cond]["refusal_rate"]
        rows.append({
            "condition": cond,
            "ours_refusal": o,
            "theirs_refusal": t,
            "delta": o - t,
            "within_tol": abs(o - t) <= tol,
        })
    return rows


def render_comparison(rows: list[dict], *, tol: float = 0.05) -> str:
    """Markdown table of our-vs-paper refusal rates."""
    if not rows:
        return "## vs. paper\n\n_(no committed rates fetched for this model)_\n"
    head = ("| arm | ours | paper | Δ | within ±{:.2f} |\n"
            "|---|---|---|---|---|\n").format(tol)
    body = "\n".join(
        f"| {r['condition']} | {r['ours_refusal']:.3f} | {r['theirs_refusal']:.3f} "
        f"| {r['delta']:+.3f} | {'✓' if r['within_tol'] else '✗'} |"
        for r in rows
    )
    n_ok = sum(r["within_tol"] for r in rows)
    return f"## vs. paper (refusal rate)\n\n{head}{body}\n\n_{n_ok}/{len(rows)} arms within ±{tol:.2f}._\n"
