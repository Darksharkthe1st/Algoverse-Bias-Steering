"""Exp-3 — the coherence gate: is a steered completion still real text?

`docs/HANDOFF_lane_a.md` §Exp-3. Closes the reviewer objection "your 95% opinion
rate is just broken text" (the c40-INVALID risk). Scores every completion of a run
and writes `runs/<id>/coherence.csv`.

**Independent of the judge, by construction.** Nothing here reads a judge verdict.
The judge's own `incoherent`/`nonsense` bucket cannot be the gate: it is produced
by the same model whose numbers are under suspicion, so using it would let one
judgement both make the claim and vouch for it. These are measures over the TEXT —
lexical diversity, repetition, and perplexity under a *different*, unsteered model.

    # score a run against its own unsteered arm
    python scripts/coherence_gate.py runs/<id>

    # score a run that has no unsteered arm against another run's baseline
    python scripts/coherence_gate.py runs/<dose_run> --baseline-run runs/<anchor>

A completion FAILS the gate if any of:

  * `unclosed_think`        — it opened a `<think>` trace and never closed it, so
                              the generation stopped inside the reasoning and there
                              is NO answer under it
  * `distinct_3 < 0.5`      — trigram diversity collapsed (looping)
  * `max_repeat_run >= 4`   — four or more identical consecutive lines
  * `ppl > P95(baseline)`   — less fluent than the 95th percentile of the
                              UNSTEERED arm, i.e. outside the range the model
                              produces when nobody touches it

The `unclosed_think` leg is the one that fires on the c20/c30/c40 dose ladder
(`runs/..._adaptive-add-linear-c*`): those runs used `max_tokens=128` with
`enable_thinking` unset — so ON for Qwen3-8B — and `strip_reasoning` off, which is
exactly the configuration HANDOFF §0.3 forbids. It is judge-independent and needs no
reference LM, so it is computed and reported even when perplexity is not.

The perplexity threshold is *relative to the unsteered baseline of the same run*,
never an absolute number: absolute perplexity depends on the prompt distribution
(IssueBench essay requests differ from forced-choice questions), so a fixed cutoff
would measure the dataset, not the damage.

**The reportable dose** is the largest dose whose coherence-pass rate is at least
the unsteered baseline's; the per-arm table printed at the end marks each arm
PASS or BELOW-BASELINE against its run's unsteered arm. A dose whose opinion rate
rises while its pass rate falls has not steered the behaviour; it has degraded the
generator.

## What is deliberately NOT a failure

Hitting `max_tokens` mid-sentence in the ANSWER. IssueBench prompts ask for essays,
so every arm truncates at the cap; it is universal, not a steering effect, and
counting it would fail the baseline too. The cap is recorded (`hit_token_cap`) for
transparency and excluded from the verdict.

That benign case is why truncation is gated via `unclosed_think` rather than by
length: a completion cut off mid-answer still has an answer to judge, while one cut
off mid-`<think>` has none. The two look identical to a length check and could not
be more different downstream.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.bias_steer import runlogs  # noqa: E402

# Thresholds from the handoff. Named, not inlined, so a reader can see the gate's
# whole definition in one place and a change is a visible diff.
DISTINCT_N = 3
DISTINCT_MIN = 0.5
MAX_REPEAT_RUN_FAIL = 4
PPL_BASELINE_PCTL = 95
# The reference LM: a small model that is NOT the model under test and is never
# steered, so its perplexity is an outside opinion about the text.
DEFAULT_REF_MODEL = "Qwen/Qwen2.5-1.5B"
UNSTEERED_COND = "initial"
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def distinct_n(text: str, n: int = DISTINCT_N) -> float:
    """Share of word n-grams that are unique. 1.0 = no repetition at all.

    Short texts (< n words) have no n-grams; they are scored 1.0 rather than 0.0 so
    a terse answer is not called a repetition loop. Terseness is a different
    failure and is not this gate's business.
    """
    toks = text.split()
    if len(toks) < n:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def max_repeat_run(text: str) -> int:
    """Longest run of an immediately-repeated non-blank line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    best = cur = 1 if lines else 0
    for i in range(1, len(lines)):
        cur = cur + 1 if lines[i] == lines[i - 1] else 1
        best = max(best, cur)
    return best


def unclosed_think(text: str) -> bool:
    """True if a reasoning trace was opened and never closed.

    A hybrid-reasoning model that runs out of budget mid-`<think>` emits no answer at
    all, so any stance read out of such a completion is read out of the reasoning —
    which is not the behaviour under study. This is the single check that separates
    "the model hedged" from "the run was misconfigured", and it costs nothing.
    """
    return _THINK_OPEN in text and _THINK_CLOSE not in text


def top_trigram_share(text: str) -> float:
    """Share of the most common trigram — diagnostic only, not part of the gate.

    Recorded because it is what `logs._degenerate_flag` fires on, so a reader can
    see why the cheap in-log hint and this gate disagree on a given completion.
    """
    toks = text.split()
    if len(toks) < 3:
        return 0.0
    grams = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return Counter(grams).most_common(1)[0][1] / len(grams)


class RefLM:
    """Perplexity under a small unsteered reference model (lazy torch import)."""

    def __init__(self, model_id: str, device: str | None = None, max_length: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device)
        self.model.eval()

    def ppl(self, text: str) -> float:
        import torch

        if not text.strip():
            return float("nan")
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_length).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return float("nan")
        with torch.no_grad():
            loss = self.model(ids, labels=ids).loss
        return math.exp(loss.item())

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text).input_ids)


class _NoPPL:
    """Stand-in for the reference LM when --no-ppl is passed.

    Perplexity is NaN (which cannot fail the fluency leg) and token counts fall back
    to whitespace words, so `hit_token_cap` becomes an underestimate — flagged in the
    output rather than silently reported as exact.
    """

    model_id = "(none: --no-ppl)"
    device = "n/a"

    def ppl(self, text: str) -> float:
        return float("nan")

    def n_tokens(self, text: str) -> int:
        return len(text.split())


def percentile(values, pctl: float) -> float:
    """Nearest-rank percentile of the finite values (no numpy dependency)."""
    vals = sorted(v for v in values if v == v and v != float("inf"))
    if not vals:
        return float("nan")
    k = max(0, min(len(vals) - 1, int(round(pctl / 100 * (len(vals) - 1)))))
    return vals[k]


def score_run(run_dir, ref: RefLM, max_tokens: int | None) -> list[dict]:
    """Per-completion coherence measures for one run (no pass/fail yet).

    Pass/fail needs a baseline perplexity threshold, which may come from another
    run, so it is applied by `apply_gate` after every run is scored.
    """
    records = runlogs.read_run(run_dir)
    rows = []
    for r in records:
        n_tok = ref.n_tokens(r.response)
        rows.append({
            "run": Path(run_dir).name,
            "example_id": r.example_id,
            "condition": r.condition,
            "n_words": len(r.response.split()),
            "n_ref_tokens": n_tok,
            f"distinct_{DISTINCT_N}": round(distinct_n(r.response), 4),
            "max_repeat_run": max_repeat_run(r.response),
            "top_trigram_share": round(top_trigram_share(r.response), 4),
            "unclosed_think": int(unclosed_think(r.response)),
            "ppl": round(ref.ppl(r.response), 3),
            # Approximate: the cap is counted in the MODEL's tokenizer, this is the
            # ref model's, so treat it as a flag rather than an exact equality.
            "hit_token_cap": int(max_tokens is not None and n_tok >= 0.95 * max_tokens),
            "logged_degenerate_flag": int(r.degenerate_flagged),
        })
    return rows


def apply_gate(rows, ppl_threshold: float) -> None:
    """Add `ppl_fail` / `distinct_fail` / `repeat_fail` / `coherent` in place."""
    for row in rows:
        t_fail = bool(row.get("unclosed_think", 0))
        d_fail = row[f"distinct_{DISTINCT_N}"] < DISTINCT_MIN
        r_fail = row["max_repeat_run"] >= MAX_REPEAT_RUN_FAIL
        p = row["ppl"]
        # NaN perplexity (empty/1-token text) cannot fail the fluency test; it is
        # caught by the diversity test or it is simply a very short answer.
        p_fail = bool(p == p and ppl_threshold == ppl_threshold and p > ppl_threshold)
        row.update({
            "think_fail": int(t_fail),
            "distinct_fail": int(d_fail), "repeat_fail": int(r_fail),
            "ppl_fail": int(p_fail),
            "coherent": int(not (t_fail or d_fail or r_fail or p_fail)),
        })


def pass_rates(rows) -> dict:
    """Coherence-pass rate per (run, condition)."""
    out: dict = {}
    for row in rows:
        key = (row["run"], row["condition"])
        agg = out.setdefault(key, {"n": 0, "coherent": 0})
        agg["n"] += 1
        agg["coherent"] += row["coherent"]
    for agg in out.values():
        agg["pass_rate"] = agg["coherent"] / agg["n"] if agg["n"] else float("nan")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run dirs to score")
    ap.add_argument("--ref-model", default=DEFAULT_REF_MODEL,
                    help=f"reference LM for perplexity (default {DEFAULT_REF_MODEL})")
    ap.add_argument("--baseline-run", type=Path, default=None,
                    help="run whose unsteered arm sets the perplexity threshold "
                         "(default: each run's own unsteered arm)")
    ap.add_argument("--baseline-cond", default=UNSTEERED_COND,
                    help=f"the unsteered arm's condition name (default {UNSTEERED_COND})")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="generation cap, for the hit_token_cap flag (default: read "
                         "from each run's manifest)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--no-ppl", action="store_true",
                    help="skip the perplexity leg (no reference LM loaded). The "
                         "text-only legs still run, which is enough when a run fails "
                         "on unclosed_think.")
    ap.add_argument("--out-name", default="coherence.csv",
                    help="filename written inside each run dir")
    a = ap.parse_args(argv)

    ref = _NoPPL() if a.no_ppl else RefLM(a.ref_model, device=a.device)
    print(f"reference LM: {ref.model_id}" + ("" if a.no_ppl else f" on {ref.device}"))

    scored: dict[Path, list[dict]] = {}
    for run in a.runs:
        run = run if run.is_absolute() else _REPO / run
        max_tokens = a.max_tokens
        if max_tokens is None:
            mf = run / "manifest.json"
            if mf.is_file():
                max_tokens = json.loads(mf.read_text())["config"].get("max_tokens")
        rows = score_run(run, ref, max_tokens)
        scored[run] = rows
        print(f"scored {len(rows)} completions in {run.name}")

    # The perplexity threshold: P95 of the unsteered arm. One shared threshold when
    # --baseline-run names an anchor (so doses are comparable to ONE reference), else
    # each run against its own unsteered arm.
    shared_threshold = None
    if a.baseline_run:
        b = a.baseline_run if a.baseline_run.is_absolute() else _REPO / a.baseline_run
        rows = scored.get(b)
        if rows is None:
            rows = score_run(b, ref, a.max_tokens)
        base = [r["ppl"] for r in rows if r["condition"] == a.baseline_cond]
        if not base:
            raise SystemExit(
                f"--baseline-run {b.name} has no {a.baseline_cond!r} arm to calibrate on")
        shared_threshold = percentile(base, PPL_BASELINE_PCTL)
        print(f"shared ppl P{PPL_BASELINE_PCTL} from {b.name}/{a.baseline_cond}: "
              f"{shared_threshold:.2f} (n={len(base)})")

    summary_lines = []
    for run, rows in scored.items():
        threshold = shared_threshold
        if threshold is None:
            base = [r["ppl"] for r in rows if r["condition"] == a.baseline_cond]
            if not base and not a.no_ppl:
                raise SystemExit(
                    f"{run.name} has no {a.baseline_cond!r} arm, so its perplexity "
                    f"threshold is undefined — pass --baseline-run to calibrate it "
                    f"against another run's unsteered arm."
                )
            threshold = percentile(base, PPL_BASELINE_PCTL)
            print(f"{run.name}: ppl P{PPL_BASELINE_PCTL} = {threshold:.2f} "
                  f"(n={len(base)} {a.baseline_cond})")
        apply_gate(rows, threshold)

        out = run / a.out_name
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")

        rates = pass_rates(rows)
        base_rate = rates.get((run.name, a.baseline_cond), {}).get("pass_rate")
        for (rname, cond), agg in sorted(rates.items()):
            mark = ""
            if base_rate is not None and cond != a.baseline_cond:
                mark = "  PASS" if agg["pass_rate"] >= base_rate else "  BELOW-BASELINE"
            line = (f"{rname}  {cond:12s} coherence-pass "
                    f"{agg['coherent']}/{agg['n']} = {agg['pass_rate']:.3f}{mark}")
            summary_lines.append(line)
            print(line)

    print("\n--- gate: distinct_%d < %.2f OR max_repeat_run >= %d OR ppl > P%d(%s)"
          % (DISTINCT_N, DISTINCT_MIN, MAX_REPEAT_RUN_FAIL, PPL_BASELINE_PCTL,
             a.baseline_cond))
    print("--- reportable dose = the largest dose whose pass rate is >= the "
          "unsteered baseline's")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
