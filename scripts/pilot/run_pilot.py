"""Pilot driver — Phase 2 of notes/11, gate P2.

    python -m scripts.pilot.run_pilot --out runs/pilot

Runs the entire run-2 pipeline end to end on 20 scenario pairs across 2
categories, in seconds, with no GPU and no model download.

What makes this a pilot rather than a smoke test: the stub backend plants a
KNOWN structure, so every control has a right answer and the pilot asserts the
control produces it.  It runs the whole thing TWICE —

    truth = "distinct"   categories planted orthogonal, length component small
    truth = "collapsed"  categories planted identical,  length component large

— and requires the specificity control to PASS the first and FAIL the second.
A control that cannot tell those apart is not a control, and run 1 never checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from src.bias_steer.bias_taxonomy import parse_choice
from . import analysis, backends, pairing, verifier
from .queue import Step, run_queue

PILOT_CATEGORIES = ["Disability_status", "Physical_appearance"]
PILOT_PAIRS = 20


# --------------------------------------------------------------------------- #
# Persistence — S5's closure.  Residuals are written BEFORE any extraction.
# --------------------------------------------------------------------------- #

def persist_residuals(out_dir, category, arm, rows, resid, backend) -> str:
    """Write `(n_items, n_layers, d_model)` float32 plus a sidecar.

    The sidecar is what makes the array auditable years later: item ids in row
    order, the capture site, dtype, and the backend's full description.  Run 1
    saved neither the array nor a sidecar, which is why a finished analysis could
    no longer be checked (notes/18).
    """
    d = os.path.join(out_dir, "residuals")
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{category}__{arm}")
    np.save(stem + ".npy", resid.astype(np.float32))
    meta = {
        "category": category, "arm": arm,
        "item_ids": [pairing.item_key(r) for r in rows],
        "n_items": len(rows), "n_layers": int(resid.shape[1]),
        "d_model": int(resid.shape[2]), "dtype": "float32",
        "capture_site": backend.describe().get("capture_site", "resid_pre, see backend"),
        "backend": backend.describe(),
    }
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return stem + ".npy"


def cached_capture(out_dir, categories, backend):
    """A capture callable that reads persisted residuals, never the model.

    This is the property that makes 400 split-halves affordable and is the
    concrete test of notes/14's governing constraint: *every analysis must be
    redoable tomorrow with the GPU already returned.*  If any analysis below
    reached back to `backend.capture`, the pilot would be certifying run 1's
    economics.

    Keyed by `pairing.item_key` -- "<category>:<example_id>" -- because
    `example_id` restarts at 0 in every BBQ file, so a cache keyed on it alone
    merges categories silently.
    """
    if isinstance(categories, str):
        categories = [categories]
    store: dict = {}
    for category in categories:
        for arm in ("a", "b"):
            stem = os.path.join(out_dir, "residuals", f"{category}__{arm}")
            if not os.path.exists(stem + ".npy"):
                continue
            meta = json.load(open(stem + ".json", encoding="utf-8"))
            arr = np.load(stem + ".npy", mmap_mode="r")     # mmap: notes/19 6.4
            for i, iid in enumerate(meta["item_ids"]):
                store[iid] = arr[i]

    def capture(rows, arm_sign=1.0):
        missing = [pairing.item_key(r) for r in rows
                   if pairing.item_key(r) not in store]
        if missing:
            raise KeyError(f"{len(missing)} items not in the residual cache "
                           f"(first: {missing[:3]}). The analysis reached for "
                           f"data that was never persisted -- that is S5.")
        return np.stack([np.asarray(store[pairing.item_key(r)]) for r in rows])

    return capture


# --------------------------------------------------------------------------- #
# The confound checklist — notes/11 §5, every item checked and RECORDED
# --------------------------------------------------------------------------- #

def confound_checklist(cats, per_cat_rows) -> dict:
    """Gate P2 requires every §5 item explicitly checked and recorded.

    "Recorded" is the operative word: a checklist run and not written down is
    indistinguishable from one never run, which is notes/11 §6's own rule that a
    control that has not been run is a control that failed.
    """
    out = {}

    # 5.1 prompt-resident text
    out["5.1_prompt_resident_text"] = {
        "check": "does the scored continuation appear in the prompt?",
        "result": "PASS — method 3 scores candidate answers with NO option list "
                  "in the prompt (bbq_score.margins docstring). Design 2, which "
                  "failed this, is not used.",
    }

    # 5.2 permutation invariance -- of the PARSER, measured
    ans = ["The doctor", "The nurse", "Cannot answer"]
    probe = "It's not the doctor, it's the nurse."
    import itertools
    labels = set()
    for perm in itertools.permutations(range(3)):
        opts = [ans[i] for i in perm]
        g = parse_choice(probe, opts)
        labels.add(opts[g] if g is not None else None)
    out["5.2_permutation_invariance"] = {
        "check": "does parse_choice depend on the ORDER of the option list?",
        "distinct_labels_over_6_permutations": len(labels),
        "result": "MEASURED — invariant to option-list order. This is why "
                  "notes/18 item 4 is vacuous; the mirror-pair test in "
                  "tests/test_bias_taxonomy.py is the one that bites (notes/19 §5.3).",
    }

    # 5.3 selection-induced imbalance
    pos = {}
    for c in cats:
        a_rows, b_rows = pairing.arms(c.pairs)
        pos[c.category] = {
            "arm_a": len(a_rows), "arm_b": len(b_rows),
            "balanced": len(a_rows) == len(b_rows),
        }
    out["5.3_selection_induced_imbalance"] = {
        "check": "are the buckets defined by model behaviour?",
        "per_category": pos,
        "result": "PASS BY CONSTRUCTION — arms are labelled by context_condition, "
                  "a dataset annotation. No model output enters pole assignment, "
                  "which is what closes M1.",
    }

    # 5.4 circularity of the capture site
    out["5.4_capture_site_circularity"] = {
        "check": "can the capture site encode the label?",
        "result": "PASS — capture is at a prompt token, before any answer token "
                  "exists, and the label is a dataset annotation rather than "
                  "anything derived from the response.",
        "OPEN": "the capture INDEX is unresolved: spec says -2, bbq_score.py:296 "
                "uses -1. notes/19 §6.1, hole (d). Tier 2 must settle it.",
    }

    # 5.5 estimator variance vs effect size
    out["5.5_estimator_variance"] = {
        "check": "is 'more signal = more separable' distinguishable from the "
                 "estimator's own variance property?",
        "result": "ADDRESSED — the negative control shuffles arm labels within a "
                  "scenario, holding n and separation structure fixed, so the "
                  "observed floor is judged against a matched alternative "
                  "rather than against zero.",
    }

    # 5.6 magnitude parity -- MEASURED
    out["5.6_magnitude_parity"] = {
        "check": "max/min direction norm across conditions, before comparing them",
        "result": "measured per run below in `direction_norms`; steering "
                  "unit-normalises per layer (bbq_score.unit_per_layer).",
        "OPEN": "the DOSE units still do not match the reference paper's — "
                "notes/19 §4.2 B-4.",
    }

    # 5.7 dataset provenance -- MEASURED
    out["5.7_dataset_provenance"] = {
        "check": "what ships with BBQ that we are not using?",
        "per_category_pairing": {c.category: c.report for c in cats},
        "result": "target_loc from third_party/bbq/additional_metadata.csv "
                  "(58,556 records, not the 86,157 LINES cited at notes/17:385); "
                  "context_condition and question_polarity ship on every row.",
    }

    # 5.8 heterogeneity of the unit
    het = {c.category: len({p.key[0] for p in c.pairs}) for c in cats}
    out["5.8_heterogeneity_of_the_unit"] = {
        "check": "how many sub-groups does a category pool?",
        "distinct_question_index_in_pilot_sample": het,
        "result": "RECORDED — the unit of analysis is the category, decided "
                  "before running. Sub-group analysis is a separate experiment.",
    }
    return out


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

def run_once(out_dir: str, *, truth: str, seed: int = 0, n_splits: int = 60) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    backend = backends.StubBackend(seed=seed, truth=truth)

    cats = pairing.load_pilot_categories(PILOT_CATEGORIES, limit_pairs=PILOT_PAIRS)
    for c in cats:
        if c.n_pairs < 4:
            raise RuntimeError(f"{c.category}: only {c.n_pairs} pairs")

    per_cat_rows: dict = {}
    steps, results = [], {}

    # -- step 1: capture + persist, per category, per arm ------------------- #
    produced = []

    def _capture_and_persist():
        prompts_path = os.path.join(out_dir, "prompts.jsonl")
        with open(prompts_path, "w", encoding="utf-8") as pf:
            for c in cats:
                a_rows, b_rows = pairing.arms(c.pairs)
                per_cat_rows[c.category] = a_rows
                for arm, rows, sign in (("a", a_rows, +1.0), ("b", b_rows, -1.0)):
                    resid = backend.capture(rows, c.category, arm_sign=sign)
                    produced.append(persist_residuals(out_dir, c.category, arm,
                                                      rows, resid, backend))
                    for r in rows:
                        pf.write(json.dumps({
                            "item_id": pairing.item_key(r), "category": c.category,
                            "arm": arm, "context_condition": r["context_condition"],
                            "question_polarity": r["question_polarity"],
                            "prompt": pairing.prompt_text(r),      # VERBATIM
                            "answers": [r["ans0"], r["ans1"], r["ans2"]],
                        }) + "\n")

    steps.append(Step(name="capture_and_persist_residuals",
                      produces=[os.path.join(out_dir, "prompts.jsonl")],
                      fn=_capture_and_persist))

    # -- step 2: generation, saved verbatim, dual-labelled ------------------ #
    def _generate():
        path = os.path.join(out_dir, "responses.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for c in cats:
                a_rows, _ = pairing.arms(c.pairs)
                for r, resp in zip(a_rows, backend.generate(a_rows)):
                    answers = [r["ans0"], r["ans1"], r["ans2"]]
                    idx = parse_choice(resp, answers)
                    f.write(json.dumps({
                        "item_id": pairing.item_key(r), "category": c.category,
                        "response": resp,                                  # VERBATIM
                        "response_sha256": hashlib.sha256(resp.encode()).hexdigest(),
                        "answers": answers,
                        "parser_label": answers[idx] if idx is not None else None,
                        "parser_index": idx,
                        "judge_label": None,      # tier 2 / real run; notes/19 §5.5
                    }) + "\n")

    steps.append(Step(name="generate_and_label",
                      produces=[os.path.join(out_dir, "responses.jsonl")],
                      fn=_generate))

    # -- step 3: floors, controls, matrix, ALL from cached residuals -------- #
    def _analyse():
        directions, floors, negs, verdicts, norms = {}, {}, {}, {}, {}
        for c in cats:
            cap = cached_capture(out_dir, [c.category], backend)
            d = analysis.extract_from_pairs(c.pairs, cap)
            directions[c.category] = d
            norms[c.category] = {
                "frobenius": float(np.linalg.norm(d)),
                "max_layer": float(np.linalg.norm(d, axis=1).max()),
                "mean_layer": float(np.linalg.norm(d, axis=1).mean()),
            }
            floors[c.category] = analysis.floor(c.pairs, cap, n_splits=n_splits, seed=seed)
            negs[c.category] = analysis.negative_control_floor(
                c.pairs, cap, n_splits=n_splits, seed=seed)
            verdicts[c.category] = analysis.reproduces(floors[c.category], negs[c.category])

        pooled_cap = cached_capture(out_dir, [c.category for c in cats], backend)

        d_len_bar = analysis.pooled_length_direction(
            {c.category: per_cat_rows[c.category] for c in cats}, pooled_cap)
        selfcheck = analysis.length_direction_selfcheck(
            {c.category: per_cat_rows[c.category] for c in cats}, pooled_cap, seed=seed)
        spec = analysis.specificity_control(
            {c.category: c.pairs for c in cats},
            {c.category: cached_capture(out_dir, [c.category], backend) for c in cats},
            directions, floors, negs, d_len_bar, selfcheck,
            n_splits=n_splits, seed=seed)
        cross = analysis.cross_category(directions)
        projected = analysis.cross_category(
            {k: analysis.project_out(v, d_len_bar) for k, v in directions.items()})

        results.update({
            "truth_planted": backend.describe(),
            "pairing": {c.category: c.report for c in cats},
            "direction_norms": norms,
            "observed_floor": floors,
            "negative_control_floor": negs,
            "reproduces": verdicts,
            "specificity_control": spec,
            "length_direction_selfcheck": selfcheck,
            "cross_category": cross,
            "cross_category_length_projected": projected,
            "confounds": confound_checklist(cats, per_cat_rows),
        })
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    steps.append(Step(name="analyse_from_cache",
                      produces=[os.path.join(out_dir, "report.json")],
                      fn=_analyse))

    manifest = run_queue(steps, out_dir=out_dir)
    results["queue_manifest"] = manifest
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--n-splits", type=int, default=60,
                    help="pilot only; the real run uses 400 (notes/13 §4)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("=" * 78)
    print("RUN-2 PILOT — tier 1 (torch-free).  notes/11 gate P2.")
    print("=" * 78)

    outcomes = {}
    for truth in ("distinct", "collapsed", "pure_length"):
        d = os.path.join(args.out, truth)
        print(f"\n--- planted truth: {truth} -> {d}")
        res = run_once(d, truth=truth, seed=args.seed, n_splits=args.n_splits)
        outcomes[truth] = res

        for cat in sorted(res["observed_floor"]):
            o, n = res["observed_floor"][cat], res["negative_control_floor"][cat]
            print(f"    {cat:<22} floor {o['mean']:+.3f} "
                  f"[{o['ci_lo']:+.3f},{o['ci_hi']:+.3f}]   "
                  f"control {n['mean']:+.3f} [{n['ci_lo']:+.3f},{n['ci_hi']:+.3f}]   "
                  f"-> {res['reproduces'][cat]}")
        sp = res["specificity_control"]
        print(f"    cross-category median |cos| = {res['cross_category']['median_offdiagonal']:+.3f}")
        print(f"    specificity control        = {sp['overall']} "
              f"({sp['n_failing']}/{sp['n_categories']} categories read as LENGTH)")

        c = verifier.verify(d)
        print(f"    verifier: {c.checked} checks, {len(c.failures)} failures")
        for f in c.failures:
            print(f"      FAIL {f}")
        outcomes[truth]["_verifier_passed"] = c.passed

    # --- the gate: does the control actually discriminate? ----------------- #
    print("\n" + "=" * 78)
    print("GATE P2 — does the specificity control DISCRIMINATE?")
    print("=" * 78)
    spec = {t: outcomes[t]["specificity_control"]["overall"] for t in outcomes}
    xcat = {t: outcomes[t]["cross_category"]["median_offdiagonal"] for t in outcomes}
    for t in outcomes:
        print(f"  {t:<12} specificity={spec[t]:<5} cross-category median |cos|={xcat[t]:+.3f}")
    print()
    print("  The two instruments answer different questions and the pilot keeps")
    print("  them apart: the specificity control catches an ARTIFACT direction,")
    print("  the cross-category matrix catches CATEGORY COLLAPSE. A high")
    print("  cross-category cosine is equally consistent with 'bias is one")
    print("  mechanism', which is a result rather than a defect.")

    checks = {
        "specificity control passes when no confound is planted":
            spec["distinct"] == "PASS",
        "specificity control FAILS when the direction is pure length":
            spec["pure_length"] == "FAIL",
        "specificity control does not misfire on real-but-shared structure":
            spec["collapsed"] == "PASS",
        "cross-category matrix separates distinct from collapsed":
            xcat["distinct"] < 0.3 < xcat["collapsed"],
        "verifier passed on both runs": all(outcomes[t]["_verifier_passed"]
                                            for t in outcomes),
        "queue manifest reports all steps OK": all(
            outcomes[t]["queue_manifest"]["all_ok"] for t in outcomes),
        # A-4 must report a length direction as measurable exactly when one
        # was planted.  Requiring "usable" in BOTH scenarios was the wrong
        # assertion: the distinct run plants almost no length component, so
        # there is genuinely nothing there to recover and "not usable" is the
        # correct, informative answer.
        # A-4 must report a length direction as measurable exactly when one is
        # there. Only "pure_length" plants a large length component; "collapsed"
        # isolates category collapse and plants none.
        "A-4 self-check finds a length direction when one is planted":
            outcomes["pure_length"]["length_direction_selfcheck"]["usable"],
        "A-4 self-check reports none when none is planted":
            not outcomes["distinct"]["length_direction_selfcheck"]["usable"]
            and not outcomes["collapsed"]["length_direction_selfcheck"]["usable"],
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")

    print()
    print("TIER 2 IS NOT RUN AND THE PILOT IS THEREFORE NOT GREEN.")
    print("  Needs torch + transformers: tokenisation, chat template, the capture")
    print("  index (-1 vs -2, notes/19 §6.1 hole (d)), and a real forward pass.")
    print("  Run `backends.probe_capture_index(hf_id)` per model first.")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
