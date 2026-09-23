"""Phase 3, judge-free: does injecting V_C shift the model's stereotype margin?

    python3 -m scripts.run3_steer_likelihood --model yi-6b \
        --out runs/r3_behavioural_yi-6b --alphas 0.02 0.05 0.1 0.5 1.0

WHY NO JUDGE
------------
`notes/19` §4.1 specified this as the PRIMARY steering metric and argued it is
better than a judge, on three grounds it checked against the code: `score_answers`
already accepts `fwd_hooks`; the measurement is deterministic, carries no judge
version and has no position bias; and it answers "what counts as stereotyped"
with BBQ's own answer key rather than our rubric.  Run 1 used exactly this path.
`EXPERIMENT_PLAN_RUN3.md` §6.4 later replaced it with judge-scored generation;
this restores the earlier specification.

    margin = logP(stereotyped named option) - logP(other named option)

Both terms are mean log-probability per token, so the option lengths cancel.
The model generates NOTHING, so there is no completion to parse and no hedged
answer to adjudicate -- "I think it's the black guy but I can't answer" cannot
arise, because no sentence is produced.  That whole class of ambiguity is
designed out rather than resolved.

THE OPTION LIST MUST NOT BE IN THE PROMPT
-----------------------------------------
`bbq_score`'s module docstring records why: scoring options that the prompt has
just displayed measures list position, not preference.  Moving one option's slot
shifted its score by 0.38 nats and 0.38*sqrt(2) = 0.54 matched the observed mean
margin exactly -- the entire signal was position.  That was design 2 and it is
dead.  This uses design 3, `context + " " + question` with no option list, which
is the validated path that produced the committed margins cache.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
A margin is a DISPOSITION -- which answer the model rates higher -- while the
extraction buckets were a BEHAVIOUR, what it actually said.  They are not the
same construct and the write-up must not imply otherwise.

`notes/19` §4.1 is also explicit that a margin shift alone cannot separate "the
direction moved bias" from "the direction broke the model": at a large enough
dose the model degenerates and likelihood scoring will happily report a clean
shift from a model emitting garbage.  Two things guard that here.  The dose grid
was measured for coherence on this hardware (0.02, 0.05, 0.10 coherent; 0.35 and
above emit one repeated token), and the informative-arm task control is scored
at every dose, so a direction that merely destroys the model shows up as its
accuracy collapsing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pilot import behavioural as bh, pairing              # noqa: E402
from scripts import run3_behavioural_contrast as r3               # noqa: E402
from scripts.run2_annotation_contrast import MODELS               # noqa: E402


def _boot_ci(v, n_boot=2000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _margins(bs, loaded, rows, roles_of, hooks):
    """Per-item logP(stereotyped) - logP(other named), and task accuracy."""
    marg, correct, scored = [], 0, 0
    for row in rows:
        roles = roles_of(row)
        if not roles.usable or roles.biased is None or roles.unknown is None:
            continue
        other = [i for i in range(3) if i not in (roles.biased, roles.unknown)]
        if not other:
            continue
        ans = [row["ans0"], row["ans1"], row["ans2"]]
        s = bs.score_answers(loaded, pairing.prompt_text(row), ans, "", fwd_hooks=hooks)
        marg.append(s[roles.biased] - s[other[0]])
        lab = row.get("label")
        if lab is not None:
            scored += 1
            correct += int(int(np.argmax(s)) == int(lab))
    return marg, (correct / scored if scored else float("nan")), scored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.02, 0.05, 0.1, 0.5, 1.0])
    ap.add_argument("--n-eval", type=int, default=None,
                    help="default: the eval_holdout_n recorded in the sidecars, "
                         "so this scores the SAME items the vectors were held out of")
    ap.add_argument("--n-control", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from src.bias_steer import bbq_score as bs, models as M, steering   # noqa: PLC0415
    from src.bias_steer.bias_taxonomy import resolve_answer_roles       # noqa: PLC0415
    from src.bias_steer.registry import MODELS as SPECS                 # noqa: PLC0415
    from src.bias_steer.datasets import bbq_target_loc                  # noqa: PLC0415
    import torch                                                        # noqa: PLC0415

    ddir = os.path.join(args.out, "directions")
    dirs = {f[:-4]: np.load(os.path.join(ddir, f))
            for f in sorted(os.listdir(ddir)) if f.endswith(".npy")}
    if not dirs:
        print(f"no directions in {ddir} -- run extract first")
        return 1

    # The SAME held-out evaluation items the vectors were kept away from.
    held = args.n_eval
    if held is None:
        for c in dirs:
            sp = os.path.join(args.out, "residuals", f"{c}__ambig.json")
            if os.path.exists(sp):
                v = json.load(open(sp, encoding="utf-8")).get("eval_holdout_n")
                if v:
                    held = v
                    break
    if held is None:
        print("no eval_holdout_n in the sidecars; refusing to score on items that "
              "may have built the vectors. Pass --n-eval to override knowingly.")
        return 1

    cats = sorted(dirs)
    rows_by_cat = r3._load_rows(cats, None, r3.AMBIG, eval_only=True, n_eval=held)
    ctrl_by_cat = r3._load_rows(cats, args.n_control, r3.DISAMBIG)
    targets = bbq_target_loc()

    def roles_of(row):
        row = dict(row)
        row["target_loc"] = targets.get((row["category"], str(row["example_id"])))
        return resolve_answer_roles(bh.row_metadata(row))

    loaded = M.load_model(SPECS[args.model], device=args.device)
    n_layers = loaded.model.cfg.n_layers
    print(f"  {args.model}: {len(dirs)} directions, {held} held-out items each, "
          f"alphas {args.alphas}")

    report = {"model": args.model, "metric": "logP(stereotyped) - logP(other named)",
              "scored_with": "bbq_score.score_answers, no option list in the prompt "
                             "(design 3), no generation, no judge",
              "eval_holdout_n": held, "alphas": args.alphas, "cells": {}}

    for c in cats:
        rows, ctrl = rows_by_cat[c], ctrl_by_cat[c]
        resid = np.load(os.path.join(args.out, "residuals", f"{c}__ambig.npy"),
                        mmap_mode="r")
        base, base_acc, n_sc = _margins(bs, loaded, rows, roles_of, None)
        bm, blo, bhi = _boot_ci(base)
        cell = {"n_items": len(base), "baseline_margin": bm,
                "baseline_ci": [blo, bhi],
                "baseline_task_accuracy": None, "doses": {}}
        _, ctrl_acc, _ = _margins(bs, loaded, ctrl, roles_of, None)
        cell["baseline_task_accuracy"] = ctrl_acc
        print(f"\n  {c}: baseline margin {bm:+.4f} [{blo:+.4f},{bhi:+.4f}]  "
              f"n={len(base)}  task acc {ctrl_acc:.3f}")

        for a in args.alphas:
            vec = r3.dose_vector(dirs[c], resid, a)
            rnd = r3._matched_random(vec, seed=0, resid=np.asarray(resid))
            row = {}
            for tag, v, coeff in (("plus", vec, +float(n_layers)),
                                  ("minus", vec, -float(n_layers)),
                                  ("random_plus", rnd, +float(n_layers))):
                h = steering.apply_resid_pre_add(
                    loaded.model, torch.tensor(v, device=loaded.model.cfg.device,
                                               dtype=torch.float16), coeff)
                m, acc, _ = _margins(bs, loaded, rows, roles_of, h)
                mm, lo, hi = _boot_ci(m)
                # paired shift: same items, steered minus unsteered
                d = [x - y for x, y in zip(m, base)]
                dm, dlo, dhi = _boot_ci(d)
                row[tag] = {"margin": mm, "margin_ci": [lo, hi],
                            "shift_vs_baseline": dm, "shift_ci": [dlo, dhi],
                            "shift_excludes_zero": bool(dlo > 0 or dhi < 0)}
            h = steering.apply_resid_pre_add(
                loaded.model, torch.tensor(vec, device=loaded.model.cfg.device,
                                           dtype=torch.float16), float(n_layers))
            _, tacc, _ = _margins(bs, loaded, ctrl, roles_of, h)
            row["task_accuracy_under_plus"] = tacc
            row["task_accuracy_drop"] = (ctrl_acc - tacc) if np.isfinite(ctrl_acc) else None
            cell["doses"][str(a)] = row
            print(f"    a={a:<5} +{row['plus']['shift_vs_baseline']:+.4f} "
                  f"-{row['minus']['shift_vs_baseline']:+.4f} "
                  f"rand {row['random_plus']['shift_vs_baseline']:+.4f}  "
                  f"task acc {tacc:.3f}"
                  + ("  [+ shift CI excludes 0]" if row["plus"]["shift_excludes_zero"] else ""))
        report["cells"][c] = cell

    p = os.path.join(args.out, "report_steering_likelihood.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {p}")
    print("  Read every dose against BOTH the random control AND the task accuracy: "
          "a shift a norm/covariance-matched random vector also produces is not "
          "evidence about this direction, and a shift with collapsed task accuracy "
          "is damage rather than steering.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
