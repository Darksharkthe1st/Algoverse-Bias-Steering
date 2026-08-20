"""Why did likelihood scoring fail on BOTH models? Three checks.

Both qwen-1.8b and gemma-2b failed C1 (order robustness) with order_r ~0.14-0.29.
When every model fails, suspect the measurement.

HYPOTHESIS
    We score option text that is already present in the prompt. The prompt ends
    "Pick one of three options: A, B, C" and we then ask for P(A | prompt).
    Models strongly favour copying recent context, so the option listed LAST is
    boosted for reasons unrelated to the stereotype. That alone would produce a
    near-zero correlation between the two orderings.

CHECK 1 - POSITION EFFECT SIZE
    Score the same options in both orderings and measure how much a given
    option's score changes when it moves from first to last in the list. If the
    position effect dwarfs the stereotype margin, the measurement is dominated
    by list order and no model swap will help.

CHECK 2 - THE FIX
    Drop the option list from the scoring prompt entirely: score each candidate
    answer as a continuation of "context question". With no list, there is no
    list position, and the confound is gone by construction rather than by
    averaging.

CHECK 3 - POSITIVE CONTROL (replaces the now-vacuous C1)
    Removing the option list makes order-robustness undefined, so it cannot be
    the gate any more. Substitute a stronger check: run the DISAMBIGUATED items,
    where the context does identify the right person and there IS a correct
    answer. If the model cannot pick the correct person above chance there, it
    cannot do the task at all and any bias number from the ambiguous items is
    meaningless. This is the same discipline as the project's G1 gate: prove the
    instrument works before reading it.

    python scripts/bbq_scoring_diagnosis.py --model qwen-1.8b
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer.bias_taxonomy import resolve_answer_roles  # noqa: E402
from src.bias_steer.config import DatasetSpec, SampleSpec  # noqa: E402
from src.bias_steer.datasets import load_bbq, sample  # noqa: E402
from scripts.bbq_likelihood_margins import (  # noqa: E402
    OPTION_MARKER, pearson, reorder_prompt, score_options,
)

BBQ_DIR = "datasets/BBQ_Prompt_Sets"


def strip_options(prompt: str) -> str:
    """"<context> <question> Pick one of three options: A, B, C" -> "<context> <question>"."""
    head, _, _tail = prompt.partition(OPTION_MARKER)
    return head.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--categories", nargs="+", default=["Race_ethnicity", "Gender_identity"])
    ap.add_argument("--per-category", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/_bbq_scoring_diagnosis.json")
    args = ap.parse_args()

    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    loaded = models.load_model(MODELS[args.model])
    report = {"model": args.model, "categories": {}}

    for cat in args.categories:
        exs_all = load_bbq(DatasetSpec(name="bbq", path=f"{BBQ_DIR}/{cat}.jsonl"))

        # ---------- ambiguous items: checks 1 and 2 ----------
        amb = sample(exs_all, SampleSpec(filter={"context_condition": ["ambig"]},
                                         limit=args.per_category, seed=args.seed))
        pos_deltas, m_list_a, m_list_b, m_nolist = [], [], [], []
        for e in amb:
            r = resolve_answer_roles(e.metadata)
            if not r.usable or r.nonstereo is None:
                continue
            a = e.metadata["answers"]
            perm = list(range(3))
            perm[r.biased], perm[r.nonstereo] = perm[r.nonstereo], perm[r.biased]
            opts_b = [a[perm[i]] for i in range(3)]

            sa = score_options(loaded, e.prompt, a, DEFAULT_SYS)
            sb = score_options(loaded, reorder_prompt(e.prompt, opts_b), opts_b, DEFAULT_SYS)
            sb_by_id = [0.0, 0.0, 0.0]
            for slot, orig in enumerate(perm):
                sb_by_id[orig] = sb[slot]

            # CHECK 1: how much did the SAME option's score move when it moved slot?
            pos_deltas.append(abs(sa[r.biased] - sb_by_id[r.biased]))
            pos_deltas.append(abs(sa[r.nonstereo] - sb_by_id[r.nonstereo]))

            m_list_a.append(sa[r.biased] - sa[r.nonstereo])
            m_list_b.append(sb_by_id[r.biased] - sb_by_id[r.nonstereo])

            # CHECK 2: no option list at all
            bare = strip_options(e.prompt)
            s_nolist = score_options(loaded, bare, [a[r.biased], a[r.nonstereo]], DEFAULT_SYS)
            m_nolist.append(s_nolist[0] - s_nolist[1])

        n = len(m_nolist)
        mean_pos_delta = sum(pos_deltas) / len(pos_deltas) if pos_deltas else None
        mean_abs_margin = sum(abs(x) for x in m_list_a) / len(m_list_a) if m_list_a else None

        # ---------- disambiguated items: check 3, the positive control ----------
        dis = sample(exs_all, SampleSpec(filter={"context_condition": ["disambig"]},
                                         limit=args.per_category, seed=args.seed))
        correct = total = 0
        for e in dis:
            gold = e.metadata.get("label")
            a = e.metadata["answers"]
            if not isinstance(gold, int):
                continue
            bare = strip_options(e.prompt)
            s = score_options(loaded, bare, a, DEFAULT_SYS)
            if max(range(3), key=lambda i: s[i]) == gold:
                correct += 1
            total += 1
        acc = correct / total if total else None
        # binomial test against 1/3
        z_acc = None
        if total:
            p0 = 1 / 3
            z_acc = (acc - p0) / math.sqrt(p0 * (1 - p0) / total)

        report["categories"][cat] = {
            "n_ambiguous": n,
            "position_effect_mean_abs_logprob_shift": mean_pos_delta,
            "stereotype_margin_mean_abs": mean_abs_margin,
            "position_to_signal_ratio": (mean_pos_delta / mean_abs_margin)
            if (mean_pos_delta and mean_abs_margin) else None,
            "order_r_with_list": pearson(m_list_a, m_list_b),
            "n_disambiguated": total,
            "positive_control_accuracy": acc,
            "positive_control_z_vs_chance": z_acc,
        }

        c = report["categories"][cat]
        print(f"\n===== {args.model} / {cat} =====")
        print(f"  CHECK 1  mean |logprob shift| when an option changes slot : "
              f"{c['position_effect_mean_abs_logprob_shift']:.4f}")
        print(f"           mean |stereotype margin|                        : "
              f"{c['stereotype_margin_mean_abs']:.4f}")
        print(f"           position / signal ratio                         : "
              f"{c['position_to_signal_ratio']:.2f}"
              f"   {'<-- position dominates' if c['position_to_signal_ratio'] and c['position_to_signal_ratio'] > 1 else ''}")
        print(f"  CHECK 3  positive control (disambiguated, no option list) :"
              f" {acc:.1%} correct, n={total}, z vs chance = {z_acc:+.1f}"
              if acc is not None else "  CHECK 3  no disambiguated items")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    print("\nRead: if the position/signal ratio is > 1, the option list is the")
    print("problem and no model swap fixes it. If the positive control is at")
    print("chance, the model cannot do the task and no bias number is meaningful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
