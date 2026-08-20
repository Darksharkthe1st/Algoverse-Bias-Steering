"""Step 4 of the bias-taxonomy plan: does each BBQ category actually produce
enough biased answers to build a direction from?

RUN THIS BEFORE ANYTHING ELSE ON THE BOX. The whole experiment contrasts
responses where the model WAS biased against ones where it was not. If a model
almost never picks the stereotyped answer on ambiguous items, the biased bucket
is too small to average and no direction can be built for that category. Nobody
has measured this, and it is cheap to find out.

    python scripts/bbq_base_rates.py --model qwen-1.8b --per-category 120

Prints, per category: how many responses were biased / unknown / other /
unresolved, the bias rate WITH its denominator, and a verdict on whether the
category can support a direction.

Needs the GPU box (it generates). Everything it depends on is unit-tested on CPU.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer import bias_taxonomy as bt          # noqa: E402
from src.bias_steer.config import DatasetSpec, SampleSpec  # noqa: E402
from src.bias_steer.datasets import load_bbq, sample     # noqa: E402
from src.bias_steer.judge import bbq_choice_judge        # noqa: E402

BBQ_DIR = "datasets/BBQ_Prompt_Sets"

CATEGORIES = [
    "Religion", "Race_ethnicity", "Gender_identity", "Age", "Nationality",
    "Physical_appearance", "Disability_status", "Sexual_orientation",
    "Race_x_gender", "Race_x_SES",
]

#: Below this share of biased responses, a category cannot support a stable
#: difference-of-means direction: the positive bucket is too small for its mean
#: to be estimated. Not a deep constant — it is the point at which a
#: split-half of the biased bucket drops under ~20 items.
MIN_BIAS_RATE = 0.15
MIN_BIASED_ITEMS = 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.8b", help="key in the MODELS registry")
    ap.add_argument("--per-category", type=int, default=120,
                    help="ambiguous items sampled per category")
    ap.add_argument("--max-tokens", type=int, default=24,
                    help="short: we only need which option it names")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/_bbq_base_rates.json")
    args = ap.parse_args()

    # Imported here so --help works without torch installed.
    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    print(f"loading {args.model} ...", flush=True)
    loaded = models.load_model(MODELS[args.model])
    print(f"  n_layers={loaded.model.cfg.n_layers} d_model={loaded.model.cfg.d_model}\n")

    header = f"{'category':<22}{'biased':>8}{'unknown':>9}{'other':>7}{'unres':>7}{'rate':>9}  verdict"
    print(header)
    print("-" * len(header))

    report = {"model": args.model, "per_category": args.per_category, "categories": {}}
    usable_categories = []

    for cat in args.categories:
        exs = load_bbq(DatasetSpec(name="bbq", path=f"{BBQ_DIR}/{cat}.jsonl"))
        exs = sample(exs, SampleSpec(
            filter={"context_condition": ["ambig"]},
            limit=args.per_category,
            seed=args.seed,
        ))
        if not exs:
            print(f"{cat:<22}{'— no ambiguous items —':>40}")
            continue

        responses: list[str] = []
        for i in range(0, len(exs), args.batch_size):
            chunk = exs[i:i + args.batch_size]
            responses.extend(models.generate(
                loaded, [e.prompt for e in chunk], args.max_tokens, DEFAULT_SYS,
            ))

        verdicts = bbq_choice_judge(responses, exs)
        counts = bt.ChoiceCounts(
            biased=verdicts.count("biased"),
            unknown=verdicts.count("unknown"),
            other=verdicts.count("other"),
            unparsed=verdicts.count("unresolved"),
        )
        rate = counts.bias_rate

        # A category is usable when BOTH poles can be averaged: enough biased
        # responses, and enough correct ones to contrast them against.
        ok = (rate is not None and rate >= MIN_BIAS_RATE
              and counts.biased >= MIN_BIASED_ITEMS and counts.unknown >= MIN_BIASED_ITEMS)
        if ok:
            usable_categories.append(cat)
            verdict = "OK"
        elif rate is None:
            verdict = "UNUSABLE (nothing scored)"
        elif counts.biased < MIN_BIASED_ITEMS:
            verdict = f"UNUSABLE (only {counts.biased} biased)"
        elif counts.unknown < MIN_BIASED_ITEMS:
            verdict = f"UNUSABLE (only {counts.unknown} unknown)"
        else:
            verdict = f"UNUSABLE (rate {rate:.2f} < {MIN_BIAS_RATE})"

        rate_str = f"{counts.biased}/{counts.scored}" if counts.scored else "-"
        print(f"{cat:<22}{counts.biased:>8}{counts.unknown:>9}{counts.other:>7}"
              f"{counts.unparsed:>7}{rate_str:>9}  {verdict}")

        report["categories"][cat] = {
            "biased": counts.biased, "unknown": counts.unknown,
            "other": counts.other, "unresolved": counts.unparsed,
            "scored": counts.scored, "bias_rate": rate, "usable": ok,
            "sampled": len(exs),
        }

    report["usable_categories"] = usable_categories
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("-" * len(header))
    print(f"\nusable categories: {len(usable_categories)}/{len(args.categories)}"
          f" -> {usable_categories}")
    print(f"written to {out}")

    # Rates are counts over a denominator, never bare percentages
    # (RUNBOOK_JEREMIAH.md standing rule).
    if len(usable_categories) < 3:
        print("\n*** STOP. Fewer than 3 usable categories — there is nothing to")
        print("*** cluster. Do NOT proceed to extraction. Options: a larger or")
        print("*** more opinionated model (see notes/01-models-and-datasets.md),")
        print("*** or the prompt-level contrast, which needs no generation.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
