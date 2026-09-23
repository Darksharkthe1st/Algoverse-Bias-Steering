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

#: Both contrasts must be viable, and they need different buckets:
#:   biased vs OTHER   — the stereotype axis; the contrast the paper leans on
#:   biased vs UNKNOWN — the abstention comparison
#: Gating only on biased/unknown would let a category through with an empty
#: `other` bucket, i.e. with the primary contrast unbuildable.
REQUIRED_BUCKETS = ("biased", "other", "unknown")

#: Max allowed gap between two buckets' chosen-answer-position distributions
#: before the pair is flagged. BBQ balances `target_loc` across positions to
#: within 0.2% of chance, so the DATASET is clean — but the buckets are the
#: dataset filtered by what the model did, and that filter is not position-blind.
#: If the model leans toward one slot, the biased bucket over-represents items
#: where the stereotyped answer sat in that slot, because that is exactly when
#: the lean and the stereotype agree. The prompt lists the options in order, so
#: prompt-token activations carry the layout and the skew reaches the direction.
#: Balanced dataset, skewed buckets: the skew enters at selection.
MAX_POSITION_GAP = 0.10


def position_profile(responses, examples, verdicts) -> dict:
    """Per bucket, the distribution over the answer INDEX the model emitted.

    Returns `{bucket: {"n": int, "share": [p0, p1, p2]}}`.

    This is the diagnostic for the selection-induced position confound. BBQ
    balances which slot the biased answer occupies (verified: 33.4/33.5/33.4
    over 25,814 ambiguous rows), so the dataset is clean. The buckets are not
    the dataset: they are the dataset filtered by the model's choice, and a
    model with a slot preference will fill the `biased` bucket preferentially
    with items where the stereotyped answer sat in its preferred slot — that is
    exactly when its lean and the stereotype agree. `other` skews the opposite
    way for the mirror reason.

    Because the prompt lists the options in order, prompt-token activations
    carry the layout, so this reaches the extracted direction even though we
    capture before generation.
    """
    from src.bias_steer.bias_taxonomy import parse_choice

    acc: dict = {}
    for resp, ex, v in zip(responses, examples, verdicts):
        if v == "unresolved":
            continue
        picked = parse_choice(resp, (ex.metadata or {}).get("answers") or [])
        if picked is None:
            continue
        acc.setdefault(v, [0, 0, 0])
        if 0 <= picked < 3:
            acc[v][picked] += 1

    out = {}
    for bucket, c in acc.items():
        n = sum(c)
        out[bucket] = {"n": n, "share": [round(x / n, 4) if n else 0.0 for x in c]}
    return out


def max_position_gap(profile: dict, a: str, b: str) -> float | None:
    """Largest per-position difference between two buckets' chosen-position
    distributions. None when either bucket is missing or empty.

    This is the number to act on: if `biased` emits position 0 on 55% of its
    items and `other` on 30%, the extracted direction is partly "position 0 vs
    not". The fix is to subsample both buckets to matched position
    distributions before averaging — which costs items, so it has to be known
    before the extraction budget is set.
    """
    pa, pb = profile.get(a), profile.get(b)
    if not pa or not pb or not pa["n"] or not pb["n"]:
        return None
    return max(abs(x - y) for x, y in zip(pa["share"], pb["share"]))


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

        # Which answer INDEX did the model actually emit, per bucket? Not
        # target_loc — the position it chose. See MAX_POSITION_GAP.
        pos = position_profile(responses, exs, verdicts)
        gap = max_position_gap(pos, "biased", "other")

        n_by_bucket = {b: getattr(counts, b) for b in REQUIRED_BUCKETS}
        too_small = [b for b in REQUIRED_BUCKETS if n_by_bucket[b] < MIN_BIASED_ITEMS]

        ok = (rate is not None and rate >= MIN_BIAS_RATE and not too_small)
        if ok:
            usable_categories.append(cat)
            verdict = "OK"
        elif rate is None:
            verdict = "UNUSABLE (nothing scored)"
        elif too_small:
            verdict = "UNUSABLE (" + ", ".join(
                f"only {n_by_bucket[b]} {b}" for b in too_small) + ")"
        else:
            verdict = f"UNUSABLE (rate {rate:.2f} < {MIN_BIAS_RATE})"

        # A position skew does not disqualify a category — it is fixed later by
        # subsampling the buckets to matched position distributions. But it must
        # be visible now, because it changes how many items survive that match.
        if ok and gap is not None and gap > MAX_POSITION_GAP:
            verdict += f"  [!] position gap {gap:.0%} — needs matching"

        rate_str = f"{counts.biased}/{counts.scored}" if counts.scored else "-"
        print(f"{cat:<22}{counts.biased:>8}{counts.unknown:>9}{counts.other:>7}"
              f"{counts.unparsed:>7}{rate_str:>9}  {verdict}")

        report["categories"][cat] = {
            "biased": counts.biased, "unknown": counts.unknown,
            "other": counts.other, "unresolved": counts.unparsed,
            "scored": counts.scored, "bias_rate": rate, "usable": ok,
            "sampled": len(exs),
            "chosen_position": pos,
            "position_gap_biased_vs_other": gap,
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
