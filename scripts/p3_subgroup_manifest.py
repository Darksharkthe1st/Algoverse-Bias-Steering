"""WP-43 P3 — freeze the sub-group split BEFORE any floor is computed.

    python -m scripts.p3_subgroup_manifest --out runs/_p3_manifest.json

WHAT P3 ASKS
------------
Race_ethnicity pools nine annotated stereotyped-group sets, so a pooled
direction is an average over however many distinct directions those groups
have. If a single-group subset reproduces where the pooled category does not,
the campaign's race negative is about the UNIT OF ANALYSIS rather than about
race -- and the negative gets a mechanism.

WHY THIS SCRIPT EXISTS SEPARATELY FROM THE RUN
----------------------------------------------
Splitting a category into k subsets multiplies the number of floors computed by
k, and each is measured on less data than the parent. Both push toward finding
SOMETHING above the bar by chance. Run the split first and choose which subsets
to report afterwards and this becomes the cleanest possible example of the
defect the manuscript is about.

So the subset list, the inclusion rule and the reporting rule are all fixed
here, from item counts alone, with no floor computed and no model loaded. Freeze
this file, record its hash in the run, and report every subset it names --
including the ones that fail.

THE INCLUSION RULE, AND WHY THIS NUMBER
---------------------------------------
A subset is tested iff it has **>= 160 scoreable ambiguous items**.

That is not chosen from our data. The extremes contrast takes the top and bottom
`quintile` (0.20) of a subset as its two poles, so n = 160 gives exactly **32
items per pole** -- the standard unit Arditi et al. use and Joad et al. adopt,
and the n at which the reference paper obtains within-category floors of
0.95-0.99. Below it, a null result cannot be distinguished from insufficient
data, which is the trap this campaign keeps falling into.

Two consequences worth seeing before the run: `white` in Race_x_SES (155) and
`asian` in Race_ethnicity (79) fall below the bar and are NOT tested. The rule
was derived before those counts were looked at and is applied as written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bias_steer import bbq_score as bs   # noqa: E402

#: 32 items per pole at quintile 0.20. See the module docstring.
MIN_ITEMS_PER_POLE = 32
QUINTILE = 0.20
MIN_SUBSET_N = int(MIN_ITEMS_PER_POLE / QUINTILE)   # 160

#: Only categories that fail in EVERY model are in scope: the hypothesis is
#: about explaining a negative, so a category that already reproduces has
#: nothing for it to explain.
CANDIDATE_CATEGORIES = [
    "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Gender_identity", "Sexual_orientation", "Nationality",
]


def build(limit: int, seed: int) -> dict:
    cats = {}
    tested = 0
    for cat in CANDIDATE_CATEGORIES:
        sets = bs.stereotyped_group_sets(cat, "ambig", limit, seed)
        rows = []
        for label, info in sets.items():
            ok = info["n"] >= MIN_SUBSET_N
            tested += ok
            rows.append({
                "group": label,
                "aliases": info["aliases"],
                "n": info["n"],
                "poles_at_quintile": int(info["n"] * QUINTILE),
                "tested": ok,
                "reason": ("" if ok else
                           f"n={info['n']} < {MIN_SUBSET_N} "
                           f"({MIN_ITEMS_PER_POLE} per pole at quintile {QUINTILE})"),
            })
        cats[cat] = {"pooled_n": sum(1 for _ in bs.load_scoreable(cat, "ambig", limit, seed)),
                     "n_group_sets": len(sets), "groups": rows}
    return {
        "work_package": "WP-43 P3",
        "frozen": "subset list, inclusion rule and reporting rule fixed before "
                  "any floor was computed and before any model was loaded",
        "sampling": {"condition": "ambig", "limit": limit, "seed": seed},
        "inclusion_rule": {
            "min_items_per_pole": MIN_ITEMS_PER_POLE,
            "quintile": QUINTILE,
            "min_subset_n": MIN_SUBSET_N,
            "justification": "Arditi et al. / Joad et al. standard unit of 32 "
                             "prompts per class; below it a null cannot be "
                             "separated from insufficient data",
        },
        "reporting_rule": [
            "Report EVERY subset named here, including those that fail. No "
            "subset may be added or dropped after a floor is seen.",
            "Report each floor with its n. Subset floors are NOT comparable to "
            "the pooled floor at face value; compare against the pooled "
            "category's floor_vs_n curve at matched n (bias_taxonomy_run stage 6).",
            "A subset counts as a POSITIVE only if it (a) clears the usability "
            "bar and (b) exceeds the pooled category's floor at matched n. "
            "Clearing the bar alone is not enough, because smaller n changes "
            "the floor's distribution.",
            f"{'{n_tested}'} subsets are tested, so under the null the chance of "
            "at least one clearing by luck is not the per-subset rate. Apply "
            "Holm-Bonferroni across the tested subsets within a model before "
            "calling any single subset a positive.",
        ],
        "n_tested_subsets": tested,
        "categories": cats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=600,
                    help="must match the pooled run's --ambig-limit so the "
                         "subset is a strict subset of the same sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/_p3_manifest.json")
    args = ap.parse_args(argv)

    man = build(args.limit, args.seed)
    man["reporting_rule"] = [r.replace("{n_tested}", str(man["n_tested_subsets"]))
                             for r in man["reporting_rule"]]

    print(f"P3 subset manifest — limit={args.limit} seed={args.seed}")
    print(f"inclusion rule: n >= {MIN_SUBSET_N} "
          f"({MIN_ITEMS_PER_POLE} per pole at quintile {QUINTILE})\n")
    print(f"{'category':<21}{'group':<20}{'n':>6}{'poles':>7}  tested")
    print("-" * 66)
    for cat, info in man["categories"].items():
        for r in info["groups"]:
            alias = f"  [= {', '.join(r['aliases'])}]" if r["aliases"] else ""
            print(f"{cat:<21}{r['group']:<20}{r['n']:>6}{r['poles_at_quintile']:>7}"
                  f"  {'YES' if r['tested'] else '  -'}{alias}")
        print()

    print(f"{man['n_tested_subsets']} subsets will be tested, across "
          f"{sum(1 for c in man['categories'].values() if any(r['tested'] for r in c['groups']))} categories.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blob = json.dumps(man, indent=2, sort_keys=True)
    # newline="" so the file is byte-identical on Windows and Linux. Without it
    # Python rewrites "\n" as "\r\n" here and the hash printed below --- taken
    # from the in-memory string --- does not match the hash anyone else computes
    # from the file. A pre-registration hash that does not reproduce is worthless,
    # and the whole point of this file is that a reader can check the split was
    # not chosen after the fact.
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(blob)

    with open(args.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"\nwrote {args.out}")
    print(f"manifest sha256[:16] = {digest}")
    print("Record that hash in the run report. If it changes, the split changed.")
    print(f"Verify with:  python -c \"import hashlib,pathlib;"
          f"print(hashlib.sha256(pathlib.Path('{args.out}').read_bytes()).hexdigest()[:16])\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
