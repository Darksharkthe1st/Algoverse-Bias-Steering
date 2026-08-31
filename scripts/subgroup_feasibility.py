"""Is the sub-group experiment feasible? Size it before proposing it.

    python -m scripts.subgroup_feasibility

README §8 lists "split a failing category by stereotyped group" as the most
interesting open lead: Race_ethnicity pools many target groups, so a pooled
direction may be an average over several genuinely different ones. If
Black-targeted items alone yield a direction where pooled Race_ethnicity does
not, the negative result is about the UNIT OF ANALYSIS rather than about race.

That is a real hypothesis and it has never been sized. Sizing it is free: the
stereotyped-group labels ship with BBQ. The reference paper obtains within-
category floors of 0.95-0.99 from 32 items per class, and our own run used
172-320, so the question is how many categories have sub-groups above those
marks once the data is actually split.

This does not run the experiment. It says whether the experiment can be run,
which is the step that was skipped before run 1.
"""

from __future__ import annotations

import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBQ = os.path.join(ROOT, "datasets", "BBQ_Prompt_Sets")

MIN_PAPER = 32      # Joad et al. / Arditi et al. standard unit per class
MIN_RUN1 = 172      # the smallest n any run-1 direction was fit on


def groups_of(row):
    """The stereotyped groups BBQ annotates on this item."""
    md = row.get("additional_metadata") or {}
    g = md.get("stereotyped_groups") or []
    return tuple(sorted(str(x).strip() for x in g if str(x).strip()))


def main():
    cats = sorted(f[:-6] for f in os.listdir(BBQ) if f.endswith(".jsonl"))
    print(f"{'category':<21}{'ambig':>7}{'groups':>8}{'>=32':>7}{'>=172':>7}"
          f"   largest sub-groups (ambiguous items only)")
    print("-" * 108)

    summary = {}
    for c in cats:
        rows = [json.loads(l) for l
                in open(os.path.join(BBQ, c + ".jsonl"), encoding="utf-8")]
        amb = [r for r in rows if r["context_condition"] == "ambig"]
        cnt = collections.Counter(groups_of(r) for r in amb)
        cnt.pop((), None)
        big32 = sum(1 for v in cnt.values() if v >= MIN_PAPER)
        big172 = sum(1 for v in cnt.values() if v >= MIN_RUN1)
        top = ", ".join(f"{'+'.join(k)}={v}" for k, v in cnt.most_common(3))
        summary[c] = dict(n_ambig=len(amb), n_groups=len(cnt),
                          n_ge_32=big32, n_ge_172=big172,
                          counts={"+".join(k): v for k, v in cnt.items()})
        print(f"{c:<21}{len(amb):>7}{len(cnt):>8}{big32:>7}{big172:>7}   {top[:62]}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)

    # The lead is specifically about categories that FAIL everywhere.
    failing = ["Race_ethnicity", "Race_x_SES", "Race_x_gender",
               "Gender_identity", "Sexual_orientation", "Nationality"]
    viable32 = [c for c in failing if summary[c]["n_ge_32"] >= 2]
    viable172 = [c for c in failing if summary[c]["n_ge_172"] >= 2]

    print(f"""
  The hypothesis only bites on categories that fail in every model. Of those
  six -- {', '.join(failing)} --

    {len(viable32):>2} have at least TWO sub-groups with >= {MIN_PAPER} ambiguous items
       ({', '.join(viable32) if viable32 else 'none'})

    {len(viable172):>2} have at least TWO sub-groups with >= {MIN_RUN1} ambiguous items
       ({', '.join(viable172) if viable172 else 'none'})
""")
    if viable172:
        print(f"""  FEASIBLE at run-1 sample sizes on {len(viable172)} categories. The experiment is:
  extract a direction per sub-group instead of per category, run the identical
  floor and negative control, and ask whether any sub-group clears a bar its
  parent category never did.

  Cost: it is a RE-ANALYSIS of the same items, so with cached residuals it is
  pure CPU. If run 2 caches residuals as planned, this experiment costs zero
  additional GPU time -- which is the strongest argument for the caching
  requirement that has come up so far.""")
    elif viable32:
        print(f"""  FEASIBLE only at the reference paper's n={MIN_PAPER}, not at run-1 sizes. That is
  not fatal -- Joad et al. get floors of 0.95-0.99 from exactly 32 per class --
  but it means a null result would be genuinely ambiguous between "no sub-group
  direction" and "not enough items", which is the trap this project keeps
  falling into. Declare the power analysis first.""")
    else:
        print("""  NOT FEASIBLE. No failing category has two sub-groups large enough to compare.
  The lead should be closed rather than left open, and README §8 updated to say
  so with these numbers.""")

    print("""
  ONE CONFOUND TO DECLARE BEFORE RUNNING IT. Splitting a category into k
  sub-groups multiplies the number of floors computed by k, and each is measured
  on less data than the parent. Both push toward finding SOMETHING above the bar
  by chance. Pre-declare the sub-group list and a multiple-comparison correction
  before looking, or this becomes the cleanest possible example of the defect
  the paper is about.
""")
    out = os.path.join(os.path.dirname(ROOT), "runs", "_reanalysis")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "subgroup_feasibility.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
