"""Build the IssueBench exclusion list every Lane A eval run shares.

Two independent reasons an IssueBench row must not be evaluated, resolved once
here into one committed artifact (`datasets/laneA/issuebench_exclusions.json`) so
each config just *names* it and every run's manifest records the same list:

1. **Held-out guard.** The forced-contrast opinion vector was FIT on the TRAIN
   split of an earlier IssueBench run (`--fit-run`). Evaluating "does the
   direction beat prompting" on those same items measures the fit partly on its
   own training data. Two independent 200-item draws from the 636k `sample` split
   overlap on only ~0.06 items in expectation, so this is a small effect — but
   "small in expectation" is not "held out", and the guard costs nothing.

2. **ShareGPT scraping artifacts.** IssueBench materialises 1000 real
   user-prompt templates; 10 of them are scrape debris carrying the ShareGPT web
   UI's chrome ("1 / 1..." / "...Share Prompt"). Those 10 templates are 100%
   artifact, so naming the templates removes every artifact row (636 topics each,
   6360 rows, 1.0% of the split) without hand-picking rows. The handoff asks for
   this filter "before reporting"; doing it before *generating* also spends no GPU
   on prompts we would then discard.

    python scripts/build_heldout_exclusions.py \
        --fit-run runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b

Re-run it if the fit vector changes. It reads only committed artifacts (the fit
run's `examples.csv` + its `manifest.json`, and the IssueBench parquet) and needs
no GPU or API key.
"""

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "datasets" / "laneA" / "issuebench_exclusions.json"
_SPLIT_GLOB = "third_party/issuebench/prompts/prompts_{split}-*.parquet"

# A row is scrape debris if its prompt carries the ShareGPT UI's chrome. Applied
# to the TEMPLATE text, not the materialised prompt, so the rule names templates.
_ARTIFACT_MARKERS = ("Share Prompt", "1 / 1")


def fit_ids(fit_run: Path) -> tuple[list[str], dict]:
    """The example ids the vector was FITTED on = the fit run's TRAIN split.

    `examples.csv` snapshots `train + test` in that order and the manifest records
    `dataset.train_split`, so the train prefix is recoverable exactly. Derived from
    the run's own artifacts rather than re-deriving the split from a seed, so this
    stays correct even if `sample()`'s ordering ever changes.
    """
    import pandas as pd

    manifest = json.loads((fit_run / "manifest.json").read_text())
    cfg = manifest["config"]
    rows = pd.read_csv(fit_run / "examples.csv")
    n_train = int(len(rows) * cfg["dataset"]["train_split"])
    if n_train <= 0:
        raise SystemExit(f"{fit_run}: train_split yields no TRAIN rows to hold out")
    ids = rows["example_id"].astype(str).tolist()[:n_train]
    provenance = {
        "fit_run": fit_run.name,
        "fit_vector_contrast_mode": cfg.get("contrast_mode"),
        "fit_split": cfg["dataset"]["path"],
        "fit_n_snapshot": len(rows),
        "fit_train_split": cfg["dataset"]["train_split"],
        "fit_n_train_heldout": len(ids),
    }
    return ids, provenance


def artifact_template_ids(split: str) -> tuple[list[str], dict]:
    """Template ids whose prompts are ShareGPT scrape debris.

    Asserts each flagged template is *entirely* artifact: if a template were only
    partly debris, dropping it by template would also drop good rows, and the rule
    would have to move to row granularity. Better to fail than to silently thin
    the pool.
    """
    import pandas as pd

    shards = sorted(_REPO.glob(_SPLIT_GLOB.format(split=split)))
    if not shards:
        raise SystemExit(
            f"no IssueBench '{split}' parquet under third_party/issuebench/prompts\n"
            f"    python scripts/fetch_issuebench.py --split {split}"
        )
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    text = df["template_text"].astype(str)
    flagged = text.str.contains("|".join(_ARTIFACT_MARKERS), regex=True, na=False)
    ids = sorted(df.loc[flagged, "template_id"].astype(str).unique())

    in_flagged = df["template_id"].astype(str).isin(ids)
    n_rows, n_bad = int(in_flagged.sum()), int(flagged.sum())
    if n_rows != n_bad:
        raise SystemExit(
            f"{n_rows - n_bad} clean rows share a template with flagged rows — the "
            f"artifact rule is not template-pure on split {split!r}, so it cannot be "
            f"applied by template_id. Drop by example id instead."
        )
    return ids, {
        "split": split,
        "n_templates_total": int(df["template_id"].nunique()),
        "n_templates_artifact": len(ids),
        "n_rows_total": int(len(df)),
        "n_rows_artifact": n_bad,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit-run", required=True, type=Path,
                    help="run dir of the extraction that produced the applied vector")
    ap.add_argument("--split", default="sample", help="IssueBench split being evaluated")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    a = ap.parse_args(argv)

    ids, fit_prov = fit_ids(a.fit_run if a.fit_run.is_absolute() else _REPO / a.fit_run)
    tmpl, art_prov = artifact_template_ids(a.split)

    payload = {
        "purpose": "IssueBench rows Lane A eval runs must not score: the vector's "
                   "own fit items, and ShareGPT scraping artifacts.",
        "built_by": "scripts/build_heldout_exclusions.py",
        "heldout": {**fit_prov, "exclude_ids": ids},
        "artifacts": {**art_prov, "markers": list(_ARTIFACT_MARKERS),
                      "exclude_template_ids": tmpl},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {a.out}")
    print(f"  held out {len(ids)} fit items from {fit_prov['fit_run']}")
    print(f"  excluding {len(tmpl)} artifact templates "
          f"({art_prov['n_rows_artifact']} rows, "
          f"{art_prov['n_rows_artifact'] / art_prov['n_rows_total']:.1%} of split "
          f"{a.split!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
