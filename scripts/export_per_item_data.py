"""Export the per-item record behind every extracted direction.

    python scripts/export_per_item_data.py

Writes, under `results/per_item/`:

    <model>/<Category>.csv    one row per BBQ item actually scored
    <model>.md                a readable summary per model

WHAT THIS CONTAINS, AND WHAT IT DOES NOT
----------------------------------------
It does NOT contain model responses, because **the reported method never
generated any**. The final pipeline scores the log-probability the model assigns
to each answer option as a continuation of the prompt; it does not sample text.
An earlier, abandoned design did generate text, but those completions were never
persisted either — only the resulting counts, in `runs/_base_rates_*.json`.

So the per-item record is the scoring record, which is what the directions were
actually built from:

    prompt              the exact string scored, reconstructed deterministically
    ans0/ans1/ans2      the three options as BBQ ships them
    biased_idx          which option counts as biased (BBQ's own `target_loc`)
    unknown_idx         which option is the "can't answer" option
    margin              logP(biased) - logP(other named option), length-normalised
                        positive => the model leans toward the stereotyped answer
    abstention_margin   logP(unknown) - max(logP(named)); context, not the contrast
    used_in             top / bottom / middle quintile, i.e. whether this item
                        entered the positive pole, the negative pole, or neither

`margin` is the number everything downstream rests on: the direction is the mean
prompt-token residual of the top-quintile items minus that of the bottom-quintile
items.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer.bias_taxonomy import resolve_answer_roles  # noqa: E402
from src.bias_steer.config import DatasetSpec, SampleSpec  # noqa: E402
from src.bias_steer.datasets import load_bbq, sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "_margins_cache"
OUT = ROOT / "results" / "per_item"
BBQ_DIR = "datasets/BBQ_Prompt_Sets"
OPTION_MARKER = "Pick one of three options: "
QUINTILE = 0.20


def bare_prompt(example) -> str:
    """The prompt as actually scored: context + question, no option list."""
    head, _, _tail = example.prompt.partition(OPTION_MARKER)
    return head.strip()


def main() -> int:
    if not CACHE.exists():
        print(f"no margin cache at {CACHE}")
        return 1

    by_model: dict = {}
    for f in sorted(CACHE.glob("*.json")):
        stem = f.stem                      # <model>_<Category>_<limit>_<seed>
        parts = stem.rsplit("_", 2)        # -> ["<model>_<Category>", limit, seed]
        limit, seed = int(parts[1]), int(parts[2])
        head = parts[0]
        # model names contain '-', categories contain '_'; split on the first
        # '_' that starts a known category by probing the dataset dir.
        model, category = None, None
        for cand in sorted((ROOT / BBQ_DIR).glob("*.jsonl")):
            c = cand.stem
            if head.endswith("_" + c):
                model, category = head[: -(len(c) + 1)], c
                break
        if not model:
            print(f"  skip (cannot parse) {f.name}")
            continue
        by_model.setdefault(model, []).append((category, limit, seed, f))

    OUT.mkdir(parents=True, exist_ok=True)
    grand = 0

    for model in sorted(by_model):
        mdir = OUT / model
        mdir.mkdir(exist_ok=True)
        rows_summary = []

        for category, limit, seed, f in sorted(by_model[model]):
            blob = json.loads(f.read_text())
            exs = load_bbq(DatasetSpec(name="bbq",
                                       path=f"{BBQ_DIR}/{category}.jsonl"))
            exs = sample(exs, SampleSpec(filter={"context_condition": ["ambig"]},
                                         limit=limit, seed=seed))
            items = [(e, resolve_answer_roles(e.metadata)) for e in exs]
            items = [(e, r) for e, r in items if r.usable and r.nonstereo is not None]

            ids = blob["ids"]
            if [e.id for e, _ in items] != ids:
                print(f"  {model}/{category}: id mismatch, skipping")
                continue

            margins = blob["margins"]
            abst = blob["abstention"]
            order = sorted(range(len(margins)), key=lambda i: margins[i])
            k = max(1, int(len(order) * QUINTILE))
            bottom = set(order[:k])
            top = set(order[-k:])

            path = mdir / f"{category}.csv"
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["item_id", "prompt", "ans0", "ans1", "ans2",
                            "biased_idx", "unknown_idx", "other_idx",
                            "question_polarity", "margin", "abstention_margin",
                            "used_in"])
                for i, (e, r) in enumerate(items):
                    a = e.metadata["answers"]
                    used = "top" if i in top else ("bottom" if i in bottom else "middle")
                    w.writerow([e.id, bare_prompt(e), a[0], a[1], a[2],
                                r.biased, r.unknown, r.nonstereo,
                                r.polarity, round(margins[i], 6),
                                round(abst[i], 6), used])

            grand += len(items)
            mean = sum(margins) / len(margins)
            rows_summary.append({
                "category": category, "n": len(items), "mean": mean,
                "top_n": len(top), "bottom_n": len(bottom),
                "top_mean": sum(margins[i] for i in top) / len(top),
                "bottom_mean": sum(margins[i] for i in bottom) / len(bottom),
                "csv": f"{model}/{category}.csv",
                "items": items, "margins": margins, "order": order,
            })
            print(f"  {model}/{category}: {len(items)} items -> {path.relative_to(ROOT)}")

        # ---- per-model markdown ------------------------------------------
        md = [f"# Per-item scoring record — `{model}`", ""]
        md += [
            "Every BBQ item scored for this model, with the margin that placed it",
            "into the positive pole, the negative pole, or neither.",
            "",
            "**There are no model responses here, and none exist.** The reported",
            "method scores the log-probability of each answer option as a",
            "continuation of the prompt — it never samples text. `margin` is",
            "`logP(biased option) - logP(other named option)`, length-normalised.",
            "Positive means the model leans toward the stereotyped answer.",
            "",
            "The direction for a category is the mean prompt-token residual of its",
            "`used_in=top` items minus that of its `used_in=bottom` items.",
            "",
            "| category | items | mean margin | top quintile mean | bottom quintile mean | full data |",
            "|---|---|---|---|---|---|",
        ]
        for s in rows_summary:
            md.append("| %s | %d | %+.3f | %+.3f | %+.3f | [`%s`](%s) |" % (
                s["category"], s["n"], s["mean"], s["top_mean"], s["bottom_mean"],
                Path(s["csv"]).name, Path(s["csv"]).name))

        for s in rows_summary:
            md += ["", f"## {s['category']}", "",
                   f"{s['n']} items scored. The ten most stereotype-leaning and ten "
                   f"least, i.e. the extremes of each pole:", ""]
            items, margins, order = s["items"], s["margins"], s["order"]
            for label, idxs in (("most stereotype-leaning (top pole)", order[-10:][::-1]),
                                ("least / anti-stereotype (bottom pole)", order[:10])):
                md += [f"**{label}**", "",
                       "| margin | prompt | biased option |", "|---|---|---|"]
                for i in idxs:
                    e, r = items[i]
                    p = bare_prompt(e).replace("|", "\\|")
                    if len(p) > 150:
                        p = p[:147] + "..."
                    md.append("| `%+.3f` | %s | %s |" % (
                        margins[i], p, e.metadata["answers"][r.biased]))
                md.append("")

        (OUT / f"{model}.md").write_text("\n".join(md), encoding="utf-8")
        print(f"  -> {(OUT / (model + '.md')).relative_to(ROOT)}")

    print(f"\n{grand} item rows written across {len(by_model)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
