"""BBQ scenario pairing, and the split construction that rests on it.

This module exists because `notes/17` §5.1 and `notes/13` §2.1 both specify the
primary contrast as "matched on `question_index`" — and `question_index` is not
a 1:1 key.  It takes 25-50 distinct values per category, so a literal join on it
produces a cross product: 1,954,800 "pairs" in Race_x_gender instead of 7,980.
See `notes/19` §6.3.

The real scenario key is

    (question_index, question_polarity, ans0, ans1, ans2)

which yields 1,828 keys in Age, of which 1,816 are exactly one ambiguous plus
one disambiguated row and 12 are (2,2).  BBQ also ships the two members of a
pair as consecutive `example_id`s, which is a useful cross-check and is asserted
here rather than assumed.
"""

from __future__ import annotations

import collections
import json
import os
import random
from dataclasses import dataclass, field

BBQ_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets", "BBQ_Prompt_Sets",
)

AMBIG, DISAMBIG = "ambig", "disambig"
NEG, NONNEG = "neg", "nonneg"


def categories() -> list[str]:
    return sorted(f[:-6] for f in os.listdir(BBQ_DIR) if f.endswith(".jsonl"))


def load_category(category: str) -> list[dict]:
    path = os.path.join(BBQ_DIR, category + ".jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def scenario_key(row: dict) -> tuple:
    """The key on which an ambiguous item has exactly one disambiguated partner.

    `question_index` alone is NOT this key — see the module docstring.  Adding
    polarity and the answer triple is what makes the two arms share the same
    question text and the same three options, which is the property `notes/17`
    §5.1 claims for the contrast and which is what makes the difference of means
    a within-scenario comparison rather than a between-scenario one.
    """
    return (row["question_index"], row["question_polarity"],
            row["ans0"], row["ans1"], row["ans2"])


def polarity_key(row: dict) -> tuple:
    """Key for the SECONDARY contrast: holds context fixed, varies the question.

    Measured over all 25,814 pairs: the two arms share a byte-identical context
    and byte-identical options 100% of the time, and differ in full prompt length
    by 0.985-1.040x.  That makes this the only length-clean contrast available,
    which is why `notes/19` §3.3 promotes it from "a comparison" to the primary's
    control twin.
    """
    return (row["question_index"], row["context_condition"], row["context"],
            row["ans0"], row["ans1"], row["ans2"])


@dataclass(frozen=True)
class Pair:
    """One scenario, in both arms.  The unit of analysis and of splitting."""
    key: tuple
    category: str
    a: dict          # the arm that elicits the behaviour   (ambig / neg)
    b: dict          # the arm that does not                (disambig / nonneg)

    @property
    def pair_id(self) -> str:
        return f"{self.category}:{self.a['example_id']}-{self.b['example_id']}"


def build_pairs(rows: list[dict], category: str, *, contrast: str = "context") -> list[Pair]:
    """Match every row to its partner.  `contrast` is "context" or "polarity".

    Rows that have no partner are dropped and counted by the caller via
    `pairing_report`; they are never silently folded into an arm.
    """
    if contrast == "context":
        keyfn, arm, a_val, b_val = scenario_key, "context_condition", AMBIG, DISAMBIG
    elif contrast == "polarity":
        keyfn, arm, a_val, b_val = polarity_key, "question_polarity", NEG, NONNEG
    else:
        raise ValueError(f"unknown contrast {contrast!r}; expected 'context' or 'polarity'")

    buckets: dict = collections.defaultdict(lambda: {a_val: [], b_val: []})
    for r in rows:
        buckets[keyfn(r)][r[arm]].append(r)

    pairs = []
    for key, sides in buckets.items():
        # A key with (2,2) multiplicity contributes two pairs, matched in the
        # order BBQ ships them.  zip() truncates to the shorter side, which is
        # the drop that `pairing_report` counts.
        for x, y in zip(sorted(sides[a_val], key=lambda r: r["example_id"]),
                        sorted(sides[b_val], key=lambda r: r["example_id"])):
            pairs.append(Pair(key=key, category=category, a=x, b=y))
    pairs.sort(key=lambda p: (p.a["example_id"], p.b["example_id"]))
    return pairs


def pairing_report(rows: list[dict], pairs: list[Pair], *, contrast: str = "context") -> dict:
    """Everything a reader needs to check the pairing was not silently lossy.

    `naive_question_index_join` is reported deliberately.  It is the number the
    spec's literal wording would have produced, and printing it next to the true
    count is what makes the defect visible rather than arguable.
    """
    arm = "context_condition" if contrast == "context" else "question_polarity"
    counts = collections.Counter(r[arm] for r in rows)
    byq: dict = collections.defaultdict(collections.Counter)
    for r in rows:
        byq[r["question_index"]][r["context_condition"]] += 1
    naive = sum(v.get(AMBIG, 0) * v.get(DISAMBIG, 0) for v in byq.values())

    consecutive = sum(1 for p in pairs if abs(int(p.b["example_id"]) - int(p.a["example_id"])) == 1)
    return {
        "contrast": contrast,
        "rows": len(rows),
        "arm_counts": dict(counts),
        "arms_balanced": len(set(counts.values())) == 1,
        "distinct_question_index": len(byq),
        "pairs": len(pairs),
        "rows_dropped_unpaired": len(rows) - 2 * len(pairs),
        "pairs_with_consecutive_example_ids": consecutive,
        "naive_question_index_join": naive,
    }


def arms(pairs: list[Pair]) -> tuple[list[dict], list[dict]]:
    """(behaviour-eliciting arm, benign arm), aligned index for index."""
    return [p.a for p in pairs], [p.b for p in pairs]


def split_pairs(pairs: list[Pair], seed: int) -> tuple[list[Pair], list[Pair]]:
    """Split-half BY SCENARIO PAIR, not by item.  Closes N5, one level up.

    `notes/13` §15 says splits are "stratified by arm".  That is necessary but
    not sufficient: if a scenario's ambiguous row lands in half A and its
    disambiguated row in half B, the two half-directions are estimated from
    DIFFERENT scenarios, and the floor picks up scenario-sampling variance that
    has nothing to do with reproducibility.  Splitting by pair keeps both arms of
    a scenario together, so each half is an independent SAMPLE OF SCENARIOS —
    which is the quantity the floor is supposed to be about.

    Splitting by pair also gives exact arm balance in both halves for free, so it
    strictly subsumes "stratified by arm".  See `notes/19` §6.3.
    """
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    mid = len(idx) // 2
    return [pairs[i] for i in idx[:mid]], [pairs[i] for i in idx[mid:]]


def shuffle_arm_labels(pairs: list[Pair], seed: int) -> list[Pair]:
    """The NEGATIVE CONTROL: swap the two arms within a random half of pairs.

    `notes/13` §3.1 step 2 — items are assigned to arms at random while topic,
    vocabulary, prompt format and n are held exactly fixed.  Swapping within a
    pair is the tightest possible version of that: the two items involved are the
    same scenario, so nothing whatsoever changes except which arm each is called.
    """
    rng = random.Random(seed)
    return [Pair(p.key, p.category, p.b, p.a) if rng.random() < 0.5 else p
            for p in pairs]


# --------------------------------------------------------------------------- #
# The specificity control's raw material — notes/19 §3.3 A-1
# --------------------------------------------------------------------------- #

def length_terciles(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(longest third, shortest third) of `rows` by context length.

    Both returned groups come from the SAME arm, so `context_condition` is held
    exactly fixed and length is the only systematic difference between them.
    That is what makes the resulting direction a measurement of "what reading a
    longer context does" rather than a proxy for it.

    There is enough within-arm variation for this to be well-posed: measured
    p90/p10 of context length inside the ambiguous arm is 1.55-3.42x per
    category, against a between-arm gap of 2.1-2.3x.
    """
    ordered = sorted(rows, key=lambda r: (len(r["context"]), r["example_id"]))
    k = max(1, len(ordered) // 3)
    return ordered[-k:], ordered[:k]


def item_key(row: dict) -> str:
    """A globally unique id for one BBQ row: "<category>:<example_id>".

    `example_id` restarts at 0 in EVERY category file, so it is unique only
    within a file.  Keying a residual cache on it alone silently merges rows
    from different categories that happen to share an index — and because the
    merge is silent, the direction it produces looks entirely normal.

    The pilot hit this the moment an analysis pooled two categories.  In the real
    run, with ten categories and 51,628 items per model, it would have been a
    9-in-10 chance of collision on every id.
    """
    return f"{row['category']}:{row['example_id']}"


def prompt_text(row: dict) -> str:
    """The exact string that gets scored, before the chat template is applied.

    Verbatim persistence of this string is required by `notes/13` §13 and is the
    concrete lesson of N6: a label whose input was not kept cannot be audited.
    """
    return f"{row['context']} {row['question']}"


@dataclass
class CategoryData:
    """Everything one category contributes, with its provenance attached."""
    category: str
    pairs: list = field(default_factory=list)
    report: dict = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def load_pilot_categories(names: list[str], *, limit_pairs: int | None = None,
                          contrast: str = "context") -> list[CategoryData]:
    """Load, pair, and (for the pilot only) truncate to `limit_pairs`.

    `limit_pairs` exists ONLY for the pilot.  The real run uses every matched
    pair, with 32 per arm as the declared minimum (`notes/13` §15).

    Subsampling takes EVENLY SPACED pairs across the file, not the first N.
    BBQ orders its rows by scenario template, so the head of a file is one or two
    templates: the first 20 Disability_status pairs span a context-length range
    of 132-144 characters, against 75-141 for the arm as a whole.  A pilot drawn
    from the head therefore has almost no length variation, the length direction
    is estimated from noise, and the specificity control cannot fire.

    That is not hypothetical — it is what the first pilot run did, and the A-4
    self-check is what caught it.  A pilot sample that does not span the real
    distribution certifies nothing about a control that depends on it.

    Evenly spaced is deterministic, so a pilot run stays reproducible.
    """
    out = []
    for name in names:
        rows = load_category(name)
        pairs = build_pairs(rows, name, contrast=contrast)
        rep = pairing_report(rows, pairs, contrast=contrast)
        if limit_pairs is not None and limit_pairs < len(pairs):
            step = len(pairs) / limit_pairs
            pairs = [pairs[int(i * step)] for i in range(limit_pairs)]
            rep["pilot_subsample"] = {
                "n": len(pairs), "method": "evenly spaced across the file",
                "context_len_range": [
                    min(len(p.a["context"]) for p in pairs),
                    max(len(p.a["context"]) for p in pairs)],
            }
        out.append(CategoryData(category=name, pairs=pairs, report=rep))
    return out
