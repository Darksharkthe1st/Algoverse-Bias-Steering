"""Judge v2.1 verification: fine labels, the codebase-side collapse, spec factory.

Plain Python (no torch, no openai):

    python3 tests/test_judge_v2.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.bias_steer.judge import parse_verdict, UNMATCHED  # noqa: E402
from src.bias_steer.judge_v2 import (  # noqa: E402
    V21_LABELS, COLLAPSE_TO_IGNORED, COLLAPSED_LABELS, BEHAVIOR_LABELS,
    V21_RUBRIC, collapse, judge_v2_spec,
)


def test_nine_fine_labels():
    assert len(V21_LABELS) == 9
    assert len(set(V21_LABELS)) == 9  # no dupes


def test_labels_have_no_substring_collisions():
    # parse_verdict finds a label as a substring after ANSWER:; if one label were a
    # substring of another, an emitted long label could be misread as the short one.
    bad = [(a, b) for a in V21_LABELS for b in V21_LABELS if a != b and a in b]
    assert bad == [], f"substring collisions: {bad}"


def test_parse_verdict_extracts_each_label():
    # Every fine label round-trips through the real extractor the judge uses.
    for lbl in V21_LABELS:
        reply = f"Some reasoning about the response.\nANSWER: {lbl}"
        assert parse_verdict(reply, V21_LABELS) == lbl


def test_collapse_filters_to_ignored():
    for filt in ("unjudgeable", "incoherent", "meta-comment", "unclassifiable"):
        assert collapse(filt) == "ignored"


def test_collapse_passes_behaviors_through():
    for beh in BEHAVIOR_LABELS:
        assert collapse(beh) == beh


def test_collapse_keeps_unmatched_separate():
    # A JUDGE-side parse failure must NOT be folded into ignored (CLAUDE.md §3/§4).
    assert collapse(UNMATCHED) == UNMATCHED
    assert UNMATCHED not in COLLAPSED_LABELS


def test_every_fine_label_maps_into_the_collapsed_set():
    for lbl in V21_LABELS:
        assert collapse(lbl) in COLLAPSED_LABELS


def test_collapse_map_only_touches_non_behaviors():
    # The collapse dict must not accidentally remap a behavior label.
    assert set(COLLAPSE_TO_IGNORED) == {
        "unjudgeable", "incoherent", "meta-comment", "unclassifiable",
    }
    assert all(v == "ignored" for v in COLLAPSE_TO_IGNORED.values())


def test_spec_factory_shape():
    spec = judge_v2_spec(model="gpt-4o-mini", seed=7, temperature=0.0)
    assert spec.name == "neutrality"          # reuses the registered judge fn
    assert spec.labels == V21_LABELS
    assert spec.rubric == V21_RUBRIC
    assert spec.seed == 7 and spec.temperature == 0.0


def test_rubric_mentions_every_label():
    # Cheap guard that the prompt and the label list can't silently drift apart.
    for lbl in V21_LABELS:
        assert lbl in V21_RUBRIC, f"{lbl} missing from the rubric prompt"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
