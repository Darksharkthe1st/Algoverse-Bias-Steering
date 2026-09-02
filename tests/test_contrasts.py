"""Phase 2 verification: collapse+pool bucketing, group-size gate, 3-vector build.

Torch-free — residuals are opaque stand-ins (plain strings), so this runs anywhere:

    python3 tests/test_contrasts.py
"""

import json
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.bias_steer.contrasts import (  # noqa: E402
    STANCE, CONTRASTS, DEFAULT_N_FLOOR,
    collapse_and_pool, bucket_counts, floor_gate, format_gate, build_three_vectors,
    norm_profile, save_three_vectors,
)
from src.bias_steer.steering import SteeringShapeError  # noqa: E402


class _Row:
    """Stand-in for one layer's (d_model,) slice; only .norm() is exercised."""
    def __init__(self, v):
        self._v = v

    def norm(self):
        return self._v


class _FakeVec:
    """Torch-free stand-in for a (n_layers, d_model) steering vector."""
    def __init__(self, n_layers, d_model, norms=None):
        self.shape = (n_layers, d_model)
        self._norms = norms if norms is not None else [1.0] * n_layers

    def __getitem__(self, i):
        return _Row(self._norms[i])


def _fine_buckets():
    # Fine 9-way -> opaque residual stand-ins. Sizes chosen so V2 clears a floor of
    # 3 (stance=4, soft=3) but V1 does not (hard=1) and V3 does not (non-eng=2).
    return {
        "hard-refusal": ["h1"],
        "soft-refusal": ["s1", "s2", "s3"],
        "non-engagement": ["n1", "n2"],
        "stance-factual": ["f1", "f2"],
        "stance-evaluative": ["e1", "e2"],
        "incoherent": ["i1", "i2"],       # -> ignored
        "meta-comment": ["m1"],           # -> ignored
        "unjudgeable": ["u1"],            # -> ignored
    }


def test_collapse_folds_filters_into_ignored():
    b = collapse_and_pool(_fine_buckets())
    assert set(b["ignored"]) == {"i1", "i2", "m1", "u1"}
    assert len(b["ignored"]) == 4


def test_pooled_stance_is_factual_plus_evaluative():
    b = collapse_and_pool(_fine_buckets())
    assert sorted(b[STANCE]) == ["e1", "e2", "f1", "f2"]
    # fine stance buckets are preserved for reporting
    assert b["stance-factual"] == ["f1", "f2"]
    assert b["stance-evaluative"] == ["e1", "e2"]


def test_behaviors_pass_through_unchanged():
    b = collapse_and_pool(_fine_buckets())
    assert b["hard-refusal"] == ["h1"]
    assert b["soft-refusal"] == ["s1", "s2", "s3"]
    assert b["non-engagement"] == ["n1", "n2"]


def test_collapse_does_not_mutate_input():
    fine = _fine_buckets()
    before = {k: list(v) for k, v in fine.items()}
    collapse_and_pool(fine)
    assert fine == before


def test_bucket_counts():
    b = collapse_and_pool(_fine_buckets())
    c = bucket_counts(b)
    assert c["stance"] == 4 and c["soft-refusal"] == 3 and c["ignored"] == 4
    assert c["missing-label"] == 0  # Counter -> 0, not KeyError


def test_floor_gate_marks_buildable_per_contrast():
    b = collapse_and_pool(_fine_buckets())
    g = floor_gate(b, n_floor=3)
    assert g["V1"]["buildable"] is False   # hard-refusal has 1 (< 3)
    assert g["V2"]["buildable"] is True    # stance 4, soft 3
    assert g["V3"]["buildable"] is False   # non-engagement 2 (< 3)
    assert g["V1"]["pos_n"] == 3 and g["V1"]["neg_n"] == 1


def test_contrasts_reference_real_bucket_keys():
    # Every pole a contrast names must be a key collapse_and_pool can produce.
    producible = {"hard-refusal", "soft-refusal", "non-engagement",
                  "stance-factual", "stance-evaluative", STANCE, "ignored"}
    for pos, neg in CONTRASTS.values():
        assert pos in producible and neg in producible


def test_build_three_vectors_skips_under_floor():
    b = collapse_and_pool(_fine_buckets())
    calls = []

    def fake_build(buckets, contrast):
        calls.append(contrast)
        return f"vec{contrast}"

    out = build_three_vectors(b, build=fake_build, n_floor=3)
    assert set(out) == {"V2"}                       # only V2 clears floor=3
    assert calls == [CONTRASTS["V2"]]               # under-floor contrasts not built


def test_build_three_vectors_can_ignore_floor():
    b = collapse_and_pool(_fine_buckets())
    out = build_three_vectors(b, build=lambda bk, c: c, n_floor=3, require_floor=False)
    assert set(out) == {"V1", "V2", "V3"}


def test_format_gate_renders_all_three():
    g = floor_gate(collapse_and_pool(_fine_buckets()), n_floor=3)
    text = format_gate(g)
    for name in ("V1", "V2", "V3"):
        assert name in text
    assert "UNDER" in text and "OK" in text


def test_default_floor_is_sane():
    assert DEFAULT_N_FLOOR >= 1


# ------------------------------------------------------------------ Phase 3

def test_norm_profile_returns_one_float_per_layer():
    v = _FakeVec(4, 8, norms=[0.0, 1.5, 2.0, 0.25])
    assert norm_profile(v) == [0.0, 1.5, 2.0, 0.25]


def test_save_three_vectors_writes_manifest_and_calls_save_per_vector():
    buckets = collapse_and_pool(_fine_buckets())
    vectors = {"V2": _FakeVec(3, 4, norms=[1.0, 2.0, 3.0]),
               "V3": _FakeVec(3, 4)}
    saved = []

    def fake_save(path, vector, *, n_layers, d_model):
        saved.append((os.path.basename(str(path)), n_layers, d_model))

    with tempfile.TemporaryDirectory() as d:
        manifest = save_three_vectors(vectors, d, n_layers=3, d_model=4,
                                      buckets=buckets, save_fn=fake_save)
        # a manifest file was written and round-trips
        on_disk = json.load(open(os.path.join(d, "vectors_manifest.json"), encoding="utf-8"))
    assert on_disk == manifest
    assert saved == [("V2.safetensors", 3, 4), ("V3.safetensors", 3, 4)]

    v2 = manifest["vectors"]["V2"]
    assert v2["contrast"] == {"pos": "stance", "neg": "soft-refusal"}
    assert v2["shape"] == [3, 4]
    assert v2["norm_profile"] == [1.0, 2.0, 3.0]
    assert v2["pos_n"] == 4 and v2["neg_n"] == 3   # stance=4, soft=3 in _fine_buckets
    assert v2["path"] == "V2.safetensors"


def test_save_three_vectors_rejects_wrong_shape_before_saving():
    saved = []

    def fake_save(path, vector, *, n_layers, d_model):
        saved.append(path)

    with tempfile.TemporaryDirectory() as d:
        try:
            # vector is (3,4) but we claim n_layers=5 -> must fail loud, not save
            save_three_vectors({"V2": _FakeVec(3, 4)}, d, n_layers=5, d_model=4,
                               save_fn=fake_save)
            assert False, "expected SteeringShapeError"
        except SteeringShapeError:
            pass
    assert saved == []  # nothing persisted after the shape guard tripped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
