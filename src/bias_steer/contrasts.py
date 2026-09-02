"""The 3 judge-v2.1 steering contrasts, the bucket prep, and the group-size gate.

`docs/IMPL_PLAN_judge_steer_v2.1.md` Phase 2/3. The generate -> capture -> judge
-> bucket loop already exists (`experiment._extract_vector`); this module is the
v2.1 layer on top:

  1. collapse each fine 9-way verdict to the behavior view (`judges.v2.collapse`),
  2. pool the two stance labels into a single `stance` bucket,
  3. count each bucket and GATE on group size (a difference-of-means over a tiny
     pole is noise) BEFORE building any vector,
  4. build the three contrast vectors via `steering.build_mean_difference`.

Torch-free by construction: residuals are opaque list items here (moved between
buckets, never inspected), so collapse/pool/count/gate all unit-test on CPU. The
only torch touch is `build_three_vectors`, which delegates to
`build_mean_difference` (which lazy-imports torch itself).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from .judges.v2 import collapse
from .steering import assert_steering_shape

# The pooled "any stance" bucket = stance-factual + stance-evaluative.
STANCE = "stance"
STANCE_POOL = ("stance-factual", "stance-evaluative")

# The three contrasts, (positive_label, negative_label); +coeff steers toward pos.
# Mirrors docs/judges/judge_v2.1.md. `stance` is the pooled bucket above.
CONTRASTS: dict[str, tuple[str, str]] = {
    "V1": ("soft-refusal", "hard-refusal"),   # +coeff -> soft refusal
    "V2": (STANCE, "soft-refusal"),           # +coeff -> any stance
    "V3": (STANCE, "non-engagement"),         # +coeff -> any stance
}

# A difference-of-means over fewer than this many examples per pole is too noisy
# to trust (IMPL_PLAN Phase 2). Start here; tune once real counts are in.
DEFAULT_N_FLOOR = 40


def collapse_and_pool(resids_by_fine: dict[str, list]) -> dict[str, list]:
    """Fine 9-way buckets -> the buckets the contrasts consume.

    - each fine label is collapsed (`judges.v2.collapse`): the four non-behavioral
      labels fold into `ignored`; the five behaviors and UNMATCHED pass through,
    - the two stance behaviors are ALSO pooled into a `stance` bucket (the fine
      `stance-factual`/`stance-evaluative` buckets are kept too, for reporting).

    Residual lists are concatenated, never mutated in place. New lists are
    returned so the caller's input is untouched.
    """
    out: dict[str, list] = {}
    for fine, items in resids_by_fine.items():
        key = collapse(fine)
        out.setdefault(key, []).extend(items)
    pooled = list(out.get("stance-factual", [])) + list(out.get("stance-evaluative", []))
    if pooled:
        out[STANCE] = pooled
    return out


def bucket_counts(buckets: dict[str, list]) -> dict[str, int]:
    """{label: n} for every bucket (Counter so missing labels read as 0)."""
    return Counter({k: len(v) for k, v in buckets.items()})


def floor_gate(
    buckets: dict[str, list],
    *,
    contrasts: dict[str, tuple[str, str]] = CONTRASTS,
    n_floor: int = DEFAULT_N_FLOOR,
) -> dict[str, dict]:
    """Per-contrast group-size gate. For each vector, report both poles' counts and
    whether it clears the floor. `buildable` is True iff BOTH poles have >= n_floor
    examples — the gate the IMPL_PLAN puts before Phase 3.
    """
    counts = bucket_counts(buckets)
    report: dict[str, dict] = {}
    for name, (pos, neg) in contrasts.items():
        pos_n, neg_n = counts.get(pos, 0), counts.get(neg, 0)
        report[name] = {
            "pos": pos, "neg": neg, "pos_n": pos_n, "neg_n": neg_n,
            "n_floor": n_floor,
            "buildable": pos_n >= n_floor and neg_n >= n_floor,
        }
    return report


def format_gate(gate: dict[str, dict]) -> str:
    """The Phase 2 observable: a small table of each contrast's pole counts + verdict."""
    lines = [f"{'vector':6} {'contrast':34} {'pos_n':>6} {'neg_n':>6}  floor"]
    for name, r in gate.items():
        contrast = f"{r['pos']} <- {r['neg']}"
        flag = "OK" if r["buildable"] else f"UNDER (<{r['n_floor']})"
        lines.append(f"{name:6} {contrast:34} {r['pos_n']:>6} {r['neg_n']:>6}  {flag}")
    return "\n".join(lines)


def build_three_vectors(
    buckets: dict[str, list],
    *,
    contrasts: dict[str, tuple[str, str]] = CONTRASTS,
    build: Callable | None = None,
    n_floor: int = DEFAULT_N_FLOOR,
    require_floor: bool = True,
) -> dict[str, object]:
    """Build one mean-difference vector per contrast whose poles clear the floor.

    Returns {name: vector} for buildable contrasts. Under-floor contrasts are
    skipped (with `require_floor`, the default) so a noisy pole never silently
    produces a vector; pass `require_floor=False` to build them anyway.

    `build` defaults to `steering.build_mean_difference` (which asserts the pos/neg
    means are (n_layers, d_model) and returns pos - neg). Injected for testing.
    """
    if build is None:
        from .steering import build_mean_difference
        build = build_mean_difference

    gate = floor_gate(buckets, contrasts=contrasts, n_floor=n_floor)
    vectors: dict[str, object] = {}
    for name, (pos, neg) in contrasts.items():
        if require_floor and not gate[name]["buildable"]:
            continue
        vectors[name] = build(buckets, (pos, neg))
    return vectors


# --------------------------------------------------------------------------- #
# Phase 3: validate + persist the built vectors, with a per-layer norm profile.
# --------------------------------------------------------------------------- #

def norm_profile(vector) -> list[float]:
    """Per-layer L2 norm of a (n_layers, d_model) vector -> list of n_layers floats.

    The norm profile is the cheap sanity read on a built vector: a near-zero layer
    means that layer's two class means barely differ (nothing to steer there), and
    a wildly spiking layer flags a degenerate bucket. Logged in the manifest so a
    reviewer can eyeball the vector without loading tensors.
    """
    return [float(vector[i].norm()) for i in range(vector.shape[0])]


def save_three_vectors(
    vectors: dict[str, object],
    out_dir,
    *,
    n_layers: int,
    d_model: int,
    buckets: dict[str, list] | None = None,
    judge_version: str = "v2.1",
    contrasts: dict[str, tuple[str, str]] = CONTRASTS,
    save_fn: Callable | None = None,
    norm_fn: Callable = norm_profile,
) -> dict:
    """Validate every built vector, save each to `<out_dir>/<name>.safetensors`, and
    write `<out_dir>/vectors_manifest.json`. Returns the manifest dict.

    Each vector is shape-checked with `assert_steering_shape` (the (n_layers,
    d_model) guard, CLAUDE.md §6) BEFORE it is saved — a mis-shaped vector fails
    loud here, not silently in a later run. The manifest records, per vector, its
    contrast, shape, per-layer norm profile, and (if `buckets` given) the pole
    counts it was built from — the provenance a reviewer needs.

    `save_fn` defaults to `artifacts.save_vector` (which re-asserts shape at the
    safetensors boundary); injected, along with `norm_fn`, for torch-free testing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if save_fn is None:
        from . import artifacts
        save_fn = artifacts.save_vector

    counts = bucket_counts(buckets) if buckets is not None else {}
    manifest = {
        "judge_version": judge_version,
        "n_layers": n_layers,
        "d_model": d_model,
        "vectors": {},
    }
    for name, vector in vectors.items():
        assert_steering_shape(vector, n_layers, d_model)
        pos, neg = contrasts[name]
        path = out / f"{name}.safetensors"
        save_fn(path, vector, n_layers=n_layers, d_model=d_model)
        manifest["vectors"][name] = {
            "contrast": {"pos": pos, "neg": neg},
            "shape": [n_layers, d_model],
            "norm_profile": norm_fn(vector),
            "pos_n": counts.get(pos),
            "neg_n": counts.get(neg),
            "path": path.name,
        }
    (out / "vectors_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
