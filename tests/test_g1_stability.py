"""G1a direction-stability statistic (contract §12 A6, docs/PREREG.md §7a).

    python3 tests/test_g1_stability.py

These exercise the decision rule on synthetic activations, so they need torch
but no model and no download. The point is not that the numbers are pretty — it
is that the rule separates the three worlds it has to separate:

  * a real direction in noisy data          -> passes
  * no label signal at all                  -> fails, and fails on the null leg
  * a weak but genuine direction            -> beats the null yet fails the floor

That third case is the one a single criterion would get wrong, and it is why
`assess` requires both legs.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.bias_steer import g1_stability as g1  # noqa: E402

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _pool(n, d, signal, noise, *, seed=0, anisotropy=0.0):
    """(harmful, harmless) activations separated by `signal` along one axis.

    `anisotropy` adds a shared random offset to BOTH groups — the thing that
    makes a random-direction null wrong and the permutation null necessary.
    """
    g = torch.Generator().manual_seed(seed)
    axis = torch.zeros(d)
    axis[0] = 1.0
    shared = torch.randn(d, generator=g) * anisotropy
    h = torch.randn(n, d, generator=g) * noise + signal * axis + shared
    l = torch.randn(n, d, generator=g) * noise + shared
    return h, l


# ---------------------------------------------------------------- pure math

def test_disattenuated_alignment_defines_the_floor():
    # The floor is derived, not copied: S_split = 0.68 <-> alignment 0.90.
    assert abs(g1.disattenuated_alignment(g1.S_SPLIT_FLOOR) - 0.90) < 0.005
    # monotone, and bounded by 1 (perfect agreement -> perfect alignment)
    assert g1.disattenuated_alignment(0.2) < g1.disattenuated_alignment(0.6)
    assert abs(g1.disattenuated_alignment(1.0) - 1.0) < 1e-9
    assert g1.disattenuated_alignment(0.0) == 0.0
    assert g1.disattenuated_alignment(-0.5) == 0.0  # clamped, not NaN


def test_quantile_matches_linear_interpolation():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert g1.quantile(xs, 0.0) == 0.0
    assert g1.quantile(xs, 1.0) == 4.0
    assert abs(g1.quantile(xs, 0.5) - 2.0) < 1e-9
    assert abs(g1.quantile(xs, 0.25) - 1.0) < 1e-9


# ---------------------------------------------------------------- torch-gated

def test_strong_direction_passes_both_legs():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    h, l = _pool(128, 64, signal=3.0, noise=1.0, seed=1)
    r = g1.assess(h, l, n_permutations=200, seed=1)
    assert r["pass"], r
    assert r["beats_null"] and r["clears_floor"], r
    assert r["p_permutation"] < 0.01, r


def test_no_label_signal_fails_on_the_null_leg():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    # Same distribution for both groups: nothing to find.
    h, l = _pool(128, 64, signal=0.0, noise=1.0, seed=2)
    r = g1.assess(h, l, n_permutations=200, seed=2)
    assert not r["pass"], r
    assert not r["beats_null"], "a null-world direction must not beat its own null"


def test_weak_but_real_direction_beats_the_null_yet_fails_the_floor():
    """The case a single criterion gets wrong.

    With enough data a tiny true effect is detectable — it clears the
    permutation null — while still being far too noisily estimated to intervene
    with. `assess` must fail it, on the floor rather than on the null.
    """
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    # Calibrated: at this SNR s_split ~ 0.49, against a null q99 of ~0.23 and a
    # floor of 0.68. The band between them is wide, which is the point — "not
    # chance" and "good enough to intervene with" are far apart.
    h, l = _pool(128, 64, signal=1.4, noise=1.0, seed=5)
    r = g1.assess(h, l, n_permutations=300, seed=5)
    assert r["beats_null"], r          # the signal is real...
    assert not r["clears_floor"], r    # ...and not usable
    assert not r["pass"], r


def test_permutation_null_absorbs_anisotropy_a_random_null_would_miss():
    """Shared structure inflates chance agreement; the null must absorb it.

    Both groups get a large common offset and there is no label signal. A
    Gaussian random-direction null sits near 0 and would call this significant.
    The permutation null sees the same geometry, so it does not.
    """
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    h, l = _pool(128, 64, signal=0.0, noise=1.0, seed=3, anisotropy=6.0)
    r = g1.assess(h, l, n_permutations=200, seed=3)
    assert not r["beats_null"], r
    assert r["p_permutation"] > 0.01, r


def test_permutation_p_is_never_zero():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    h, l = _pool(64, 32, signal=8.0, noise=1.0, seed=4)
    r = g1.assess(h, l, n_permutations=50, seed=4)
    # 1/(B+1) floor: the null sample cannot support a claim of p = 0.
    assert r["p_permutation"] >= 1.0 / 51


def test_split_half_is_deterministic_given_a_seed():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    h, l = _pool(64, 32, signal=2.0, noise=1.0, seed=7)
    assert g1.split_half_cosine(h, l, seed=11) == g1.split_half_cosine(h, l, seed=11)
    assert g1.split_half_cosine(h, l, seed=11) != g1.split_half_cosine(h, l, seed=12)


def test_extraction_is_fp32_even_when_activations_are_fp16():
    """Contract §4 pins fp32 projection. A mean over hundreds of fp16 vectors
    loses precision exactly where the difference is small."""
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    h, l = _pool(128, 64, signal=3.0, noise=1.0, seed=9)
    s32 = g1.split_half_cosine(h, l, seed=1)
    s16 = g1.split_half_cosine(h.half(), l.half(), seed=1)
    assert abs(s32 - s16) < 1e-2, (s32, s16)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed"
          + ("" if _HAS_TORCH else "  (torch-gated tests skipped)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
