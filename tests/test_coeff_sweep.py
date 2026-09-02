"""Coefficient sweep: pick a steering coefficient by judging swept outputs.

The selection math is pure (target rate, guard fraction, argmax-under-guard) and
tests against a synthetic label table — no model, no API. One end-to-end test
drives `sweep_coeff` with injected fakes and asserts the 1-D vector guard fires.

    python tests/test_coeff_sweep.py
"""

import os
import sys
import tempfile
import types
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.bias_steer.coeff_sweep import (  # noqa: E402
    choose_coeff, guard_frac, sweep_coeff, target_rate,
)
from src.bias_steer.schema import Example  # noqa: E402
from src.bias_steer.steering import SteeringShapeError  # noqa: E402


def test_rate_and_guard():
    assert target_rate(["stance", "stance", "soft-refusal", "ignored"], "stance") == 0.5
    assert guard_frac(["stance", "ignored", "nonsense", "stance"],
                      ("ignored", "nonsense")) == 0.5


def test_choose_coeff_picks_best_within_guard():
    rates = {0: 0.1, 4: 0.6, 8: 0.9}
    guards = {0: 0.0, 4: 0.1, 8: 0.7}  # c=8 breaks the model
    # c=8 has the highest rate but blows the guard budget -> c=4 wins.
    assert choose_coeff(rates, guards, max_guard_frac=0.5) == 4
    # nothing helps: c=0 (smallest |coeff|) wins the tie.
    assert choose_coeff({0: 0.0, 4: 0.0, -4: 0.0}, {0: 0, 4: 0, -4: 0}, 0.5) == 0


def _fake_model(n_layers=2, d_model=4):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=n_layers, d_model=d_model)),
        spec=types.SimpleNamespace(name="fake"),
    )


def test_sweep_coeff_end_to_end_with_fakes():
    import torch

    def fake_generate(model, prompts, mt, sp):
        return ["soft-refusal"] * len(prompts)               # baseline

    def fake_apply(hooked, vector, c):
        return [("c", c)]

    def fake_hooks(model, prompts, hooks, mt, sp):
        c = hooks[0][1]
        if c >= 8:
            return ["ignored"] * len(prompts)                # model-breaking dose
        if c > 0:
            return ["stance"] * len(prompts)                 # moves to target
        return ["soft-refusal"] * len(prompts)

    examples = [Example(f"e{i}", f"p{i}") for i in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "sweep.csv"
        result = sweep_coeff(
            _fake_model(), torch.zeros(2, 4), examples,
            judge=lambda r, e, s: list(r),
            judge_spec=types.SimpleNamespace(labels=["stance", "soft-refusal", "ignored"]),
            target_label="stance", coeff_grid=[-4, 0, 4, 8],
            guard_labels=("ignored",), max_guard_frac=0.5,
            out_csv=csv_path, generate=fake_generate,
            generate_with_hooks=fake_hooks, apply=fake_apply,
        )
        # c=4 fully converts and stays within guard; c=8 breaks the model.
        assert result.c_star == 4
        assert result.rates[4] == 1.0
        assert result.guard_fracs[8] == 1.0
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "coeff,target_rate,guard_frac,n,chosen"
        assert len(lines) == 1 + 4  # one row per coeff


def test_sweep_coeff_rejects_1d_vector():
    import torch

    try:
        sweep_coeff(
            _fake_model(), torch.zeros(4), [Example("e0", "p")],  # 1-D -> DC offset
            judge=lambda r, e, s: list(r),
            judge_spec=types.SimpleNamespace(labels=["stance"]),
            target_label="stance", coeff_grid=[0, 4],
            generate=lambda *a: ["stance"], generate_with_hooks=lambda *a: ["stance"],
            apply=lambda *a: [],
        )
    except SteeringShapeError:
        return
    raise AssertionError("expected SteeringShapeError on a 1-D steering vector")


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
