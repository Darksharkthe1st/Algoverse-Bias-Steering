"""
sim_lambda_supplement.py
========================
Supplement to sim_lambda_identifiability.py. Two things the main run does not cover:

 1. CONTAMINATED SHARED: a real shared knob PLUS lambda-proportional incoherence.
    This is the realistic worry -- ablation both moves the knob and degrades the
    model. Does the composite rule correctly refuse to claim, or does it
    manufacture a false "distinct"?

 2. n/k requirement for the PRECISION gate (G2). The stance-side effect is the
    binding constraint; this reports the pass rate of "both trajectories >= 4
    noise-SD" over an n x k grid, plus the generation and judge-call cost.

Run: python3 analysis/sim_lambda_supplement.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim_lambda_identifiability import (  # noqa: E402
    GRID_UNIFORM, SCENARIOS, angle_boot, calibrate, pop_values,
    simulate_cells, stat_gates, run_mc,
)


def verdict(sc, n, k, reps, theta_eq=25.0, mag_min=4.0, B=400, seed=7):
    rng = np.random.default_rng(seed)
    c = dict(g1=0, g2=0, g3=0, g4=0, shared=0, notone=0, inc=0, gated=0)
    for _ in range(reps):
        y, N, q = simulate_cells(sc, GRID_UNIFORM, None, n, n, k, rng,
                                 arms=("r_stance", "r_harm", "r_random"))
        g = stat_gates(y, N, q, GRID_UNIFORM, has_random=True)
        g1 = g["dPh_rharm"] <= -0.15
        g2 = min(g["z_r_stance"], g["z_r_harm"]) >= mag_min
        g3 = g["mag_random"] < mag_min
        g4 = g["same_quadrant"] > 0.5 and g["monotone"] > 0.5
        c["g1"] += g1; c["g2"] += g2; c["g3"] += g3; c["g4"] += g4
        if not (g1 and g2 and g3 and g4):
            c["gated"] += 1
            continue
        bs = angle_boot(y, N, q, GRID_UNIFORM, B=B, rng=rng)
        lo, hi = np.quantile(bs, [0.05, 0.95])
        if hi < theta_eq:
            c["shared"] += 1
        elif lo > theta_eq:
            c["notone"] += 1
        else:
            c["inc"] += 1
    return {kk: v / reps for kk, v in c.items()}


def main():
    reps = 150
    base = SCENARIOS["SHARED"]

    print("=" * 100)
    print("S1. CONTAMINATED SHARED  (true shared knob + lambda-proportional incoherence)")
    print("    correct verdict = SHARED at damage 0, GATED once damage is material")
    print("=" * 100)
    print(f"    {'damage@l=1':>11s} {'popANG_L':>9s} {'G1':>5s} {'G2':>5s} {'G3':>5s} {'G4':>5s}"
          f" {'->SHARED':>9s} {'->NOT-1K':>9s} {'INCONCL':>8s} {'GATED':>7s}")
    for dmg in (0.0, 0.05, 0.10, 0.20, 0.35):
        sc = calibrate(replace(base, damage=dmg))
        pop = pop_values(sc, GRID_UNIFORM)
        v = verdict(sc, 296, 5, reps, seed=int(dmg * 1000) + 4)
        print(f"    {dmg:11.2f} {float(pop['ang_logit'][0]):9.1f} {v['g1']:5.2f} {v['g2']:5.2f}"
              f" {v['g3']:5.2f} {v['g4']:5.2f} {v['shared']:9.2f} {v['notone']:9.2f}"
              f" {v['inc']:8.2f} {v['gated']:7.2f}")

    print("\n" + "=" * 100)
    print("S2. PRECISION GATE (G2: both trajectories >= 4 noise-SD) vs n and k")
    print("    gens = 2 batteries x 9 cells x n x k   (judge calls = gens, 1 call/item)")
    print("=" * 100)
    sc = calibrate(base)
    print(f"    {'n':>5s} {'k':>3s} {'gens':>8s} {'G2 pass':>8s} {'->SHARED':>9s} {'INCONCL':>8s}")
    for n in (150, 296, 450):
        for k in (1, 3, 5, 8):
            v = verdict(sc, n, k, max(reps // 2, 60), seed=n * 13 + k)
            print(f"    {n:5d} {k:3d} {2 * 9 * n * k:8d} {v['g2']:8.2f}"
                  f" {v['shared']:9.2f} {v['inc']:8.2f}")

    print("\n" + "=" * 100)
    print("S3. SENSITIVITY OF THE EQUIVALENCE BOUND theta_eq   (n=296, k=5)")
    print("=" * 100)
    shared = calibrate(base)
    nested = calibrate(SCENARIOS["NESTED"])
    print(f"    {'theta_eq':>9s} {'SHARED->SHARED':>15s} {'SHARED->NOT1K':>14s}"
          f" {'NESTED->NOT1K':>14s} {'NESTED->SHARED':>15s}")
    for te in (15.0, 20.0, 25.0, 30.0, 40.0):
        a = verdict(shared, 296, 5, max(reps // 2, 60), theta_eq=te, seed=int(te))
        b = verdict(nested, 296, 5, max(reps // 2, 60), theta_eq=te, seed=int(te) + 1)
        print(f"    {te:9.1f} {a['shared']:15.2f} {a['notone']:14.2f}"
              f" {b['notone']:14.2f} {b['shared']:15.2f}")


if __name__ == "__main__":
    main()
