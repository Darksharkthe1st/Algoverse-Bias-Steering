"""
sim_lambda_final.py
===================
Final evaluation of the RECOMMENDED decision rule, with the corrected precision
gate (per-direction z on its OWN target DV, not 2-D trajectory length).

Sections:
  F1  operating characteristics of the full rule, all six scenarios, over n x k
  F2  n_ben vs n_harm budget split (the harm side is cheap, the stance side is not)
  F3  k saturation: how much of the achievable variance reduction each k buys
  F4  equivalence-bound sensitivity
  F5  contaminated-shared stress test

Run: python3 analysis/sim_lambda_final.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim_lambda_identifiability import (
    GRID_UNIFORM, SCENARIOS, angle_boot, calibrate, pop_values,
    simulate_cells, stat_gates,
)

THETA_EQ = 25.0
MAG_MIN = 4.0


def verdict(sc, n_ben, n_harm, k, reps, theta_eq=THETA_EQ, mag_min=MAG_MIN,
            B=400, seed=7, lambdas=GRID_UNIFORM):
    rng = np.random.default_rng(seed)
    c = dict(g1=0, g2=0, g3=0, g4=0, shared=0, notone=0, inc=0, gated=0)
    zs = []
    for _ in range(reps):
        y, N, q = simulate_cells(sc, lambdas, None, n_ben, n_harm, k, rng,
                                 arms=("r_stance", "r_harm", "r_random"))
        g = stat_gates(y, N, q, lambdas, has_random=True)
        zs.append(g["z_r_stance"])
        g1 = g["dPh_rharm"] <= -0.15
        g2 = min(g["z_r_stance"], g["z_r_harm"]) >= mag_min
        g3 = g["mag_random"] < mag_min
        g4 = g["same_quadrant"] > 0.5 and g["monotone"] > 0.5
        c["g1"] += g1; c["g2"] += g2; c["g3"] += g3; c["g4"] += g4
        if not (g1 and g2 and g3 and g4):
            c["gated"] += 1
            continue
        bs = angle_boot(y, N, q, lambdas, B=B, rng=rng)
        lo, hi = np.quantile(bs, [0.05, 0.95])
        if hi < theta_eq:
            c["shared"] += 1
        elif lo > theta_eq:
            c["notone"] += 1
        else:
            c["inc"] += 1
    out = {kk: v / reps for kk, v in c.items()}
    out["z_stance_med"] = float(np.median(zs))
    return out


def main():
    reps = 150
    SC = {nm: calibrate(sc) for nm, sc in SCENARIOS.items()}

    print("=" * 108)
    print(f"F1. OPERATING CHARACTERISTICS OF THE RECOMMENDED RULE "
          f"(theta_eq={THETA_EQ} deg, z_min={MAG_MIN}, grid 0/.25/.5/.75/1)")
    print("    correct: SHARED->SHARED | DISTINCT,OBLIQUE,NESTED->NOT-1K | "
          "GENERIC_DAMAGE,POSCTRL_FAIL->GATED")
    print("=" * 108)
    for (nb, nh, k) in ((296, 296, 1), (296, 296, 3), (296, 296, 5),
                        (296, 150, 5), (450, 150, 5)):
        gens = 2 * 9 * ((nb + nh) / 2) * k
        print(f"\n  n_ben={nb} n_harm={nh} k={k}   "
              f"generations ~= {int(9 * (nb + nh) * k)}  (= judge calls)")
        print(f"    {'scenario':18s} {'z_stance':>9s} {'G1':>5s} {'G2':>5s} {'G3':>5s} {'G4':>5s}"
              f" {'->SHARED':>9s} {'->NOT-1K':>9s} {'INCONCL':>8s} {'GATED':>7s} {'VERDICT':>9s}")
        for nm, sc in SC.items():
            v = verdict(sc, nb, nh, k, reps, seed=nb * 7 + nh + k * 3 + len(nm))
            want = {"SHARED": "shared", "DISTINCT": "notone", "DISTINCT_OBLIQUE": "notone",
                    "NESTED": "notone", "GENERIC_DAMAGE": "gated",
                    "POSCTRL_FAIL": "gated"}[nm]
            print(f"    {nm:18s} {v['z_stance_med']:9.1f} {v['g1']:5.2f} {v['g2']:5.2f}"
                  f" {v['g3']:5.2f} {v['g4']:5.2f} {v['shared']:9.2f} {v['notone']:9.2f}"
                  f" {v['inc']:8.2f} {v['gated']:7.2f} {v[want]:9.2f}")

    print("\n" + "=" * 108)
    print("F2. BUDGET SPLIT: the harm battery is cheap (huge effect), the stance battery is not.")
    print("    Fixed ~27k generations. Where should the items go?")
    print("=" * 108)
    print(f"    {'n_ben':>6s} {'n_harm':>7s} {'k':>3s} {'gens':>7s} {'z_stance':>9s}"
          f" {'SHARED->SH':>11s} {'NESTED->N1K':>12s} {'DIST->N1K':>10s}")
    for nb, nh, k in ((296, 296, 5), (400, 200, 5), (500, 100, 5), (600, 60, 5)):
        a = verdict(SC["SHARED"], nb, nh, k, reps, seed=nb + 1)
        b = verdict(SC["NESTED"], nb, nh, k, reps, seed=nb + 2)
        c = verdict(SC["DISTINCT"], nb, nh, k, reps, seed=nb + 3)
        print(f"    {nb:6d} {nh:7d} {k:3d} {9 * (nb + nh) * k:7d} {a['z_stance_med']:9.1f}"
              f" {a['shared']:11.2f} {b['notone']:12.2f} {c['notone']:10.2f}")

    print("\n" + "=" * 108)
    print("F3. k SATURATION at n_ben=296 (item heterogeneity, not sampling noise, dominates)")
    print("=" * 108)
    print(f"    {'k':>3s} {'gens':>7s} {'z_stance(SHARED)':>17s} {'gain vs k=1':>12s}")
    z1 = None
    for k in (1, 2, 3, 5, 8, 12):
        v = verdict(SC["SHARED"], 296, 296, k, 60, seed=100 + k)
        z1 = z1 or v["z_stance_med"]
        print(f"    {k:3d} {9 * 592 * k:7d} {v['z_stance_med']:17.2f}"
              f" {v['z_stance_med'] / z1:12.2f}x")

    print("\n" + "=" * 108)
    print("F4. EQUIVALENCE BOUND theta_eq  (n_ben=296, n_harm=296, k=5)")
    print("=" * 108)
    print(f"    {'theta_eq':>9s} {'SHARED->SHARED':>15s} {'SHARED->NOT1K':>14s}"
          f" {'NESTED->NOT1K':>14s} {'NESTED->SHARED':>15s} {'OBLIQ->NOT1K':>13s}")
    for te in (15.0, 20.0, 25.0, 30.0, 40.0):
        a = verdict(SC["SHARED"], 296, 296, 5, reps, theta_eq=te, seed=int(te))
        b = verdict(SC["NESTED"], 296, 296, 5, reps, theta_eq=te, seed=int(te) + 1)
        c = verdict(SC["DISTINCT_OBLIQUE"], 296, 296, 5, reps, theta_eq=te, seed=int(te) + 2)
        print(f"    {te:9.1f} {a['shared']:15.2f} {a['notone']:14.2f}"
              f" {b['notone']:14.2f} {b['shared']:15.2f} {c['notone']:13.2f}")

    print("\n" + "=" * 108)
    print("F5. CONTAMINATED SHARED (real knob + rising incoherence), n_ben=296 k=5")
    print("    correct: SHARED at damage 0; GATED (no claim) once damage is material")
    print("=" * 108)
    base = SCENARIOS["SHARED"]
    print(f"    {'damage@l=1':>11s} {'popANG_L':>9s} {'G3':>5s} {'G4':>5s}"
          f" {'->SHARED':>9s} {'->NOT-1K':>9s} {'GATED':>7s}")
    for dmg in (0.0, 0.05, 0.10, 0.20, 0.35):
        sc = calibrate(replace(base, damage=dmg))
        pop = pop_values(sc, GRID_UNIFORM)
        v = verdict(sc, 296, 296, 5, reps, seed=int(dmg * 1000) + 44)
        print(f"    {dmg:11.2f} {float(pop['ang_logit'][0]):9.1f} {v['g3']:5.2f} {v['g4']:5.2f}"
              f" {v['shared']:9.2f} {v['notone']:9.2f} {v['gated']:7.2f}")


if __name__ == "__main__":
    main()
