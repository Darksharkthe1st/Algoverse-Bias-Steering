"""
sim_lambda_identifiability.py
=============================

Does the partial-directional-ablation lambda-family identify "one shared knob"
vs "two mechanisms", where the single-point selectivity ratio provably cannot?

Generative simulation + statistic comparison + power analysis.

Design under test
-----------------
Two behaviours, each a binary DV measured on its OWN item battery:
  P_stance = P(engages but declines to commit to X or Y)  on a benign-evaluative battery
  P_harm   = P(refuses)                                   on a harmful-request battery
Two candidate directions r_stance, r_harm (plus a covariance-matched random control).
Partial directional ablation x_lambda = x - lambda (x.rhat) rhat, lambda in a grid.

Latent geometry
---------------
Activation coordinates per item: c = (c1, c2, o1, o2)
  c1, c2  : two "semantic" axes (the candidate mechanisms)
  o1, o2  : orthogonal nuisance axes (what a direction estimate leaks into)
A direction is a unit vector r = (a1, a2, b1, b2).
Ablation at strength lambda:
  proj_i   = a1 c1_i + a2 c2_i + b1 o1_i + b2 o2_i
  c1_i    -= lambda * proj_i * a1
  c2_i    -= lambda * proj_i * a2
This is EXACTLY x - lambda (x.rhat) rhat restricted to the readout coordinates,
so the "estimation-noise leakage" term (-lambda rho sqrt(1-rho^2) eta) is present,
not hand-waved.

Readouts (item-level, then averaged -> the observed rate):
  u_s_i = ws . (c1_i, c2_i);   P_stance_i = sigmoid(a_s (u_s_i - th_s))
  u_h_i = wh . (c1_i, c2_i);   P_harm_i   = sigmoid(a_h (u_h_i - th_h))
th_s, th_h are calibrated numerically so the lambda=0 rates hit the target baselines.
a_s != a_h and baseline rates differ: this IS the confound the critique names
("one knob sampled at two operating points with different thresholds").

Scenarios: SHARED, DISTINCT, DISTINCT_OBLIQUE, NESTED, GENERIC_DAMAGE, POSCTRL_FAIL.

Run:  python3 analysis/sim_lambda_identifiability.py [--quick] [--reps N]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit

# --------------------------------------------------------------------------- #
# Scenario definitions
# --------------------------------------------------------------------------- #


@dataclass
class Scenario:
    name: str
    ws: tuple          # loadings of stance readout on (c1, c2)
    wh: tuple          # loadings of harm readout   on (c1, c2)
    r_stance: tuple    # (a1, a2, b1, b2), normalised at build time
    r_harm: tuple
    a_s: float = 2.0   # readout slope, benign battery
    a_h: float = 1.8   # readout slope, harmful battery  (DIFFERENT on purpose)
    mu_ben: tuple = (1.20, 1.20)    # mean (c1, c2) for benign-evaluative items
    mu_harm: tuple = (3.00, 3.00)   # mean (c1, c2) for harmful-request items
    sd_c: tuple = (1.0, 1.0)
    sd_o: float = 1.0
    p_ben0: float = 0.35            # target baseline P_stance
    p_harm0: float = 0.90           # target baseline P_harm
    damage: float = 0.0             # P(incoherent generation) at lambda=1
    damage_forces: tuple = (1, 0)   # incoherent -> (stance=1 i.e. no commitment, harm=0 i.e. no refusal)
    th_s: float = 0.0               # filled by calibrate()
    th_h: float = 0.0

    def dirs(self):
        out = {}
        for nm, v in (("r_stance", self.r_stance), ("r_harm", self.r_harm)):
            v = np.asarray(v, float)
            out[nm] = v / np.linalg.norm(v)
        # covariance-matched random control: essentially all mass on nuisance axes.
        # cos to c1/c2 ~ 1/sqrt(D_eff) with D_eff ~ 50 -> ~0.14.
        v = np.array([0.14, 0.14, 0.70, 0.70])
        out["r_random"] = v / np.linalg.norm(v)
        return out


# Direction-estimate quality. r_harm is the WELL-estimated direction (refusal DiM
# on a large contrast set, Arditi-style); r_stance is the shakier one. The two
# alignments are DIFFERENT on purpose: under SHARED this makes the two directions
# land at different points on the same curve, which is precisely what makes the
# single-point selectivity ratio depart from 1 with no second mechanism present.
# cos(r_stance, r_harm) = 0.85 * 0.55 = 0.47 under SHARED -> inside Joad et al.'s
# reported 0.4-0.6 band, i.e. geometrically indistinguishable from real DiM pairs.
_ALH = 0.85                       # r_harm alignment to its target axis
_ALS = 0.55                       # r_stance alignment to its target axis
_LKH = np.sqrt(1 - _ALH**2)       # 0.527
_LKS = np.sqrt(1 - _ALS**2)       # 0.835

SCENARIOS = {
    # (a) one latent, both readouts on c1, both directions estimate c1
    "SHARED": Scenario(
        name="SHARED",
        ws=(1.0, 0.0), wh=(1.0, 0.0),
        r_stance=(_ALS, 0.0, _LKS, 0.0),
        r_harm=(_ALH, 0.0, 0.0, _LKH),
    ),
    # (b) two latents, near-orthogonal, each direction estimates its own
    "DISTINCT": Scenario(
        name="DISTINCT",
        ws=(0.0, 1.0), wh=(1.0, 0.0),
        r_stance=(0.0, _ALS, _LKS, 0.0),
        r_harm=(_ALH, 0.0, 0.0, _LKH),
    ),
    # (b') two genuinely distinct latents, but r_stance is CONTAMINATED: it has
    #      cos 0.26 with r_harm and a real 0.30 loading on the harm axis. The
    #      Wollschlaeger "orthogonality != independence" stress case.
    "DISTINCT_OBLIQUE": Scenario(
        name="DISTINCT_OBLIQUE",
        ws=(0.0, 1.0), wh=(1.0, 0.0),
        r_stance=(0.30, _ALS, 0.778, 0.0),
        r_harm=(_ALH, 0.0, 0.0, _LKH),
    ),
    # (c) nested / asymmetric: r_harm moves both, r_stance moves stance only
    "NESTED": Scenario(
        name="NESTED",
        ws=(0.70, 1.0), wh=(1.0, 0.0),
        r_stance=(0.0, _ALS, _LKS, 0.0),
        r_harm=(_ALH, 0.0, 0.0, _LKH),
    ),
    # (d) generic rank-1 damage: directions barely touch the readout axes; all
    #     movement comes from lambda-proportional incoherence, so a random
    #     direction produces the same trajectory.
    "GENERIC_DAMAGE": Scenario(
        name="GENERIC_DAMAGE",
        ws=(1.0, 0.0), wh=(1.0, 0.0),
        r_stance=(0.12, 0.0, 0.993, 0.0),
        r_harm=(0.12, 0.0, 0.0, 0.993),
        damage=0.55,
    ),
    # (e) positive-control failure: r_harm was mis-estimated and does not move
    #     harm refusal at all.
    "POSCTRL_FAIL": Scenario(
        name="POSCTRL_FAIL",
        ws=(0.0, 1.0), wh=(1.0, 0.0),
        r_stance=(0.0, _ALS, _LKS, 0.0),
        r_harm=(0.0, _ALH, 0.0, _LKH),  # loads on the STANCE axis, not the harm axis
    ),
}


# --------------------------------------------------------------------------- #
# Item generation, ablation, and readout
# --------------------------------------------------------------------------- #


def draw_items(sc: Scenario, battery: str, n: int, rng) -> np.ndarray:
    mu = sc.mu_ben if battery == "ben" else sc.mu_harm
    c = np.empty((n, 4))
    c[:, 0] = rng.normal(mu[0], sc.sd_c[0], n)
    c[:, 1] = rng.normal(mu[1], sc.sd_c[1], n)
    c[:, 2] = rng.normal(0.0, sc.sd_o, n)
    c[:, 3] = rng.normal(0.0, sc.sd_o, n)
    return c


def ablate(c: np.ndarray, r: np.ndarray, lam: float) -> np.ndarray:
    """x_lambda = x - lambda (x . rhat) rhat, tracked on the readout coords."""
    proj = c @ r
    out = c.copy()
    out[:, 0] -= lam * proj * r[0]
    out[:, 1] -= lam * proj * r[1]
    return out


def rates(sc: Scenario, c: np.ndarray, battery: str) -> np.ndarray:
    """Per-item probability of the battery's DV."""
    if battery == "ben":
        u = c[:, 0] * sc.ws[0] + c[:, 1] * sc.ws[1]
        return expit(sc.a_s * (u - sc.th_s))
    u = c[:, 0] * sc.wh[0] + c[:, 1] * sc.wh[1]
    return expit(sc.a_h * (u - sc.th_h))


def calibrate(sc: Scenario, n_cal: int = 400_000, seed: int = 0) -> Scenario:
    """Solve th_s, th_h so the lambda=0 marginal rates hit the target baselines."""
    rng = np.random.default_rng(seed)
    cb = draw_items(sc, "ben", n_cal, rng)
    ch = draw_items(sc, "harm", n_cal, rng)
    ub = cb[:, 0] * sc.ws[0] + cb[:, 1] * sc.ws[1]
    uh = ch[:, 0] * sc.wh[0] + ch[:, 1] * sc.wh[1]

    f_s = lambda t: expit(sc.a_s * (ub - t)).mean() - sc.p_ben0
    f_h = lambda t: expit(sc.a_h * (uh - t)).mean() - sc.p_harm0
    th_s = brentq(f_s, -60, 60)
    th_h = brentq(f_h, -60, 60)
    return replace(sc, th_s=th_s, th_h=th_h)


def simulate_cells(sc: Scenario, lambdas, dir_names, n_ben, n_harm, k, rng,
                   arms=("r_stance", "r_harm")):
    """
    Returns dicts keyed by (behaviour, direction, lambda):
      yy[key]  = number of positive generations
      NN[key]  = total generations
      qq[key]  = per-item proportion array (length n) for cluster-robust SEs
    lambda=0 is simulated ONCE per battery and shared across arms (as in the
    real experiment: one baseline run).
    """
    D = sc.dirs()
    y, N, q = {}, {}, {}

    base_items = {"ben": draw_items(sc, "ben", n_ben, rng),
                  "harm": draw_items(sc, "harm", n_harm, rng)}

    def record(b, d, lam, c):
        p = rates(sc, c, b)
        if sc.damage > 0:
            pi = sc.damage * lam
            forced = sc.damage_forces[0] if b == "ben" else sc.damage_forces[1]
            p = (1 - pi) * p + pi * forced
        draws = rng.random((len(p), k)) < p[:, None]
        qi = draws.mean(axis=1)
        y[(b, d, lam)] = int(draws.sum())
        N[(b, d, lam)] = len(p) * k
        q[(b, d, lam)] = qi

    for b in ("ben", "harm"):
        record(b, "_base", 0.0, base_items[b])

    for d in arms:
        r = D[d]
        for lam in lambdas:
            if lam == 0.0:
                for b in ("ben", "harm"):
                    y[(b, d, 0.0)] = y[(b, "_base", 0.0)]
                    N[(b, d, 0.0)] = N[(b, "_base", 0.0)]
                    q[(b, d, 0.0)] = q[(b, "_base", 0.0)]
                continue
            for b in ("ben", "harm"):
                record(b, d, lam, ablate(base_items[b], r, lam))
    return y, N, q


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _p_se(qi):
    """Cluster-robust (item-level) mean and SE of a rate."""
    n = len(qi)
    p = qi.mean()
    se = qi.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return p, max(se, 1e-6)


def _logit_pt(qi, y, N):
    """Empirical logit (Haldane-corrected) and delta-method cluster-robust SE."""
    p, se_p = _p_se(qi)
    ph = (y + 0.5) / (N + 1.0)          # Haldane-Anscombe, keeps logit finite
    p_use = min(max(p, 1.0 / (2 * N)), 1 - 1.0 / (2 * N))
    L = np.log(ph / (1 - ph))
    se_L = se_p / (p_use * (1 - p_use))
    return L, se_L


def traj(y, N, q, lambdas, d, space="logit"):
    """Trajectory points (delta_stance, delta_harm) for direction d, plus SEs."""
    pts, ses = [], []
    if space == "logit":
        Lb0, sb0 = _logit_pt(q[("ben", d, 0.0)], y[("ben", d, 0.0)], N[("ben", d, 0.0)])
        Lh0, sh0 = _logit_pt(q[("harm", d, 0.0)], y[("harm", d, 0.0)], N[("harm", d, 0.0)])
        for lam in lambdas:
            if lam == 0.0:
                continue
            Lb, sb = _logit_pt(q[("ben", d, lam)], y[("ben", d, lam)], N[("ben", d, lam)])
            Lh, sh = _logit_pt(q[("harm", d, lam)], y[("harm", d, lam)], N[("harm", d, lam)])
            pts.append([Lb - Lb0, Lh - Lh0])
            ses.append([np.hypot(sb, sb0), np.hypot(sh, sh0)])
    else:
        pb0, sb0 = _p_se(q[("ben", d, 0.0)])
        ph0, sh0 = _p_se(q[("harm", d, 0.0)])
        for lam in lambdas:
            if lam == 0.0:
                continue
            pb, sb = _p_se(q[("ben", d, lam)])
            ph, sh = _p_se(q[("harm", d, lam)])
            pts.append([pb - pb0, ph - ph0])
            ses.append([np.hypot(sb, sb0), np.hypot(sh, sh0)])
    return np.array(pts), np.array(ses)


def _axis_through_origin(P, W=None):
    """Principal axis of points P (origin-anchored, no centering)."""
    if W is not None:
        P = P * W
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    v = Vt[0]
    if np.dot(P.mean(axis=0), v) < 0:
        v = -v
    return v


def stat_angle(y, N, q, lambdas, space="logit", whiten=True):
    """
    TRAJECTORY ANGLE (degrees) between r_stance's and r_harm's sweeps.
    space='prob'  -> the raw (dP_stance, dP_harm) plane (the proposal as written)
    space='logit' -> the (d logit P_stance, d logit P_harm) plane
    whiten        -> rescale each axis by its median SE so the two behaviours are
                     on a common noise scale (identical map applied to both
                     directions, so the 0-deg SHARED and 90-deg DISTINCT
                     predictions are preserved exactly).
    """
    Ps, Ss = traj(y, N, q, lambdas, "r_stance", space)
    Ph, Sh = traj(y, N, q, lambdas, "r_harm", space)
    if whiten:
        scale = np.median(np.vstack([Ss, Sh]), axis=0)
        scale = np.maximum(scale, 1e-9)
        W = 1.0 / scale
    else:
        W = np.ones(2)
    vs = _axis_through_origin(Ps * W)
    vh = _axis_through_origin(Ph * W)
    cos = float(np.clip(np.dot(vs, vh), -1, 1))
    return np.degrees(np.arccos(cos))


def stat_superposition(y, N, q, lambdas):
    """
    CURVE-SUPERPOSITION statistic (curvature-robust version of the angle).

    Under a single shared knob the reachable set in the (d logit P_s, d logit P_h)
    plane is ONE monotone curve through the common baseline; the two directions
    traverse different ARCS of it (because they have different efficacy) but the
    same curve. A straight-line angle can be fooled by that curvature. This test
    instead asks whether one curve fits both arcs:

      1. whiten both axes by their median SE
      2. rotate into the frame whose x-axis is the pooled principal axis
         (avoids the vertical-line blow-up of a plain y-on-x fit)
      3. fit  y = b1 x + b2 x^2  pooled (2 params) and per-direction (4 params)
      4. F = ((RSS_pool - RSS_sep)/2) / (RSS_sep/(m - 4))

    SHARED -> F small (one curve suffices). DISTINCT/NESTED -> F large.
    """
    Ps, Ss = traj(y, N, q, lambdas, "r_stance", "logit")
    Ph, Sh = traj(y, N, q, lambdas, "r_harm", "logit")
    scale = np.maximum(np.median(np.vstack([Ss, Sh]), axis=0), 1e-9)
    A, B = Ps / scale, Ph / scale
    P = np.vstack([A, B])
    v = _axis_through_origin(P)
    R = np.array([[v[0], v[1]], [-v[1], v[0]]])   # rotate pooled axis onto x
    Ar, Br = A @ R.T, B @ R.T

    def fit(M):
        X = np.column_stack([M[:, 0], M[:, 0] ** 2])
        beta, *_ = np.linalg.lstsq(X, M[:, 1], rcond=None)
        return float(np.sum((M[:, 1] - X @ beta) ** 2))

    rss_sep = fit(Ar) + fit(Br)
    rss_pool = fit(np.vstack([Ar, Br]))
    m = len(Ar) + len(Br)
    dfe = max(m - 4, 1)
    return float(((rss_pool - rss_sep) / 2) / max(rss_sep / dfe, 1e-9))


def stat_selectivity(y, N, q, lam_max=1.0):
    """
    The PRIOR rule: single-point 2x2 selectivity ratio at full ablation.
      R = (|dP_stance| / |dP_harm|)_{ablate r_stance}
        / (|dP_stance| / |dP_harm|)_{ablate r_harm}
    Prior decision rule was R >= 2 -> "distinct".
    """
    def dd(d):
        pb0, _ = _p_se(q[("ben", d, 0.0)]);  ph0, _ = _p_se(q[("harm", d, 0.0)])
        pb, _ = _p_se(q[("ben", d, lam_max)]); ph, _ = _p_se(q[("harm", d, lam_max)])
        return abs(pb - pb0), abs(ph - ph0)
    ds, dh = dd("r_stance")
    es, eh = dd("r_harm")
    eps = 1e-4
    return ((ds + eps) / (dh + eps)) / ((es + eps) / (eh + eps))


def _fit_rank1(yv, Nv, cells, n_b=2):
    """
    Rank-1 ("one knob") logistic model:
        logit p[b, c] = alpha_b + a_b * u_c ,  u_baseline = 0, a_stance = 1
    Returns deviance.
    """
    C = len(cells)
    # params: alpha (2), a_h (1), u (C-1)  [u for the baseline cell fixed at 0]
    def nll(th):
        al = th[:2]; ah = th[2]; u = np.concatenate([[0.0], th[3:]])
        a = np.array([1.0, ah])
        eta = al[:, None] + a[:, None] * u[None, :]
        p = np.clip(expit(eta), 1e-9, 1 - 1e-9)
        return -np.sum(yv * np.log(p) + (Nv - yv) * np.log(1 - p))

    p0 = np.clip(yv / np.maximum(Nv, 1), 1e-4, 1 - 1e-4)
    L0 = np.log(p0 / (1 - p0))
    x0 = np.concatenate([L0[:, 0], [1.0], (L0[0, 1:] - L0[0, 0])])
    res = minimize(nll, x0, method="L-BFGS-B",
                   options=dict(maxiter=4000, ftol=1e-12, gtol=1e-10))
    return 2 * res.fun


def stat_lrt_oneknob(y, N, q, lambdas, deff_cap=50.0):
    """
    Likelihood-ratio statistic for "a single scalar per (direction, lambda) cell
    explains BOTH behaviours". Clustering handled by deflating N to N_eff using
    the observed design effect (over-dispersion) per cell.
    df = 2*C - (C + 2)  where C = number of cells.
    """
    cells = [("_base", 0.0)] + [(d, l) for d in ("r_stance", "r_harm")
                                for l in lambdas if l != 0.0]
    C = len(cells)
    yv = np.zeros((2, C)); Nv = np.zeros((2, C))
    for j, (d, lam) in enumerate(cells):
        for i, b in enumerate(("ben", "harm")):
            qi = q[(b, d, lam)]
            n = len(qi)
            p = qi.mean()
            var_obs = qi.var(ddof=1) / n if n > 1 else 0.0
            k = N[(b, d, lam)] // n
            var_bin = max(p * (1 - p), 1e-9) / (n * k)
            deff = float(np.clip(var_obs / var_bin, 1.0, deff_cap))
            neff = N[(b, d, lam)] / deff
            Nv[i, j] = neff
            yv[i, j] = p * neff
    dev_r1 = _fit_rank1(yv, Nv, cells)
    ps = np.clip(yv / Nv, 1e-9, 1 - 1e-9)
    dev_sat = -2 * np.sum(yv * np.log(ps) + (Nv - yv) * np.log(1 - ps))
    return max(dev_r1 - dev_sat, 0.0), 2 * C - (C + 2)


def stat_random_gap(y, N, q, lambdas):
    """
    Scenario-(d) guard. Angle (deg) between each real direction's logit-plane
    trajectory and the covariance-matched RANDOM direction's trajectory, and the
    ratio of their trajectory lengths. Generic rank-1 damage -> angle ~0 AND
    length ratio ~1 for both real directions.
    """
    out = {}
    Pr, Sr = traj(y, N, q, lambdas, "r_random", "logit")
    scale = np.maximum(np.median(Sr, axis=0), 1e-9)
    W = 1.0 / scale
    vr = _axis_through_origin(Pr * W)
    len_r = np.linalg.norm((Pr * W)[-1])
    for d in ("r_stance", "r_harm"):
        Pd, _ = traj(y, N, q, lambdas, d, "logit")
        vd = _axis_through_origin(Pd * W)
        ang = np.degrees(np.arccos(float(np.clip(np.dot(vd, vr), -1, 1))))
        out[d] = dict(angle_vs_random=ang,
                      len_ratio=float(np.linalg.norm((Pd * W)[-1]) / max(len_r, 1e-9)))
    return out


def _axis2_batch(P):
    """
    Principal axis through the origin for a batch of 2-D point sets.
    P: (B, m, 2)  ->  (B, 2) unit vectors, oriented along the mean displacement.
    Closed form for the top eigenvector of the 2x2 second-moment matrix.
    """
    a = np.einsum("bmi,bmi->b", P[:, :, :1], P[:, :, :1])
    c = np.einsum("bmi,bmi->b", P[:, :, 1:], P[:, :, 1:])
    b = np.einsum("bm,bm->b", P[:, :, 0], P[:, :, 1])
    tr2 = (a + c) / 2.0
    d = np.sqrt(np.maximum(((a - c) / 2.0) ** 2 + b ** 2, 0.0))
    l1 = tr2 + d
    vx, vy = b, l1 - a
    flat = np.abs(b) < 1e-12
    vx = np.where(flat, np.where(a >= c, 1.0, 0.0), vx)
    vy = np.where(flat, np.where(a >= c, 0.0, 1.0), vy)
    nrm = np.maximum(np.hypot(vx, vy), 1e-15)
    v = np.stack([vx / nrm, vy / nrm], axis=1)
    mean = P.mean(axis=1)
    sgn = np.sign(np.einsum("bi,bi->b", mean, v))
    sgn[sgn == 0] = 1.0
    return v * sgn[:, None]


def angle_boot(y, N, q, lambdas, B=400, rng=None, space="logit"):
    """
    Item-level nonparametric bootstrap CI for ANG_L.

    Items are resampled ONCE per battery per replicate and reused across every
    lambda and both directions -- this preserves the real design, where the same
    battery is re-run at each lambda, so the lambda-to-lambda correlation is kept.
    """
    rng = rng or np.random.default_rng(0)
    dirs = ("r_stance", "r_harm")
    lams = [l for l in lambdas if l != 0.0]
    cells = [(d, l) for d in dirs for l in lams]

    Qb = np.stack([q[("ben", d, l)] for d, l in cells])      # (C, n_ben)
    Qh = np.stack([q[("harm", d, l)] for d, l in cells])     # (C, n_harm)
    Qb0 = q[("ben", "r_stance", 0.0)]
    Qh0 = q[("harm", "r_stance", 0.0)]
    nb, nh = len(Qb0), len(Qh0)

    ib = rng.integers(0, nb, size=(B, nb))
    ih = rng.integers(0, nh, size=(B, nh))
    pb = Qb[:, ib].mean(axis=2).T          # (B, C)
    ph = Qh[:, ih].mean(axis=2).T
    pb0 = Qb0[ib].mean(axis=1)             # (B,)
    ph0 = Qh0[ih].mean(axis=1)

    if space == "logit":
        eps_b, eps_h = 0.5 / nb, 0.5 / nh
        f = lambda p, e: np.log((np.clip(p, e, 1 - e)) / (1 - np.clip(p, e, 1 - e)))
        Xb = f(pb, eps_b) - f(pb0, eps_b)[:, None]
        Xh = f(ph, eps_h) - f(ph0, eps_h)[:, None]
    else:
        Xb = pb - pb0[:, None]
        Xh = ph - ph0[:, None]

    L = len(lams)
    Ps = np.stack([Xb[:, :L], Xh[:, :L]], axis=2)      # (B, L, 2) r_stance
    Ph = np.stack([Xb[:, L:], Xh[:, L:]], axis=2)      # (B, L, 2) r_harm
    sc = np.maximum(np.concatenate([Ps, Ph], axis=1).reshape(-1, 2).std(axis=0), 1e-12)
    W = 1.0 / sc
    vs = _axis2_batch(Ps * W)
    vh = _axis2_batch(Ph * W)
    cos = np.clip(np.einsum("bi,bi->b", vs, vh), -1, 1)
    return np.degrees(np.arccos(cos))


def stat_gates(y, N, q, lambdas, has_random=False):
    """Precondition gates that must pass before the angle means anything."""
    g = {}
    eff = stat_effects(y, N, q)
    g["dPh_rharm"] = eff["r_harm"][1]
    g["dPs_rstance"] = eff["r_stance"][0]
    # magnitude of each trajectory in units of its own noise SD
    mags = {}
    for d in ("r_stance", "r_harm"):
        P, S = traj(y, N, q, lambdas, d, "logit")
        sc = np.maximum(np.median(S, axis=0), 1e-9)
        mags[d] = float(np.linalg.norm(P[-1] / sc))
    g["mag_rstance"], g["mag_rharm"] = mags["r_stance"], mags["r_harm"]
    # OWN-TARGET precision: |d logit| / SE for each direction on the DV it is
    # supposed to control. This is the correct precondition -- the 2-D trajectory
    # LENGTH is biased, because under SHARED a direction moves both DVs (long
    # trajectory) while under DISTINCT it moves one (short), so a length gate
    # systematically screens out the very alternatives the test must detect.
    for d, axis in (("r_stance", 0), ("r_harm", 1)):
        P, S = traj(y, N, q, lambdas, d, "logit")
        g[f"z_{d}"] = float(abs(P[-1, axis]) / max(S[-1, axis], 1e-9))
    # coherence: a sign REVERSAL between the two DVs only counts as incoherent when
    # BOTH effects are large (a null component has an arbitrary sign, which is the
    # correct signature of DISTINCT and must not be gated out).
    mono = True
    quad_ok = True
    for d in ("r_stance", "r_harm"):
        P, _ = traj(y, N, q, lambdas, d, "prob")
        big = np.abs(P[-1]) > 0.05
        if big.all() and P[-1, 0] * P[-1, 1] < 0:
            quad_ok = False
        Pp, Sp = traj(y, N, q, lambdas, d, "prob")
        for j in (0, 1):
            if abs(P[-1, j]) < 0.05:
                continue                     # a null component is not "non-monotone"
            seq = np.concatenate([[0.0], Pp[:, j]])
            dd = np.diff(seq)
            se = float(np.median(Sp[:, j]))
            overall = np.sign(seq[-1])
            # only a SIGNIFICANT reversal (> 2 SE against the overall trend) counts
            if np.any(dd * overall < -2.0 * se):
                mono = False
    g["monotone"] = float(mono)
    g["same_quadrant"] = float(quad_ok)
    if has_random:
        P, S = traj(y, N, q, lambdas, "r_random", "logit")
        sc = np.maximum(np.median(S, axis=0), 1e-9)
        g["mag_random"] = float(np.linalg.norm(P[-1] / sc))
        rg = stat_random_gap(y, N, q, lambdas)
        g["lenratio_rstance"] = rg["r_stance"]["len_ratio"]
        g["lenratio_rharm"] = rg["r_harm"]["len_ratio"]
    return g


def stat_effects(y, N, q, lam=1.0):
    """Raw dP at full ablation -- for precondition gates and reporting."""
    out = {}
    for d in ("r_stance", "r_harm"):
        pb0, _ = _p_se(q[("ben", d, 0.0)]); ph0, _ = _p_se(q[("harm", d, 0.0)])
        pb, _ = _p_se(q[("ben", d, lam)]);  ph, _ = _p_se(q[("harm", d, lam)])
        out[d] = (pb - pb0, ph - ph0)
    return out


# --------------------------------------------------------------------------- #
# Monte Carlo driver
# --------------------------------------------------------------------------- #

GRID_UNIFORM = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_mc(sc, lambdas, n_ben, n_harm, k, reps, seed=1234, with_random=False):
    rng = np.random.default_rng(seed)
    arms = ("r_stance", "r_harm") + (("r_random",) if with_random else ())
    rows = []
    for _ in range(reps):
        y, N, q = simulate_cells(sc, lambdas, None, n_ben, n_harm, k, rng, arms=arms)
        lrt, df = stat_lrt_oneknob(y, N, q, lambdas)
        rec = dict(
            ang_logit=stat_angle(y, N, q, lambdas, "logit", True),
            ang_prob=stat_angle(y, N, q, lambdas, "prob", True),
            ang_prob_raw=stat_angle(y, N, q, lambdas, "prob", False),
            ang_logit_2pt=stat_angle(y, N, q, [0.0, 1.0], "logit", True),
            sup=stat_superposition(y, N, q, lambdas) if len(lambdas) >= 4 else np.nan,
            sel=stat_selectivity(y, N, q),
            lrt=lrt, lrt_df=df,
        )
        eff = stat_effects(y, N, q)
        rec["ds_rs"], rec["dh_rs"] = eff["r_stance"]
        rec["ds_rh"], rec["dh_rh"] = eff["r_harm"]
        if with_random:
            g = stat_random_gap(y, N, q, lambdas)
            rec["rand_ang_rs"] = g["r_stance"]["angle_vs_random"]
            rec["rand_ang_rh"] = g["r_harm"]["angle_vs_random"]
            rec["rand_len_rs"] = g["r_stance"]["len_ratio"]
            rec["rand_len_rh"] = g["r_harm"]["len_ratio"]
        rows.append(rec)
    keys = rows[0].keys()
    return {kk: np.array([r[kk] for r in rows], float) for kk in keys}


def pop_values(sc, lambdas, seed=7):
    """Near-noiseless population values of every statistic."""
    return run_mc(sc, lambdas, 200_000, 200_000, 1, 1, seed=seed, with_random=True)


def qtiles(a, qs=(0.05, 0.5, 0.95)):
    return [float(np.quantile(a, x)) for x in qs]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def report_population(SC):
    print("\n" + "=" * 100)
    print("A. POPULATION (noiseless) BEHAVIOUR OF EACH SCENARIO  [lambda grid 0,.25,.5,.75,1]")
    print("=" * 100)
    hdr = (f"{'scenario':18s} {'dPs|r_s':>8s} {'dPh|r_s':>8s} {'dPs|r_h':>8s} {'dPh|r_h':>8s}"
           f" {'SEL':>9s} {'ANG_P':>7s} {'ANG_L':>7s} {'SUP':>9s} {'rndANG_s':>9s} {'rndANG_h':>9s}")
    print(hdr); print("-" * len(hdr))
    pop = {}
    for nm, sc in SC.items():
        r = pop_values(sc, GRID_UNIFORM)
        pop[nm] = {k: float(v[0]) for k, v in r.items()}
        print(f"{nm:18s} {r['ds_rs'][0]:8.3f} {r['dh_rs'][0]:8.3f} {r['ds_rh'][0]:8.3f}"
              f" {r['dh_rh'][0]:8.3f} {r['sel'][0]:9.2f} {r['ang_prob'][0]:7.1f}"
              f" {r['ang_logit'][0]:7.1f} {r['sup'][0]:9.1f}"
              f" {r['rand_ang_rs'][0]:9.1f} {r['rand_ang_rh'][0]:9.1f}")
    print("\nSEL   = single-point selectivity ratio (prior rule fired if SEL >= 2).")
    print("ANG_P = angle in the raw (dP_s, dP_h) plane (the proposal as written).")
    print("ANG_L = angle in the (d logit P_s, d logit P_h) plane.")
    print("SUP   = curve-superposition F (one curve for both arcs?).")
    print("NOTE: population SEL for SHARED is the headline -- a value >= 2 with NO")
    print("      second mechanism present is exactly the critique's non-identification.")
    return pop


def report_sampling(SC, reps, n_ben, n_harm, k):
    print("\n" + "=" * 100)
    print(f"B. SAMPLING DISTRIBUTIONS  n_ben={n_ben} n_harm={n_harm} k={k} reps={reps}")
    print("=" * 100)
    hdr = (f"{'scenario':18s} | {'ANG_L  (5/50/95)':>26s} | {'ANG_P  (5/50/95)':>26s} "
           f"| {'SEL (5/50/95)':>26s} | {'SUP (50/95)':>16s}")
    print(hdr); print("-" * len(hdr))
    out = {}
    for nm, sc in SC.items():
        r = run_mc(sc, GRID_UNIFORM, n_ben, n_harm, k, reps, seed=abs(hash(nm)) % 10_000)
        out[nm] = r
        al, ap, se, su = (qtiles(r["ang_logit"]), qtiles(r["ang_prob"]),
                          qtiles(r["sel"]), qtiles(r["sup"], (0.5, 0.95)))
        print(f"{nm:18s} | {al[0]:7.1f}{al[1]:8.1f}{al[2]:8.1f}   "
              f"| {ap[0]:7.1f}{ap[1]:8.1f}{ap[2]:8.1f}   "
              f"| {se[0]:8.2f}{se[1]:9.2f}{se[2]:9.2f} "
              f"| {su[0]:7.1f}{su[1]:8.1f}")
    return out


def report_discrimination(mc):
    print("\n" + "=" * 100)
    print("C. DISCRIMINATION: SHARED vs each alternative")
    print("   crit = 95th pct of the statistic under SHARED (5% false-'distinct' rate)")
    print("=" * 100)
    stats = ("ang_logit", "ang_prob", "ang_logit_2pt", "sup", "sel", "lrt")
    for stat in stats:
        crit = float(np.quantile(mc["SHARED"][stat], 0.95))
        print(f"\n  statistic = {stat:14s}   crit(SHARED 95th) = {crit:10.2f}")
        for nm, r in mc.items():
            if nm == "SHARED":
                continue
            pw = float((r[stat] > crit).mean())
            note = ""
            if nm == "GENERIC_DAMAGE":
                note = "   <- must be caught by the RANDOM-DIRECTION gate, not this stat"
            if nm == "POSCTRL_FAIL":
                note = "   <- must be caught by the POSITIVE-CONTROL gate, not this stat"
            print(f"     power vs {nm:18s} = {pw:5.3f}{note}")
    print("\n  Pairwise AUC (SHARED vs alternative), higher = better separation:")
    hdr = f"    {'alternative':20s}" + "".join(f"{s:>15s}" for s in stats)
    print(hdr)
    for nm, r in mc.items():
        if nm == "SHARED":
            continue
        line = f"    {nm:20s}"
        for s in stats:
            a, b = mc["SHARED"][s], r[s]
            auc = float((b[:, None] > a[None, :]).mean() + 0.5 * (b[:, None] == a[None, :]).mean())
            line += f"{auc:15.3f}"
        print(line)
    print("\n  PRIOR RULE AUDIT -- fixed threshold SEL >= 2 (no calibration):")
    for nm, r in mc.items():
        lab = "FALSE 'distinct' rate" if nm == "SHARED" else "fires 'distinct'"
        print(f"     {nm:18s}  P(SEL >= 2) = {float((r['sel'] >= 2).mean()):5.3f}   ({lab})")


def report_power(SC, reps):
    print("\n" + "=" * 100)
    print("D. POWER vs n and k   (ANG_L primary; crit calibrated on SHARED at each n,k)")
    print("=" * 100)
    print(f"    {'n':>5s} {'k':>3s} {'gens':>8s} {'crit_deg':>9s}"
          f" {'pw:DISTINCT':>12s} {'pw:D_OBLIQ':>11s} {'pw:NESTED':>10s}"
          f" {'SELpw:DIST':>11s} {'SELfp':>7s}")
    rows = []
    for n in (100, 150, 296, 600):
        for k in (1, 3, 5):
            mcs = {nm: run_mc(SC[nm], GRID_UNIFORM, n, n, k, reps, seed=(n * 31 + k * 7))
                   for nm in ("SHARED", "DISTINCT", "DISTINCT_OBLIQUE", "NESTED")}
            crit = float(np.quantile(mcs["SHARED"]["ang_logit"], 0.95))
            selcrit = 2.0  # the prior fixed rule
            gens = 2 * (1 + 2 * (len(GRID_UNIFORM) - 1)) * n * k
            row = dict(n=n, k=k, gens=gens, crit=crit,
                       pw_dist=float((mcs["DISTINCT"]["ang_logit"] > crit).mean()),
                       pw_obl=float((mcs["DISTINCT_OBLIQUE"]["ang_logit"] > crit).mean()),
                       pw_nest=float((mcs["NESTED"]["ang_logit"] > crit).mean()),
                       sel_pw=float((mcs["DISTINCT"]["sel"] > selcrit).mean()),
                       sel_fp=float((mcs["SHARED"]["sel"] > selcrit).mean()))
            rows.append(row)
            print(f"    {n:5d} {k:3d} {gens:8d} {crit:9.1f} {row['pw_dist']:12.3f}"
                  f" {row['pw_obl']:11.3f} {row['pw_nest']:10.3f}"
                  f" {row['sel_pw']:11.3f} {row['sel_fp']:7.3f}")
    return rows


def report_intermediate(reps, n=296, k=1):
    """
    Continuum SHARED -> DISTINCT: minimum detectable true angle.
    Mixing m: stance readout = (1-m)*c1 + m*c2, r_stance rotates with it.
    m=0 -> SHARED, m=1 -> DISTINCT.
    """
    print("\n" + "=" * 100)
    print(f"E. MINIMUM DETECTABLE TRUE ANGLE   (n={n}, k={k}, grid 0,.25,.5,.75,1)")
    print("=" * 100)
    base = SCENARIOS["SHARED"]
    fam = {}
    for m in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        ws = (1 - m, m)
        nrm = np.hypot(*ws)
        sc = calibrate(replace(base, name=f"MIX{m}", ws=ws,
                               r_stance=(_ALS * ws[0] / nrm, _ALS * ws[1] / nrm, _LKS, 0.0)))
        fam[m] = sc
    crit = crit2 = None
    tab = []
    print(f"    {'mix m':>7s} {'true ANG_L':>11s} {'obs 5/50/95':>26s} {'pw(5pt)':>8s} {'pw(2pt)':>8s}")
    for m, sc in fam.items():
        pop = pop_values(sc, GRID_UNIFORM)
        r = run_mc(sc, GRID_UNIFORM, n, n, k, reps, seed=int(m * 1000) + 3)
        if m == 0.0:
            crit = float(np.quantile(r["ang_logit"], 0.95))
            crit2 = float(np.quantile(r["ang_logit_2pt"], 0.95))
        qq = qtiles(r["ang_logit"])
        pw = float((r["ang_logit"] > crit).mean())
        pw2 = float((r["ang_logit_2pt"] > crit2).mean())
        tab.append((float(pop["ang_logit"][0]), pw))
        print(f"    {m:7.2f} {pop['ang_logit'][0]:11.1f} "
              f"{qq[0]:8.1f}{qq[1]:8.1f}{qq[2]:8.1f} {pw:8.3f} {pw2:8.3f}")
    print(f"    (crit at m=0: 5pt={crit:.1f} deg, 2pt={crit2:.1f} deg)")
    tab.sort()
    xs = np.array([t[0] for t in tab]); ys = np.array([t[1] for t in tab])
    if ys.max() >= 0.8 and ys.min() < 0.8:
        mdd = float(np.interp(0.8, ys, xs))
        print(f"    MINIMUM DETECTABLE TRUE ANGLE at 80% power, n={n}, k={k}:  {mdd:.0f} deg")


def report_grids(SC, reps, budget_gens=6000):
    print("\n" + "=" * 100)
    print(f"F. LAMBDA GRID CHOICE at MATCHED GENERATION BUDGET (~{budget_gens} generations, k=1)")
    print("=" * 100)
    grids = {
        "{0,1}  (prior 2x2)":        [0.0, 1.0],
        "{0,.5,1}":                  [0.0, 0.5, 1.0],
        "{0,.25,.5,.75,1} uniform":  [0.0, 0.25, 0.5, 0.75, 1.0],
        "{0,.5,.75,.9,1} dense-hi":  [0.0, 0.5, 0.75, 0.9, 1.0],
        "{0,.35,.6,.8,1} log-ish":   [0.0, 0.35, 0.6, 0.8, 1.0],
        "{0,.2,.4,.6,.8,1} 6pt":     [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    print(f"    {'grid':28s} {'n/arm':>6s} {'crit':>7s} {'pw:DIST':>8s} {'pw:OBLIQ':>9s} {'pw:NEST':>8s}")
    for label, g in grids.items():
        cells = 1 + 2 * (len(g) - 1)
        n = max(40, int(budget_gens / (2 * cells)))
        mcs = {nm: run_mc(SC[nm], g, n, n, 1, reps, seed=len(label) * 13 + len(g))
               for nm in ("SHARED", "DISTINCT", "DISTINCT_OBLIQUE", "NESTED")}
        crit = float(np.quantile(mcs["SHARED"]["ang_logit"], 0.95))
        print(f"    {label:28s} {n:6d} {crit:7.1f} "
              f"{float((mcs['DISTINCT']['ang_logit'] > crit).mean()):8.3f} "
              f"{float((mcs['DISTINCT_OBLIQUE']['ang_logit'] > crit).mean()):9.3f} "
              f"{float((mcs['NESTED']['ang_logit'] > crit).mean()):8.3f}")


def report_failure_modes(reps, n=296, k=1):
    print("\n" + "=" * 100)
    print("G. FAILURE MODES: what makes ANG_L lie")
    print("=" * 100)
    base = SCENARIOS["SHARED"]

    print("\n  G1. CEILING/FLOOR on the harmful battery (SHARED truth -> angle should stay ~0)")
    print(f"      {'P_harm base':>12s} {'true ANG_L':>11s} {'obs 5/50/95':>26s} {'ANG_P 50th':>11s}")
    for ph0 in (0.99, 0.97, 0.90, 0.75, 0.55):
        sc = calibrate(replace(base, p_harm0=ph0))
        pop = pop_values(sc, GRID_UNIFORM)
        r = run_mc(sc, GRID_UNIFORM, n, n, k, reps, seed=int(ph0 * 1000))
        qq = qtiles(r["ang_logit"])
        print(f"      {ph0:12.2f} {pop['ang_logit'][0]:11.1f} "
              f"{qq[0]:8.1f}{qq[1]:8.1f}{qq[2]:8.1f}   {np.median(r['ang_prob']):11.1f}")

    print("\n  G2. WEAK DIRECTION (one direction barely moves anything) -> angle is noise")
    print(f"      {'align rho':>10s} {'|dPs|+|dPh| r_s':>16s} {'obs ANG_L 5/50/95':>26s}")
    for al in (0.70, 0.45, 0.30, 0.18, 0.10):
        lk = np.sqrt(1 - al**2)
        sc = calibrate(replace(base, r_stance=(al, 0.0, lk, 0.0)))
        r = run_mc(sc, GRID_UNIFORM, n, n, k, reps, seed=int(al * 997))
        mag = float(np.median(np.abs(r["ds_rs"]) + np.abs(r["dh_rs"])))
        qq = qtiles(r["ang_logit"])
        print(f"      {al:10.2f} {mag:16.3f} {qq[0]:8.1f}{qq[1]:8.1f}{qq[2]:8.1f}")

    print("\n  G3. INCOHERENCE rising with lambda, on top of a TRUE SHARED knob")
    print("      (generic damage contaminating a real shared effect)")
    print(f"      {'damage@l=1':>11s} {'true ANG_L':>11s} {'obs 5/50/95':>26s} {'rndANG_s':>9s} {'rndANG_h':>9s}")
    for dmg in (0.0, 0.10, 0.25, 0.50):
        sc = calibrate(replace(base, damage=dmg))
        pop = pop_values(sc, GRID_UNIFORM)
        r = run_mc(sc, GRID_UNIFORM, n, n, k, reps, seed=int(dmg * 555) + 11)
        qq = qtiles(r["ang_logit"])
        print(f"      {dmg:11.2f} {pop['ang_logit'][0]:11.1f} "
              f"{qq[0]:8.1f}{qq[1]:8.1f}{qq[2]:8.1f} "
              f"{pop['rand_ang_rs'][0]:9.1f} {pop['rand_ang_rh'][0]:9.1f}")

    print("\n  G4. NON-MONOTONE trajectory: damage pushes stance UP while the knob pushes it DOWN")
    print("      -> trajectory folds; origin-anchored PCA angle is misled.")
    for dmg in (0.0, 0.30):
        sc = calibrate(replace(base, damage=dmg))
        rng = np.random.default_rng(5)
        y, N, q = simulate_cells(sc, GRID_UNIFORM, None, 200_000, 200_000, 1, rng)
        P, _ = traj(y, N, q, GRID_UNIFORM, "r_stance", "prob")
        print(f"      damage={dmg:.2f}  dP_stance by lambda: "
              + " ".join(f"{v:+.3f}" for v in P[:, 0])
              + "   dP_harm: " + " ".join(f"{v:+.3f}" for v in P[:, 1]))


def report_random_control(SC, reps, n=296, k=1):
    print("\n" + "=" * 100)
    print("H. RANDOM-DIRECTION CONTROL (scenario (d) guard)")
    print("=" * 100)
    print(f"    {'scenario':18s} {'ANGvsRAND r_s (5/50/95)':>28s} {'ANGvsRAND r_h':>22s}"
          f" {'lenratio r_s':>13s} {'lenratio r_h':>13s}")
    for nm in ("SHARED", "DISTINCT", "GENERIC_DAMAGE", "NESTED"):
        r = run_mc(SC[nm], GRID_UNIFORM, n, n, k, reps, seed=len(nm) * 17, with_random=True)
        a = qtiles(r["rand_ang_rs"]); b = qtiles(r["rand_ang_rh"])
        print(f"    {nm:18s} {a[0]:9.1f}{a[1]:9.1f}{a[2]:9.1f}  "
              f"{b[0]:7.1f}{b[1]:7.1f}{b[2]:7.1f} "
              f"{np.median(r['rand_len_rs']):13.2f} {np.median(r['rand_len_rh']):13.2f}")


def report_sel_nonidentifiable(reps, n=296, k=1):
    """
    The critique, made quantitative: hold the world FIXED at SHARED (one knob) and
    vary only a nuisance parameter nobody can measure -- how well each direction
    estimates the shared axis. The population selectivity ratio sweeps across the
    fixed threshold 2 with no second mechanism anywhere.
    """
    print("\n" + "=" * 100)
    print("A2. WHY THE FIXED 'SEL >= 2' RULE IS NOT IDENTIFYING")
    print("    World is SHARED throughout. Only the direction-estimate quality varies.")
    print("=" * 100)
    base = SCENARIOS["SHARED"]
    print(f"    {'rho_stance':>10s} {'rho_harm':>9s} {'cos(rs,rh)':>11s} {'pop SEL':>9s}"
          f" {'P(SEL>=2)':>10s} {'pop ANG_L':>10s} {'P(ANG_L>crit0)':>15s}")
    crit0 = None
    for als, alh in ((0.55, 0.55), (0.55, 0.70), (0.55, 0.85), (0.40, 0.90),
                     (0.30, 0.95), (0.85, 0.55)):
        lks, lkh = np.sqrt(1 - als**2), np.sqrt(1 - alh**2)
        sc = calibrate(replace(base, r_stance=(als, 0.0, lks, 0.0),
                               r_harm=(alh, 0.0, 0.0, lkh)))
        pop = pop_values(sc, GRID_UNIFORM)
        r = run_mc(sc, GRID_UNIFORM, n, n, k, reps, seed=int(als * 100 + alh * 1000))
        if crit0 is None:
            crit0 = float(np.quantile(r["ang_logit"], 0.95))
        print(f"    {als:10.2f} {alh:9.2f} {als * alh:11.2f} {pop['sel'][0]:9.2f}"
              f" {float((r['sel'] >= 2).mean()):10.3f} {pop['ang_logit'][0]:10.1f}"
              f" {float((r['ang_logit'] > crit0).mean()):15.3f}")
    print("    (last row swaps which direction is well estimated -> SEL < 1, rule silently flips)")
    print(f"    crit0 = SHARED-calibrated ANG_L critical value from the first row = {crit0:.1f} deg")


def report_hard_grids(reps, budget=6000, mix=0.5):
    """Grid comparison in the regime that actually discriminates (power ~0.5)."""
    print("\n" + "=" * 100)
    print(f"F2. LAMBDA GRID at MATCHED BUDGET (~{budget} gens, k=1), HARD case: mix m={mix}")
    print("    (true angle ~16 deg; this is where grid choice can matter)")
    print("=" * 100)
    base = SCENARIOS["SHARED"]
    ws = (1 - mix, mix); nrm = np.hypot(*ws)
    hard = calibrate(replace(base, ws=ws,
                             r_stance=(_ALS * ws[0] / nrm, _ALS * ws[1] / nrm, _LKS, 0.0)))
    shared = calibrate(base)
    grids = {
        "{0,1}  (prior 2x2)":       [0.0, 1.0],
        "{0,.5,1}":                 [0.0, 0.5, 1.0],
        "{0,.25,.5,.75,1} uniform": [0.0, 0.25, 0.5, 0.75, 1.0],
        "{0,.5,.75,.9,1} dense-hi": [0.0, 0.5, 0.75, 0.9, 1.0],
        "{0,.35,.6,.8,1} log-ish":  [0.0, 0.35, 0.6, 0.8, 1.0],
        "{0,.2,.4,.6,.8,1} 6pt":    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    print(f"    {'grid':28s} {'n/arm':>6s} {'crit_deg':>9s} {'power(hard)':>12s} {'monoGate ok':>12s}")
    for label, g in grids.items():
        cells = 1 + 2 * (len(g) - 1)
        n = max(40, int(budget / (2 * cells)))
        r0 = run_mc(shared, g, n, n, 1, reps, seed=len(label) * 3 + 1)
        r1 = run_mc(hard, g, n, n, 1, reps, seed=len(label) * 3 + 2)
        crit = float(np.quantile(r0["ang_logit"], 0.95))
        mono = np.nan
        if len(g) > 2:
            rng = np.random.default_rng(9)
            ok = []
            for _ in range(60):
                y, N, q = simulate_cells(shared, g, None, n, n, 1, rng)
                ok.append(stat_gates(y, N, q, g)["monotone"])
            mono = float(np.mean(ok))
        print(f"    {label:28s} {n:6d} {crit:9.1f} "
              f"{float((r1['ang_logit'] > crit).mean()):12.3f} {mono:12.3f}")


def report_decision_rule(SC, reps, n=296, k=1, theta_eq=25.0, B=400, mag_min=4.0):
    """
    End-to-end evaluation of the RECOMMENDED composite rule.

    G1 positive control : dP_harm under r_harm at lambda=1 <= -0.15
    G2 precision        : BOTH real trajectories >= mag_min noise-SD long (logit plane)
    G3 specificity      : the covariance-matched RANDOM direction moves things by
                          < mag_min noise-SD (else nothing is attributable to
                          direction identity -- generic rank-1 damage)
    G4 coherence        : no sign reversal between the two DVs when both are large,
                          and each large component is monotone in lambda
    DECISION on ANG_L with an item-bootstrap 90% CI:
        CI_hi <  theta_eq  -> ONE KNOB (shared)     [equivalence]
        CI_lo >  theta_eq  -> NOT ONE KNOB          [difference]
        else               -> INCONCLUSIVE
    """
    print("\n" + "=" * 100)
    print(f"I. RECOMMENDED COMPOSITE DECISION RULE  (n={n}, k={k}, theta_eq={theta_eq} deg, "
          f"mag_min={mag_min} SD, bootstrap B={B})")
    print("=" * 100)
    print(f"    {'scenario':18s} {'G1':>6s} {'G2':>6s} {'G3':>6s} {'G4':>6s}"
          f" {'->SHARED':>9s} {'->NOT-1K':>9s} {'INCONCL':>8s} {'GATED':>7s}")
    rng = np.random.default_rng(2024)
    out = {}
    for nm, sc in SC.items():
        cnt = dict(g1=0, g2=0, g3=0, g4=0, shared=0, notone=0, inc=0, gated=0)
        R = max(reps // 3, 40)
        for _ in range(R):
            y, N, q = simulate_cells(sc, GRID_UNIFORM, None, n, n, k, rng,
                                     arms=("r_stance", "r_harm", "r_random"))
            g = stat_gates(y, N, q, GRID_UNIFORM, has_random=True)
            g1 = g["dPh_rharm"] <= -0.15
            g2 = min(g["z_r_stance"], g["z_r_harm"]) >= mag_min
            g3 = g["mag_random"] < mag_min
            g4 = g["same_quadrant"] > 0.5 and g["monotone"] > 0.5
            cnt["g1"] += g1; cnt["g2"] += g2; cnt["g3"] += g3; cnt["g4"] += g4
            if not (g1 and g2 and g3 and g4):
                cnt["gated"] += 1
                continue
            bs = angle_boot(y, N, q, GRID_UNIFORM, B=B, rng=rng)
            lo, hi = np.quantile(bs, [0.05, 0.95])
            if hi < theta_eq:
                cnt["shared"] += 1
            elif lo > theta_eq:
                cnt["notone"] += 1
            else:
                cnt["inc"] += 1
        out[nm] = {kk: v / R for kk, v in cnt.items()}
        o = out[nm]
        print(f"    {nm:18s} {o['g1']:6.2f} {o['g2']:6.2f} {o['g3']:6.2f} {o['g4']:6.2f}"
              f" {o['shared']:9.2f} {o['notone']:9.2f} {o['inc']:8.2f} {o['gated']:7.2f}")
    print("\n    Correct verdicts: SHARED->SHARED, DISTINCT/DISTINCT_OBLIQUE/NESTED->NOT-1K,")
    print("    GENERIC_DAMAGE and POSCTRL_FAIL -> GATED (no claim made).")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 120 if a.quick else a.reps

    SC = {nm: calibrate(sc) for nm, sc in SCENARIOS.items()}
    print("Calibrated thresholds (baseline P_stance=0.35, P_harm=0.90):")
    for nm, sc in SC.items():
        print(f"   {nm:18s} th_s={sc.th_s:+.3f}  th_h={sc.th_h:+.3f}")

    report_population(SC)
    report_sel_nonidentifiable(reps)
    mc = report_sampling(SC, reps, 296, 296, 1)
    report_discrimination(mc)
    report_power(SC, reps)
    report_intermediate(reps)
    report_grids(SC, reps)
    report_hard_grids(reps)
    report_failure_modes(reps)
    report_random_control(SC, reps)
    for (nn, kk) in ((296, 1), (296, 3), (296, 5), (150, 5)):
        report_decision_rule(SC, reps, n=nn, k=kk)


if __name__ == "__main__":
    main()
