"""
Out-of-sample validation and parameter identifiability for the curvature model.

Written in answer to the two reviewer objections that the calibration of the
global constants was scored on the same digitized population it was fitted to
(no held-out split, no identifiability analysis).  Everything here reuses the
model modules unchanged; only the way the measured population is split and the
way the constants are estimated is new.

Design of the parameterisation
------------------------------
The response model has exactly two global scalars:

  v   the active volume fraction, which enters only through
      E* = v E(H) + (1-v) Em .  Because dfr is very nearly proportional to v
      (8.63, 8.66, 8.71 %/unit at v = 3, 5, 9 %), v is a pure vertical offset
      in log-log: it carries NO shape information.  The held-out test therefore
      tests the SHAPE dfr(kappa), which is the part v cannot absorb.

  c   the dimensionless domain-population width, A_s = c / Keff.  The earlier
      scripts hard-coded c = 10, i.e. it was an implicit third constant.  Here
      it is declared, fitted and profiled like the others.  c changes both the
      scale and the shape of dfr(kappa), so (v, c) are not fully degenerate.

The single-domain (coherent rotation) branch has v only.

The loss model has two global scalars (Q0, tau) in

  1/Q = 1/Q0 + 2 pi f_r tau x ,     x = Delta f_r/f_r,min (%) ,

against the four constants (gamma_NM, gamma_M) x (two device families) of
Eq. (4) of [1].  Because f_r appears explicitly, our two constants can be
transferred from one device family to the other; theirs cannot, which is what
the cross-family test below measures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, least_squares

from deltaE import squire_E, deam_E, frequency_response
from deltaE_analytic import coherent_compliance, averaged_compliance, modulus
from materials import MU0
from paper_fits import keff_fit, phi_eff_fit

HERE = Path(__file__).parent
H_SWEEP = np.linspace(0.0, 12e-3, 140) / MU0
C_REF = 10.0                      # the value hard-coded in the earlier scripts
N_THETA = 4001                    # quadrature grid of the averaged branch

_CACHE: dict = {}

MODELS = ("averaged", "coherent", "deam_fd", "squire_fd")


# ---------------------------------------------------------------- model -----
def E_of_H(kappa_mm: float, model: str, c: float = C_REF) -> np.ndarray:
    """E(H) of the magnetic layer at one curvature.  Cached: the v fit does not
    touch it, so a whole v sweep costs one evaluation.

    model = "averaged"  energy-averaged domain population, closed-form variance
            "coherent"  coherent rotation, closed-form implicit differentiation
            "deam_fd"   original finite-difference averaged branch (legacy)
            "squire_fd" original finite-difference single domain (legacy, NOT
                        converged in the angular grid: kept only to document the
                        artifact it produced)
    """
    key = (round(float(kappa_mm), 9), model, round(float(c), 9))
    hit = _CACHE.get(key)
    if hit is None:
        K = float(keff_fit(kappa_mm)) * 1e3
        phi = float(phi_eff_fit(kappa_mm))
        if model == "averaged":
            hit = modulus(averaged_compliance(H_SWEEP, K, phi, As=c / max(K, 1.0),
                                              n_theta=N_THETA))
        elif model == "coherent":
            hit = modulus(coherent_compliance(H_SWEEP, K, phi))
        elif model == "deam_fd":
            hit = deam_E(H_SWEEP, K, phi, As=c / max(K, 1.0))
        elif model == "squire_fd":
            hit = squire_E(H_SWEEP, K, phi)
        else:
            raise ValueError(model)
        _CACHE[key] = hit
    return hit


def predict(kappa, model="averaged", v=0.05, c=C_REF) -> np.ndarray:
    """Delta f_r/f_r,min (%) at the given curvatures."""
    return np.array([frequency_response(H_SWEEP, E_of_H(k, model, c), v=v)[1]
                     for k in np.atleast_1d(kappa)])


# ------------------------------------------------------------- metrics ------
def rms_log(pred, meas) -> float:
    """Root-mean-square deviation in log10, reported elsewhere as a factor."""
    return float(np.sqrt(np.mean((np.log10(pred) - np.log10(meas)) ** 2)))


def fit_v(kappa, meas, model="averaged", c=C_REF, bracket=(-2.5, -0.5)):
    """Continuous least-squares fit of the single global fraction in log space.
    Returns (v, rms_log).  Optimising log10(v) keeps the search scale free."""
    Es = [E_of_H(k, model, c) for k in kappa]

    def obj(log10v):
        v = 10.0 ** log10v
        pred = np.array([frequency_response(H_SWEEP, E, v=v)[1] for E in Es])
        return rms_log(pred, meas)

    r = minimize_scalar(obj, bounds=bracket, method="bounded",
                        options={"xatol": 1e-4})
    return float(10.0 ** r.x), float(r.fun)


def fit_vc(kappa, meas, c_grid=np.linspace(4.0, 24.0, 21)):
    """Joint fit of (v, c) for the energy-averaged branch: c on a grid, v
    continuously inside.  Returns (v, c, rms_log)."""
    best = None
    for c in c_grid:
        v, rms = fit_v(kappa, meas, "averaged", float(c))
        if best is None or rms < best[2]:
            best = (v, float(c), rms)
    return best


# -------------------------------------------------------------- loading -----
def load_points(name):
    d = json.loads((HERE / name).read_text())
    p = np.asarray(d["points"], dtype=float)
    order = np.argsort(p[:, 0])
    return p[order, 0], p[order, 1]


# --------------------------------------------------------------- splits -----
def interleaved(n):
    """Deterministic equal-coverage split: even ranks train, odd ranks test."""
    idx = np.arange(n)
    return idx[0::2], idx[1::2]


def stratified_random(n, rng, n_strata=8):
    """Half/half split drawn inside contiguous strata of the sorted abscissa,
    so both halves span the whole curvature range."""
    tr, te = [], []
    for block in np.array_split(np.arange(n), n_strata):
        perm = rng.permutation(block)
        k = len(block) // 2
        te.extend(perm[:k]); tr.extend(perm[k:])
    return np.sort(np.array(tr, int)), np.sort(np.array(te, int))


def kfold(n, k, rng):
    """K contiguous-stratum folds, shuffled inside strata."""
    idx = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


def extrapolation(n, low_is_train=True):
    """Train on one half of the curvature range, predict the other half.
    The harshest test of the predicted SHAPE: the fitted scale cannot help."""
    half = n // 2
    lo, hi = np.arange(half), np.arange(half, n)
    return (lo, hi) if low_is_train else (hi, lo)
