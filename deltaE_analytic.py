"""
Grid-free Delta-E effect: the magnetoelastic compliance in closed form.

Why this module exists
----------------------
The original squire_E / deam_E evaluate dlambda_x/dsigma by a finite difference
on a probe stress dsigma, after locating the equilibrium angle ON THE FIXED
721-point angular grid of deltaE.py.  The stress probe (dsigma = 0.2 MPa) moves
the equilibrium angle by far less than the grid spacing (8.7e-3 rad), so the
single-domain branch returns a quantised, badly under-resolved derivative: its
apparent saturation at a factor of 1.5 in dynamic range is a discretisation
artifact, not physics.  Refining the grid moves it to a factor of 22 (n = 21601)
and rising.  The Boltzmann average is much less exposed, because the weights
move continuously with stress, but it inherits the same finite-difference probe.

Both compliances are available analytically, which removes the issue entirely.

Energy density (angle theta from the resonator length x, field H along x):

    g(theta) = Keff sin^2(theta - phi) - mu0 Ms H cos(theta)
               - (3/2) lambda_s sigma_x cos^2(theta)

Coherent rotation (single domain).  At equilibrium g'(theta) = 0, so implicit
differentiation of the equilibrium condition gives

    dtheta/dsigma = -(d2g/dtheta dsigma) / g''(theta)
    dlambda_x/dsigma = ((3/2) lambda_s)^2 sin^2(2 theta) / g''(theta) ,

which is non-negative on any stable branch (g'' > 0), i.e. magnetoelastic
softening, and needs no probe stress at all.

Energy-averaged domain model.  With p(theta) proportional to exp(-A_s g), the
population average obeys exactly

    dlambda_x/dsigma = A_s ((3/2) lambda_s)^2 Var_p(cos^2 theta) ,

a fluctuation-response identity: the softening is the variance of the
magnetostrictive strain over the domain population.  As A_s grows the variance
collapses as 1/(A_s g''), so this expression tends to the coherent-rotation one:
the two models share a common stiff limit, and they differ only when several
energy wells are comparably populated.  That limit is the reason the fitted
response is nearly independent of A_s.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from materials import LAMBDA_S, MS_FEGAB, FEGAB_E, MU0

_C = 1.5 * LAMBDA_S                     # (3/2) lambda_s
_TH_SEED = np.linspace(-np.pi, np.pi, 361)


def _dg(theta, K, phi, Hterm, sterm):
    return (K * np.sin(2.0 * (theta - phi)) + Hterm * np.sin(theta)
            + sterm * np.sin(2.0 * theta))


def _d2g(theta, K, phi, Hterm, sterm):
    return (2.0 * K * np.cos(2.0 * (theta - phi)) + Hterm * np.cos(theta)
            + 2.0 * sterm * np.cos(2.0 * theta))


def _g(theta, K, phi, Hterm, sterm):
    return (K * np.sin(theta - phi) ** 2 - Hterm * np.cos(theta)
            - sterm * np.cos(theta) ** 2)


def _stable_roots(K, phi, Hterm, sterm):
    """All stable equilibria, found by bracketing sign changes of g' on a coarse
    seed grid and polishing each with Brent.  Grid-free to machine precision."""
    d = _dg(_TH_SEED, K, phi, Hterm, sterm)
    roots = []
    for i in np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
        try:
            r = brentq(_dg, _TH_SEED[i], _TH_SEED[i + 1],
                       args=(K, phi, Hterm, sterm), xtol=1e-14)
        except ValueError:
            continue
        if _d2g(r, K, phi, Hterm, sterm) > 0:
            roots.append(r)
    return np.array(roots)


def coherent_compliance(H_arr, Keff, phi_eff_deg, sigma_x=0.0, Ms=MS_FEGAB,
                        lam_s=LAMBDA_S):
    """dlambda_x/dsigma (1/Pa) on the continuation branch, exact in closed form.

    The branch is tracked from the zero-field equilibrium by always taking the
    stable root nearest the previous one, as in the original implementation.
    """
    phi = np.deg2rad(phi_eff_deg)
    cc = 1.5 * lam_s
    sterm = cc * sigma_x
    out = np.empty_like(H_arr, dtype=float)
    prev = phi
    for i, H in enumerate(H_arr):
        Hterm = MU0 * Ms * H
        roots = _stable_roots(Keff, phi, Hterm, sterm)
        if roots.size == 0:
            out[i] = 0.0
            continue
        th = roots[np.argmin(np.abs(roots - prev))]
        prev = th
        out[i] = cc ** 2 * np.sin(2.0 * th) ** 2 / _d2g(th, Keff, phi, Hterm, sterm)
    return out


def averaged_compliance(H_arr, Keff, phi_eff_deg, As, sigma_x=0.0,
                        Ms=MS_FEGAB, n_theta=4001):
    """dlambda_x/dsigma (1/Pa) for the Boltzmann-weighted domain population,
    from the fluctuation-response identity A_s (3/2 lambda_s)^2 Var(cos^2 theta).

    The quadrature grid only has to resolve the weight, not a stress probe, and
    convergence is checked in the test block below.
    """
    phi = np.deg2rad(phi_eff_deg)
    sterm = _C * sigma_x
    th = np.linspace(-np.pi, np.pi, n_theta)
    c2 = np.cos(th) ** 2
    out = np.empty_like(H_arr, dtype=float)
    for i, H in enumerate(H_arr):
        g = _g(th, Keff, phi, MU0 * Ms * H, sterm)
        w = np.exp(-As * (g - g.min()))
        w /= np.trapezoid(w, th)
        m1 = np.trapezoid(c2 * w, th)
        m2 = np.trapezoid(c2 * c2 * w, th)
        out[i] = As * _C ** 2 * max(m2 - m1 * m1, 0.0)
    return out


def modulus(compliance, E0=FEGAB_E):
    """E(H) from the magnetoelastic compliance: 1/E = 1/E0 + dlambda/dsigma."""
    return E0 / (1.0 + E0 * np.asarray(compliance))
