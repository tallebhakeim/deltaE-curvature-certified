"""
Delta-E effect: field-dependent Young's modulus E(H) and the resonance-frequency
response, computed from the effective anisotropy (Keff, phi_eff).

Two magnetization models are provided:

  1. squire_E(H)  -- coherent-rotation / Squire domain-phase model (the paper's
     approach, Sec 2.3.4).  A single effective easy axis at phi_eff with energy
     density Keff; magnetization rotates coherently under the field.

  2. deam_E(H)    -- INNOVATION: an energy-averaged (Boltzmann-weighted) model
     over a continuous distribution of domain orientations.  No arbitrary active
     fraction v is needed to soften the response: the distribution width A_s
     captures the dispersion of local easy axes that the paper lumps into v=8%.

Both return the Young's modulus along the resonator length (x = applied-field
axis) as a function of the DC field, via the magnetoelastic compliance

    1/E(H) = 1/E0 + dlambda_x/dsigma_x |_H .

The longitudinal magnetostriction along x for magnetization at angle theta is
    lambda_x(theta) = (3/2) lambda_s (cos^2 theta - 1/3).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from materials import (LAMBDA_S, MS_FEGAB, FEGAB_E, MU0, F_R0,
                       default_stack)

_TH = np.linspace(-np.pi, np.pi, 721)     # domain-orientation grid


def _lambda_x(theta):
    return 1.5 * LAMBDA_S * (np.cos(theta) ** 2 - 1.0 / 3.0)


def _g(theta, Keff, phi_eff_rad, H, sigma_x, Ms=MS_FEGAB):
    """Total in-plane energy density (J/m^3). H in A/m along x, sigma_x in Pa."""
    return (Keff * np.sin(theta - phi_eff_rad) ** 2
            - MU0 * Ms * H * np.cos(theta)
            - 1.5 * LAMBDA_S * sigma_x * np.cos(theta) ** 2)


# --------------------------------------------------------------------------
#  Model 1: coherent rotation (Squire domain-phase, single effective axis)
# --------------------------------------------------------------------------
def _theta_eq_coherent(Keff, phi, H, sigma_x, theta_prev):
    """Equilibrium angle closest to theta_prev (continuation for hysteresis-free
    branch tracking)."""
    g = _g(_TH, Keff, phi, H, sigma_x)
    # restrict to local minima, then pick the one nearest theta_prev
    dg = np.gradient(g)
    mins = np.where((np.r_[dg[1:], dg[-1]] > 0) & (np.r_[dg[0], dg[:-1]] < 0))[0]
    if len(mins) == 0:
        return _TH[np.argmin(g)]
    cand = _TH[mins]
    return cand[np.argmin(np.abs(cand - theta_prev))]


def squire_E(H_arr, Keff, phi_eff_deg, E0=FEGAB_E, dsig=2e5):
    """E(H) from coherent rotation. Returns modulus array (Pa)."""
    phi = np.deg2rad(phi_eff_deg)
    E = np.empty_like(H_arr, dtype=float)
    theta_prev = np.deg2rad(phi_eff_deg)
    for i, H in enumerate(H_arr):
        th0 = _theta_eq_coherent(Keff, phi, H, 0.0, theta_prev)
        thp = _theta_eq_coherent(Keff, phi, H, +dsig, th0)
        thm = _theta_eq_coherent(Keff, phi, H, -dsig, th0)
        dldsig = (_lambda_x(thp) - _lambda_x(thm)) / (2.0 * dsig)
        E[i] = E0 / (1.0 + E0 * dldsig)
        theta_prev = th0
    return E


# --------------------------------------------------------------------------
#  Model 2: DEAM -- energy-averaged over a distribution of domain orientations
# --------------------------------------------------------------------------
def deam_E(H_arr, Keff, phi_eff_deg, E0=FEGAB_E, As=None, dsig=2e5):
    """
    Energy-averaged Delta-E.  The domain population at orientation theta is
    p(theta) ~ exp(-As * g(theta)); observables are population averages.  As
    (m^3/J) sets the smoothing: large As -> single-domain (Squire) limit, small
    As -> strongly averaged (soft) response.  Physically As ~ 1/(k_B T / V_act)
    i.e. an effective activation volume; here it is the single knob that REPLACES
    the paper's ad-hoc active fraction v.
    """
    if As is None:
        # scale so that As * Keff ~ O(10): comparable weighting across models
        As = 10.0 / max(Keff, 1.0)
    phi = np.deg2rad(phi_eff_deg)
    E = np.empty_like(H_arr, dtype=float)

    def lam_avg(H, sigma_x):
        g = _g(_TH, Keff, phi, H, sigma_x)
        w = np.exp(-As * (g - g.min()))
        w /= np.trapz(w, _TH) if hasattr(np, "trapz") else np.trapezoid(w, _TH)
        return np.trapezoid(_lambda_x(_TH) * w, _TH)

    for i, H in enumerate(H_arr):
        lp = lam_avg(H, +dsig)
        lm = lam_avg(H, -dsig)
        dldsig = (lp - lm) / (2.0 * dsig)
        E[i] = E0 / (1.0 + E0 * dldsig)
    return E


# --------------------------------------------------------------------------
#  From E(H) to the resonance-frequency response
# --------------------------------------------------------------------------
def _composite_Eeq(E_mag, t_fegab):
    """Voigt (thickness-weighted) equivalent modulus of the Pt/AlN/FeGaB stack
    with the magnetic layer at modulus E_mag."""
    layers = default_stack(t_fegab)
    num = 0.0; den = 0.0
    for l in layers:
        Ei = E_mag if l.name == "FeGaB" else l.E
        num += Ei * l.thickness
        den += l.thickness
    return num / den


def frequency_response(H_arr, E_H, t_fegab=500e-9, v=1.0, Em=FEGAB_E,
                       f_r0=F_R0):
    """
    Convert E(H) of the active magnetic fraction into f_r(H) and the paper's
    metric Delta f_r / f_r,min (%).  E*(H) = v E(H) + (1-v) Em is the magnetic
    layer modulus (paper Sec 2.3.4); f_r ~ sqrt(Eeq) (eq 1).
    With the DEAM model v may be left at 1.0 (no ad-hoc fraction).
    """
    Estar = v * E_H + (1.0 - v) * Em
    Eeq = np.array([_composite_Eeq(Ei, t_fegab) for Ei in Estar])
    fr = f_r0 * np.sqrt(Eeq / _composite_Eeq(Em, t_fegab))
    fr_min, fr_max = fr.min(), fr.max()
    dfr = (fr_max - fr_min) / fr_min * 100.0
    return fr, dfr


@dataclass
class DeltaEResult:
    H: np.ndarray          # A/m
    E: np.ndarray          # Pa
    fr: np.ndarray         # Hz
    dfr_pct: float         # Delta f_r / f_r,min (%)


# --------------------------------------------------------------------------
#  Through-thickness INTEGRATED Delta-E  (physical origin of the small v)
# --------------------------------------------------------------------------
def magnetic_modulus_profile(H_arr, profile, model="squire", As=None):
    """
    Instead of collapsing the stress anisotropy to a single (Keff, phi_eff),
    compute E_slice(H) for every through-thickness slice z using its LOCAL
    (Keff(z), phi_eff(z)) and average the modulus over the magnetic thickness
    (Voigt / parallel springs).

    Slices below the neutral axis have their easy axis toward y while slices
    above have it toward x; their magnetoelastic softening along the vibration
    axis (x) therefore partially CANCELS.  This cancellation is the physical
    reason the paper needs an active fraction v ~ 8% -- here it emerges from the
    stress profile with no fitting fraction.
    """
    Efun = squire_E if model == "squire" else deam_E
    z = profile.z
    Emag = np.zeros_like(H_arr, dtype=float)
    for j in range(len(z)):
        if model == "deam":
            Ej = deam_E(H_arr, profile.Keff[j], profile.phi_eff[j], As=As)
        else:
            Ej = squire_E(H_arr, profile.Keff[j], profile.phi_eff[j])
        Emag += Ej
    Emag /= len(z)
    return Emag


def frequency_response_profile(H_arr, profile, t_fegab=500e-9, model="squire",
                               As=None, f_r0=F_R0):
    """f_r(H) and Delta f_r/f_r,min from the through-thickness integrated E(H)."""
    Emag = magnetic_modulus_profile(H_arr, profile, model=model, As=As)
    return frequency_response(H_arr, Emag, t_fegab, v=1.0, f_r0=f_r0)
