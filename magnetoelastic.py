"""
Magnetoelastic anisotropy from the through-thickness stress field.

Implements the paper's Sec 2.3.2-2.3.3: from the equilibrium stress sigma(z)
in the magnetic layer, compute
  * the stress-induced anisotropy energy density  Ksigma(z)
  * the stress-induced easy-axis angle            phi_sigma(z)
and, combined with the field-induced uniaxial anisotropy Ku (easy axis along y),
the effective anisotropy Keff(z), phi_eff(z).  Volume averages over the magnetic
thickness give Keff(kappa), phi_eff(kappa) that feed the Delta-E domain model.

Convention: angle alpha measured from the x-axis (resonator length, = applied
DC field direction).  For lambda_s > 0 the magnetization prefers the maximally
tensile in-plane direction, so

    E_me(alpha) = -(3/2) lambda_s [ sxx cos^2 a + syy sin^2 a + 2 sxy sin a cos a ]

The field-induced easy axis is along y (deposition bias field, Sec 2.3.3):

    E_u(alpha) = Ku cos^2(alpha)      (minimum at alpha = 90 deg)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from materials import LAMBDA_S, KU
from mechanics import StressState

_ALPHA = np.linspace(0.0, np.pi, 361)   # 0.5 deg resolution over [0, pi)


def _energy_me(a, sxx, syy, sxy):
    # Sign of the shear cross-term follows O'Handley's convention so that a
    # positive s12 rotates the easy axis toward +45 deg (paper Sec 2.3.3:
    # "shear stress acts to rotate the magnetic easy axis to 45 deg"), landing
    # phi_eff near 60 deg from x at large kappa as in Fig 9d.
    return -1.5 * LAMBDA_S * (sxx * np.cos(a) ** 2
                              + syy * np.sin(a) ** 2
                              - 2.0 * sxy * np.sin(a) * np.cos(a))


@dataclass
class AnisoProfile:
    z: np.ndarray
    Ksigma: np.ndarray      # J/m^3
    phi_sigma: np.ndarray   # deg, from x-axis
    Keff: np.ndarray        # J/m^3, with Ku included
    phi_eff: np.ndarray     # deg
    # volume (thickness) means
    Ksigma_mean: float
    phi_sigma_mean: float
    Keff_mean: float
    phi_eff_mean: float
    # volume-mean EFFECTIVE anisotropy (energy summed over z, then axis
    # extracted) -- the paper-faithful single (Keff, phi_eff) that plateaus
    Keff_vol: float
    phi_eff_vol: float


def _circular_mean_deg(angles_deg, weights=None):
    """Mean of axial (mod 180 deg) directions."""
    a = np.deg2rad(np.asarray(angles_deg) * 2.0)   # axial -> double angle
    if weights is None:
        weights = np.ones_like(a)
    s = np.average(np.sin(a), weights=weights)
    c = np.average(np.cos(a), weights=weights)
    m = 0.5 * np.rad2deg(np.arctan2(s, c))
    return m % 180.0


def anisotropy_profile(st: StressState, ku: float = KU) -> AnisoProfile:
    """Compute Ksigma(z), phi_sigma(z), Keff(z), phi_eff(z) and their means."""
    nz = len(st.z)
    Ksig = np.empty(nz); phis = np.empty(nz)
    Keff = np.empty(nz); phie = np.empty(nz)
    for j in range(nz):
        e_me = _energy_me(_ALPHA, st.sxx[j], st.syy[j], st.sxy[j])
        Ksig[j] = e_me.max() - e_me.min()
        phis[j] = np.rad2deg(_ALPHA[np.argmin(e_me)])
        e_tot = e_me + ku * np.cos(_ALPHA) ** 2
        Keff[j] = e_tot.max() - e_tot.min()
        phie[j] = np.rad2deg(_ALPHA[np.argmin(e_tot)])

    # Volume-mean EFFECTIVE anisotropy: sum the magnetoelastic energy landscape
    # over the magnetic thickness (equivalently, use the thickness-averaged
    # stress tensor), add Ku, THEN extract the effective easy axis.  This is the
    # right way to obtain a single (Keff, phi_eff) for the whole layer and it
    # plateaus once the stress anisotropy dominates Ku -- unlike a circular mean
    # of the per-slice angles, which is degenerate for a bimodal 0/90 split.
    sxx_m = float(np.mean(st.sxx)); syy_m = float(np.mean(st.syy))
    sxy_m = float(np.mean(st.sxy))
    e_vol = _energy_me(_ALPHA, sxx_m, syy_m, sxy_m) + ku * np.cos(_ALPHA) ** 2
    Keff_vol = float(e_vol.max() - e_vol.min())
    phi_eff_vol = float(np.rad2deg(_ALPHA[np.argmin(e_vol)]))

    return AnisoProfile(
        z=st.z, Ksigma=Ksig, phi_sigma=phis, Keff=Keff, phi_eff=phie,
        Ksigma_mean=float(np.mean(Ksig)),
        phi_sigma_mean=_circular_mean_deg(phis),
        Keff_mean=float(np.mean(Keff)),
        phi_eff_mean=_circular_mean_deg(phie),
        Keff_vol=Keff_vol, phi_eff_vol=phi_eff_vol,
    )


def keff_phieff_vs_curvature(sweep, ku: float = KU):
    """
    From a curvature_sweep() result, build arrays
        kappa (mm^-1), Keff_mean (kJ/m^3), phi_eff_mean (deg from x).
    Reproduces the trend of Fig 9d.
    """
    kappa = []; Keff = []; phieff = []
    profiles = []
    for k_y, st, sc in sweep:
        p = anisotropy_profile(st, ku=ku)
        kappa.append(k_y / 1000.0)          # 1/m -> 1/mm
        Keff.append(p.Keff_vol / 1000.0)    # J/m^3 -> kJ/m^3 (paper-faithful)
        phieff.append(p.phi_eff_vol)
        profiles.append(p)
    return np.array(kappa), np.array(Keff), np.array(phieff), profiles
