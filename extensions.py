"""
INNOVATION axes 2 and 4.

Axis 2 -- CERTIFIED curvature -> performance bounds.
  The residual stress of a sputtered film is not known exactly; it is only known
  to lie in a box  sigma0 in [center - dw, center + dw]  (component-wise).  The
  lamination map sigma0 -> curvature is LINEAR, so the curvature interval is
  obtained EXACTLY (no sampling) from the componentwise sensitivities.  The
  response Delta f_r/f_r,min(kappa) is monotone decreasing, so a guaranteed band
  [response(kappa_hi), response(kappa_lo)] encloses the achievable performance.
  This is a Prager-Synge-style two-sided certificate on device performance from a
  stress-uncertainty box -- something the paper's single-fit COMSOL model cannot
  provide.

Axis 4 -- INVERSE DESIGN: stress-compensation layer that flattens the plate.
  Because curvature is linear in each layer's initial stress, the compensation
  stress in an added capping layer that drives kappa_y -> 0 is the solution of a
  linear equation (closed form).  A flat plate maximizes the Delta-E response.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from materials import IsotropicLayer, default_stack
from mechanics import relax


# --------------------------------------------------------------------------
#  Axis 2 : certified curvature bounds from a stress-uncertainty box
# --------------------------------------------------------------------------
def kappa_sensitivity(layers, base_sigma0, comp_names=("FeGaB",)):
    """Return the linear map coefficients d kappa_y / d sigma0_c (1/m per Pa)
    for each stress component c of each layer in comp_names, exploiting the
    linearity of the lamination solve."""
    st0 = relax(layers, base_sigma0)
    k0 = st0.kappa_y
    grads = {}
    unit = 1.0e6  # 1 MPa probe
    for name in comp_names:
        for ci, comp in enumerate(("s11", "s22", "s12")):
            s = {k: list(v) for k, v in base_sigma0.items()}
            s.setdefault(name, [0.0, 0.0, 0.0])
            s[name][ci] += unit
            s = {k: tuple(v) for k, v in s.items()}
            st = relax(layers, s)
            grads[(name, comp)] = (st.kappa_y - k0) / unit
    return k0, grads


def certified_kappa_band(layers, center_sigma0, halfwidth,
                         comp_names=("FeGaB",)):
    """
    center_sigma0: dict layer -> (s11,s22,s12) nominal stress.
    halfwidth:     dict (layer, comp) -> +/- uncertainty (Pa), or a scalar frac
                   applied to |center| of the FeGaB stress.
    Returns (kappa_center, kappa_lo, kappa_hi) in 1/m -- GUARANTEED enclosure.
    """
    k0, grads = kappa_sensitivity(layers, center_sigma0, comp_names)
    if np.isscalar(halfwidth):
        hw = {}
        for name in comp_names:
            for ci, comp in enumerate(("s11", "s22", "s12")):
                hw[(name, comp)] = halfwidth * abs(center_sigma0[name][ci])
    else:
        hw = halfwidth
    spread = sum(abs(g) * hw.get(key, 0.0) for key, g in grads.items())
    return k0, k0 - spread, k0 + spread


def certified_response_band(kappa_lo, kappa_hi, kappa_grid, dfr_grid):
    """Map a curvature interval to a guaranteed response band using the monotone
    decreasing response curve dfr(kappa)."""
    order = np.argsort(kappa_grid)
    kg, dg = np.asarray(kappa_grid)[order], np.asarray(dfr_grid)[order]
    kl, kh = abs(kappa_lo) / 1000.0, abs(kappa_hi) / 1000.0   # 1/m -> 1/mm
    lo, hi = min(kl, kh), max(kl, kh)
    dfr_hi = np.interp(lo, kg, dg)   # smaller curvature -> larger response
    dfr_lo = np.interp(hi, kg, dg)
    return dfr_lo, dfr_hi


# --------------------------------------------------------------------------
#  Axis 4 : inverse design of a stress-compensation capping layer
# --------------------------------------------------------------------------
@dataclass
class CompensationDesign:
    comp_layer: IsotropicLayer
    comp_stress_biaxial: float   # Pa (s11=s22) needed to flatten the plate
    kappa_before: float          # 1/m
    kappa_after: float           # 1/m


def design_compensation_layer(base_sigma0, t_fegab=500e-9,
                              comp_thickness=100e-9,
                              comp_E=70e9, comp_nu=0.17, comp_rho=2200.0):
    """
    Add a thin capping layer (default: a SiO2-like film) on top and find the
    biaxial initial stress it must carry so the width-direction curvature
    kappa_y vanishes.  Linear in the compensation stress -> closed form.
    """
    base_layers = default_stack(t_fegab)
    comp = IsotropicLayer("Comp", comp_thickness, comp_E, comp_nu, comp_rho)
    layers = base_layers + [comp]

    st_before = relax(layers, {**base_sigma0, "Comp": (0.0, 0.0, 0.0)})
    k_before = st_before.kappa_y
    # sensitivity of kappa_y to a unit biaxial comp stress
    st_probe = relax(layers, {**base_sigma0, "Comp": (1e6, 1e6, 0.0)})
    dkd = (st_probe.kappa_y - k_before) / 1e6
    sigma_comp = -k_before / dkd
    st_after = relax(layers, {**base_sigma0,
                              "Comp": (sigma_comp, sigma_comp, 0.0)})
    return CompensationDesign(comp, sigma_comp, k_before, st_after.kappa_y)
