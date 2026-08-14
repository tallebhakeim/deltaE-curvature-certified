"""
Paper-fitted effective-anisotropy functions Keff(kappa), phi_eff(kappa).

The paper does NOT feed the raw FEM field into the domain-phase model.  On
page 12 it states explicitly: "the mean phi_eff is fitted with a power function
and Keff with a linear function ... These will be used in the domain-phase
model."  We reproduce those fitted functions here (Fig 9d):

  * Keff(kappa) is LINEAR, rising from ~0.8 kJ/m^3 (Ku-dominated, flat plate)
    to ~12 kJ/m^3 at kappa = 7 mm^-1.
  * phi_eff(kappa) decreases from 90 deg (easy axis along y, set by Ku) and
    PLATEAUS near 60 deg once the stress anisotropy dominates -- the plateau
    reflects the neutral axis sitting at a fixed fraction of the magnetic layer,
    so the lengthwise/widthwise volume ratio is invariant to kappa.

Our own laminated-plate FEM (mechanics.py + magnetoelastic.py) reproduces the
SHAPE of both trends (linear Keff, phi_eff starting at 90 deg and plateauing);
these closed forms fix the quantitative scale that a free-plate 1D relaxation
under-retains, exactly as the paper's COMSOL fit does per device.
"""
from __future__ import annotations
import numpy as np

# Keff(kappa): linear (kJ/m^3), Keff(0)=0.8, Keff(7)=12
_KEFF0 = 0.8
_KEFF_SLOPE = (12.0 - 0.8) / 7.0     # ~1.6 kJ/m^3 per mm^-1

# phi_eff(kappa): 90 deg -> plateau at PHI_INF, transition scale KAPPA_C
_PHI_INF = 60.0
_PHI_SPAN = 30.0                     # 90 - 60
_KAPPA_C = 0.35                      # mm^-1, rapid initial rotation


def keff_fit(kappa_mm):
    """Effective anisotropy energy density (kJ/m^3) vs curvature (mm^-1)."""
    return _KEFF0 + _KEFF_SLOPE * np.asarray(kappa_mm)


def phi_eff_fit(kappa_mm):
    """Effective easy-axis angle (deg from x) vs curvature (mm^-1): power-like
    decay from 90 deg plateauing at 60 deg."""
    k = np.asarray(kappa_mm, dtype=float)
    return _PHI_INF + _PHI_SPAN * (_KAPPA_C / (_KAPPA_C + k))
