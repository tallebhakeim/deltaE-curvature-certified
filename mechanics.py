"""
Laminated-plate mechanics: residual-stress relaxation -> plate curvature and
through-thickness stress field.

This is the Python-autonomous stand-in for the paper's COMSOL magnetoelastic
FEM (Sec 2.3.2).  We use Classical Lamination Theory (CLT):

  * each layer i carries a stress-free "initial stress" sigma0_i =
    (s11, s22, s12) (the residual stress it would hold if clamped perfectly
    flat), representing sputter-deposition residual stress;
  * the free (unsupported after release) plate relaxes to a uniform mid-plane
    strain eps0 = (ex, ey, exy) and curvature kappa = (kx, ky, kxy) such that
    the net in-plane force and bending moment vanish:

        [ A  B ] [eps0 ]   [ N* ]
        [ B  D ] [kappa] = [ M* ]

    with N* = sum_i sigma0_i * t_i,  M* = sum_i sigma0_i * t_i * zbar_i.

The residual stress of the magnetic film is ANISOTROPIC (s22 != s11, plus a
shear s12), matching the paper's fitted sigma0 = (45, 75, 5) MPa.  Because the
bending stress in x and y grows differently with z, the in-plane anisotropy
Delta sigma(z) = sxx - syy changes sign across a neutral axis inside the
magnetic layer -- exactly the mechanism the paper invokes for the easy-axis
rotation through thickness.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from materials import IsotropicLayer


@dataclass
class StressState:
    """Through-thickness stress in the magnetic (top) layer."""
    z: np.ndarray          # m, absolute z (0 = laminate mid-reference)
    sxx: np.ndarray        # Pa
    syy: np.ndarray        # Pa
    sxy: np.ndarray        # Pa
    kappa_x: float         # 1/m
    kappa_y: float         # 1/m
    kappa_xy: float        # 1/m
    z_neutral: float | None  # m, where sxx-syy crosses 0 in the magnetic layer


def _abd(layers: list[IsotropicLayer]):
    """Assemble the 3x3 A, B, D lamination matrices and layer z-boundaries."""
    # z boundaries, bottom of stack at z=0, then shift so reference is mid-plane
    t_tot = sum(l.thickness for l in layers)
    z_bot = 0.0
    bounds = [0.0]
    for l in layers:
        z_bot += l.thickness
        bounds.append(z_bot)
    bounds = np.array(bounds) - t_tot / 2.0   # centre reference at geometric mid-plane

    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for i, l in enumerate(layers):
        q11, q12, q66 = l.Q
        Q = np.array([[q11, q12, 0.0],
                      [q12, q11, 0.0],
                      [0.0, 0.0, q66]])
        zk, zk1 = bounds[i], bounds[i + 1]
        A += Q * (zk1 - zk)
        B += Q * 0.5 * (zk1**2 - zk**2)
        D += Q * (1.0 / 3.0) * (zk1**3 - zk**3)
    return A, B, D, bounds


def relax(layers: list[IsotropicLayer],
          sigma0_by_layer: dict[str, tuple[float, float, float]],
          n_z: int = 60) -> StressState:
    """
    Relax a laminate with per-layer initial stress and return curvature and
    through-thickness stress in the top (magnetic) layer.

    sigma0_by_layer maps layer name -> (s11, s22, s12) in Pa. Layers absent
    from the dict are treated as stress-free.
    """
    A, B, D, bounds = _abd(layers)

    Nstar = np.zeros(3)
    Mstar = np.zeros(3)
    for i, l in enumerate(layers):
        s0 = np.array(sigma0_by_layer.get(l.name, (0.0, 0.0, 0.0)))
        zk, zk1 = bounds[i], bounds[i + 1]
        Nstar += s0 * (zk1 - zk)
        Mstar += s0 * 0.5 * (zk1**2 - zk**2)

    # Solve [[A,B],[B,D]] [eps;kap] = [N*;M*]
    K = np.block([[A, B], [B, D]])
    rhs = np.concatenate([Nstar, Mstar])
    sol = np.linalg.solve(K, rhs)
    eps0, kappa = sol[:3], sol[3:]

    # Reconstruct stress in the magnetic (top) layer
    top = layers[-1]
    q11, q12, q66 = top.Q
    Q = np.array([[q11, q12, 0.0],
                  [q12, q11, 0.0],
                  [0.0, 0.0, q66]])
    zk, zk1 = bounds[-2], bounds[-1]
    z = np.linspace(zk, zk1, n_z)
    s0 = np.array(sigma0_by_layer.get(top.name, (0.0, 0.0, 0.0)))
    sxx = np.empty(n_z); syy = np.empty(n_z); sxy = np.empty(n_z)
    for j, zj in enumerate(z):
        total_strain = eps0 + kappa * zj
        s = Q @ total_strain - s0
        sxx[j], syy[j], sxy[j] = s

    # neutral axis of the in-plane anisotropy sxx - syy
    dsig = sxx - syy
    z_neutral = None
    sign_change = np.where(np.diff(np.sign(dsig)))[0]
    if len(sign_change):
        k = sign_change[0]
        # linear interpolation of the zero crossing
        z_neutral = z[k] - dsig[k] * (z[k + 1] - z[k]) / (dsig[k + 1] - dsig[k])

    return StressState(z=z, sxx=sxx, syy=syy, sxy=sxy,
                       kappa_x=float(kappa[0]), kappa_y=float(kappa[1]),
                       kappa_xy=float(kappa[2]), z_neutral=z_neutral)


def curvature_sweep(layers, sigma0_ref, scales, n_z=60):
    """
    Sweep the magnitude of the residual stress (single scalar `scale` applied to
    the magnetic-layer initial stress) and return, for each scale, the measured
    width-direction curvature kappa_y (1/m) and the StressState.

    sigma0_ref: dict layer -> (s11,s22,s12) reference initial stress (Pa).
    Only the magnetic (top) layer stress is scaled; AlN/Pt kept as given.
    """
    top_name = layers[-1].name
    out = []
    for sc in scales:
        s0 = dict(sigma0_ref)
        base = np.array(sigma0_ref[top_name])
        s0[top_name] = tuple(base * sc)
        st = relax(layers, s0, n_z=n_z)
        out.append((abs(st.kappa_y), st, sc))
    return out
