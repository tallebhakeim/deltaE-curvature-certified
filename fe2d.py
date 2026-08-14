"""
2D plane-stress finite-element relaxation of the resonator plate footprint.

Motivation.  The laminated-plate (CLT) model of mechanics.py treats the plate as
free, so a spatially UNIFORM biaxial residual stress relaxes almost completely
and only the inter-layer mismatch survives.  Real devices are suspended by two
anchors that clamp the plate at its ends; that constraint PREVENTS full in-plane
relaxation, so a finite fraction of the residual in-plane anisotropy is retained,
strongest near the anchors and weakest at the centre.  This is exactly the
spatial pattern seen by MOKE in [1] (easy axis differs centre vs edge) and it is
the mechanism that closes the magnitude gap of Figure 1.

This module solves in-plane (membrane) linear elasticity on the plate footprint
(length x width) with Q4 bilinear elements, a uniform anisotropic eigenstress in
the film, and the two anchor patches clamped.  It returns the spatial map and the
area-averaged retained in-plane stress anisotropy, and the "retention fraction"
f_ret = <sxx - syy>_plate / (s0_xx - s0_yy) relative to the clamped-flat value.
"""
from __future__ import annotations
import numpy as np


def _q4_Ke_fe(Ex, Ey, nu, G, xe, ye):
    """Q4 plane-stress element stiffness and the B-at-centroid, 2x2 Gauss."""
    C = np.array([[Ex / (1 - nu * nu), nu * Ey / (1 - nu * nu), 0],
                  [nu * Ex / (1 - nu * nu), Ey / (1 - nu * nu), 0],
                  [0, 0, G]])
    g = 1.0 / np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
    Ke = np.zeros((8, 8))
    feig = np.zeros(8)      # will be filled by caller with eigenstress
    Bc = None
    for xi, eta in gp:
        dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                              [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
        J = dN @ np.column_stack([xe, ye])
        detJ = np.linalg.det(J)
        dNxy = np.linalg.solve(J, dN)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNxy[0]
        B[1, 1::2] = dNxy[1]
        B[2, 0::2] = dNxy[1]
        B[2, 1::2] = dNxy[0]
        Ke += (B.T @ C @ B) * detJ
        if Bc is None:
            Bc = B.copy()
    return Ke, C, Bc


def relax_plate(Lx=200e-6, Ly=70e-6, nx=40, ny=16,
                sigma0=(45e6, 75e6, 5e6),
                E=150e9, nu=0.3,
                anchor_frac=0.12):
    """
    Solve the clamped-anchor plate relaxation.  Anchors are the two end strips of
    length anchor_frac*Lx, clamped (u=0).  sigma0 is the uniform in-plane
    eigenstress of the film (Pa).  E is the effective in-plane modulus of the
    composite membrane.  Returns dict with the retained-anisotropy map and the
    area-averaged retention fraction.
    """
    G = E / (2 * (1 + nu))
    s0 = np.array(sigma0)
    # node grid
    X, Y = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    nnode = nodes.shape[0]
    nid = lambda i, j: j * (nx + 1) + i     # i along x, j along y

    ndof = 2 * nnode
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            elems.append(n)
            xe = nodes[n, 0]; ye = nodes[n, 1]
            Ke, C, Bc = _q4_Ke_fe(E, E, nu, G, xe, ye)
            # eigenstress nodal load: f = integral B^T sigma0 dV ~ B^T s0 * area
            area = 0.5 * abs((xe[2] - xe[0]) * (ye[3] - ye[1])
                             - (xe[3] - xe[1]) * (ye[2] - ye[0]))
            fe = Bc.T @ s0 * area
            dofs = np.array([[2 * k, 2 * k + 1] for k in n]).ravel()
            K[np.ix_(dofs, dofs)] += Ke
            F[dofs] += fe

    # clamp anchor nodes (two end strips along x)
    xa = anchor_frac * Lx
    clamp = np.where((nodes[:, 0] <= xa) | (nodes[:, 0] >= Lx - xa))[0]
    fixed = np.concatenate([2 * clamp, 2 * clamp + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    # element-centroid retained stress = C (B u) - sigma0
    cent = []; sret = []
    for n in elems:
        xe = nodes[n, 0]; ye = nodes[n, 1]
        _, C, Bc = _q4_Ke_fe(E, E, nu, G, xe, ye)
        dofs = np.array([[2 * k, 2 * k + 1] for k in n]).ravel()
        strain = Bc @ u[dofs]
        s = C @ strain - s0
        cent.append([xe.mean(), ye.mean()])
        sret.append(s)
    cent = np.array(cent); sret = np.array(sret)

    # consider only the central (non-anchor) region for the averages
    core = (cent[:, 0] > xa) & (cent[:, 0] < Lx - xa)
    dsig = sret[:, 0] - sret[:, 1]         # sxx - syy retained
    ds0 = s0[0] - s0[1]
    f_ret_core = float(np.mean(dsig[core]) / ds0)
    f_ret_all = float(np.mean(dsig) / ds0)
    return {
        "cent": cent, "sret": sret, "dsig": dsig, "core": core,
        "f_ret_core": f_ret_core, "f_ret_all": f_ret_all,
        "Lx": Lx, "Ly": Ly, "xa": xa, "ds0": ds0,
    }
