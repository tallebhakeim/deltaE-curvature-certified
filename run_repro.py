"""
Full reproduction driver for Matyushov & Spetzler et al., Adv. Mater. Technol.
2021, 6, 2100294.

Produces:
  fig_9d_aniso.png   -- Keff(kappa) and phi_eff(kappa)   (paper Fig 9d)
  fig_10_frH.png     -- f_r(H) for a flat and a curved device (paper Fig 10)
  fig_3a_response.png-- Delta f_r/f_r,min vs kappa, Squire vs DEAM (paper Fig 3a)

Run:  python3 run_repro.py
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materials import default_stack, FEGAB_E
from mechanics import curvature_sweep
from magnetoelastic import keff_phieff_vs_curvature
from deltaE import (squire_E, deam_E, frequency_response,
                    frequency_response_profile)
from paper_fits import keff_fit, phi_eff_fit

MU0 = 4e-7 * np.pi
T_FEGAB = 500e-9
SIGMA0_REF = {"FeGaB": (45e6, 75e6, 5e6)}   # paper's fitted MOKE-device stress
V_SQUIRE = 0.08                              # paper's active fraction (Fig 10)


PAPER_FLAT_DFR = 2.5   # % : best (flattest) device in Fig 3a, used to fix the
                       # single GLOBAL calibration constant (replaces per-device v)


def build_sweep(n=24, scale_max=11.0):
    layers = default_stack(T_FEGAB)
    scales = np.linspace(0.02, scale_max, n)
    sweep = curvature_sweep(layers, SIGMA0_REF, scales)
    kap, Keff_kJ, phi, prof = keff_phieff_vs_curvature(sweep)
    return kap, Keff_kJ, phi, prof


def fig_9d(kap, Keff_kJ, phi):
    kk = np.linspace(0, max(kap.max(), 7.0), 200)
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(kap, Keff_kJ, "o", color="C0", ms=4, alpha=0.7,
             label=r"$K_{eff}$ FEM (vol.)")
    ax1.plot(kk, keff_fit(kk), "-", color="C0", lw=2,
             label=r"$K_{eff}$ fit (paper)")
    ax1.set_xlabel(r"curvature $\kappa$ (mm$^{-1}$)")
    ax1.set_ylabel(r"$K_{eff}$ (kJ m$^{-3}$)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(kap, phi, "s", color="C3", ms=4, alpha=0.6,
             label=r"$\varphi_{eff}$ FEM (vol.)")
    ax2.plot(kk, phi_eff_fit(kk), "--", color="C3", lw=2,
             label=r"$\varphi_{eff}$ fit (paper)")
    ax2.axhline(60, color="C3", ls=":", lw=0.8, alpha=0.6)
    ax2.set_ylabel(r"$\varphi_{eff}$ (deg from $x$)", color="C3")
    ax2.set_ylim(0, 95)
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Reproduction of Fig. 9d: stress anisotropy vs curvature")
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig("fig_9d_aniso.png", dpi=140)
    plt.close(fig)


def fig_10_frH():
    B = np.linspace(0, 12e-3, 160)
    H = B / MU0
    fig, ax = plt.subplots(figsize=(6, 4))
    for Keff, phi, tag, col in [(0.9e3, 88, "flat  $\\kappa\\approx0.1$", "C0"),
                                (4.5e3, 28, "curved $\\kappa\\approx2.7$", "C3")]:
        Es = squire_E(H, Keff, phi)
        fr, dfr = frequency_response(H, Es, T_FEGAB, v=V_SQUIRE)
        ax.plot(B * 1e3, (fr - fr[0]) / 1e3, "-", color=col,
                label=f"{tag}  ($\\Delta f_r/f_{{r,min}}$={dfr:.2f}%)")
    ax.set_xlabel("DC magnetic field $B$ (mT)")
    ax.set_ylabel("$f_r(H)-f_r(0)$ (kHz)")
    ax.set_title("Reproduction of Fig. 10: $f_r(H)$ (Squire, $v=8\\%$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_10_frH.png", dpi=140)
    plt.close(fig)


def fig_3a(kap, profiles):
    """Response vs curvature.  Two complementary views:
      * markers: through-thickness INTEGRATED model (our mechanism, no fit),
        flat device globally scaled to the paper's 2.5%;
      * lines: paper-FITTED Keff(kappa), phi_eff(kappa) fed to the domain model
        over the full kappa range (to 8 mm^-1), Squire vs DEAM.
    DEAM collapses ~2 orders while single-domain Squire saturates -- showing the
    energy-averaged model is what reproduces the observed dynamic range."""
    B = np.linspace(0, 12e-3, 140)
    H = B / MU0

    # (1) mechanism-based, through-thickness integrated (markers)
    dfr_prof = np.array([frequency_response_profile(H, p, T_FEGAB,
                        model="deam")[1] for p in profiles])
    dfr_prof *= PAPER_FLAT_DFR / dfr_prof[0]

    # (2) paper-fitted anisotropy over extended kappa (lines)
    kk = np.linspace(0.02, 8.0, 40)
    dfr_sq = np.empty_like(kk); dfr_dm = np.empty_like(kk)
    for i, k in enumerate(kk):
        K = keff_fit(k) * 1e3; phi = phi_eff_fit(k)
        Es = squire_E(H, K, phi); _, dfr_sq[i] = frequency_response(H, Es, v=0.08)
        Ed = deam_E(H, K, phi); _, dfr_dm[i] = frequency_response(H, Ed, v=0.08)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(kk, dfr_dm, "-", color="C2", lw=2,
                label="DEAM, paper fits (×%.0f)" % (dfr_dm[0] / dfr_dm[-1]))
    ax.semilogy(kk, dfr_sq, "-", color="C1", lw=2,
                label="Squire 1-domain (×%.0f, saturates)"
                % (dfr_sq[0] / dfr_sq[-1]))
    ax.semilogy(kap, dfr_prof, "s", color="C0", ms=4, alpha=0.7,
                label="DEAM, thickness-integrated (mechanism)")
    ax.set_xlabel(r"curvature $\kappa$ (mm$^{-1}$)")
    ax.set_ylabel(r"$\Delta f_r/f_{r,min}$ (%)")
    ax.set_title("Reproduction of Fig. 3a: response vs curvature")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_3a_response.png", dpi=140)
    plt.close(fig)
    return kk, dfr_sq, dfr_dm


if __name__ == "__main__":
    kap, Keff_kJ, phi, profiles = build_sweep()
    fig_9d(kap, Keff_kJ, phi)
    fig_10_frH()
    k, ds, dd = fig_3a(kap, profiles)
    print("kappa range: %.3f .. %.3f mm^-1" % (kap.min(), kap.max()))
    print("Squire dfr: %.3f%% (flat) -> %.4f%% (curved), dynamic range x%.0f"
          % (ds[0], ds[-1], ds[0] / ds[-1]))
    print("DEAM   dfr: %.3f%% (flat) -> %.4f%% (curved), dynamic range x%.0f"
          % (dd[0], dd[-1], dd[0] / dd[-1]))
    print("Figures written: fig_9d_aniso.png, fig_10_frH.png, fig_3a_response.png")
