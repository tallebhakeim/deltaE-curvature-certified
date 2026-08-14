"""
Driver for the three innovation extensions (axes 2, 3, 4).

Produces:
  fig_6_Q.png        -- Q vs curvature and Q vs response (paper Fig 6, but from a
                        physical magnetic-loss model instead of the Eq-4 fit)
  fig_cert_band.png  -- certified response band from a stress-uncertainty box
  fig_design.png     -- inverse-designed compensation layer flattening the plate

Run:  python3 run_extensions.py
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materials import default_stack, FEGAB_E
from mechanics import curvature_sweep
from magnetoelastic import keff_phieff_vs_curvature
from deltaE import frequency_response_profile, deam_E, frequency_response
from q_factor import q_vs_curvature
from paper_fits import keff_fit, phi_eff_fit
from extensions import (certified_kappa_band, certified_response_band,
                        design_compensation_layer)

MU0 = 4e-7 * np.pi
T_FEGAB = 500e-9
SIGMA0_REF = {"FeGaB": (45e6, 75e6, 5e6)}
CALIB = None   # set from the flat device to match paper's 2.5%


def _sweep(n=16):
    layers = default_stack(T_FEGAB)
    sweep = curvature_sweep(layers, SIGMA0_REF, np.linspace(0.02, 11, n))
    kap, Keff, phi, profs = keff_phieff_vs_curvature(sweep)
    return layers, kap, profs


def _response_curve(profs, model="deam"):
    B = np.linspace(0, 12e-3, 120); H = B / MU0
    dfr = np.array([frequency_response_profile(H, p, T_FEGAB, model=model)[1]
                    for p in profs])
    return dfr * (2.5 / dfr[0])   # global calibration to paper's flat 2.5%


def fig_6_Q(kap, profs):
    B = np.linspace(0, 12e-3, 120); H = B / MU0
    Q, _ = q_vs_curvature(H, profs, model="deam")
    dfr = _response_curve(profs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(kap, Q, "o-", color="C4")
    a1.set_xlabel(r"curvature $\kappa$ (mm$^{-1}$)"); a1.set_ylabel("Q")
    a1.set_title("Q rises with curvature")
    a2.plot(dfr, Q, "s-", color="C1")
    a2.set_xscale("log")
    a2.set_xlabel(r"$\Delta f_r/f_{r,min}$ (%)"); a2.set_ylabel("Q")
    a2.set_title("Q anti-correlates with response (Fig. 6)")
    for a in (a1, a2):
        a.grid(True, alpha=0.3)
    fig.suptitle("Innovation 3: physical magnetic-loss model for Q")
    fig.tight_layout()
    fig.savefig("fig_6_Q.png", dpi=140)
    plt.close(fig)


def fig_cert(layers, kap, profs):
    dfr = _response_curve(profs)
    # certified band for the nominal device under a 20% stress-uncertainty box
    k0, klo, khi = certified_kappa_band(layers, SIGMA0_REF, 0.20)
    dlo, dhi = certified_response_band(klo, khi, kap, dfr)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.semilogy(kap, dfr, "o-", color="C2", label="modeled response")
    ax.axvspan(abs(klo)/1e3, abs(khi)/1e3, color="C0", alpha=0.18,
               label=r"certified $\kappa$ box (20% stress)")
    ax.axhspan(dlo, dhi, color="C3", alpha=0.15,
               label="certified response band")
    ax.axvline(abs(k0)/1e3, color="k", ls=":", lw=1)
    ax.set_xlabel(r"curvature $\kappa$ (mm$^{-1}$)")
    ax.set_ylabel(r"$\Delta f_r/f_{r,min}$ (%)")
    ax.set_title("Innovation 2: certified stress $\\to$ performance band")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_cert_band.png", dpi=140)
    plt.close(fig)
    return (k0, klo, khi, dlo, dhi)


def fig_detectivity(Q0=2500.0):
    """Couple the magnetic loss Q(kappa) into the sensor detectivity (limit of
    detection).  LOD ~ (f_r/Q)/S with S ~ Delta f_r.  The raw response spans ~2
    orders with curvature, but the low Q of the high-response (flat) devices
    largely CANCELS their sensitivity advantage, so the loss-limited detectivity
    is far more robust to curvature than Fig 3a suggests -- a conclusion the
    paper does not draw."""
    B = np.linspace(0, 12e-3, 140); H = B / MU0
    kk = np.linspace(0.05, 8.0, 30)

    def swing(k):
        E = deam_E(H, keff_fit(k) * 1e3, phi_eff_fit(k))
        return (E.max() - E.min()) / FEGAB_E
    alpha_m = (1.0 / 300.0 - 1.0 / Q0) / swing(kk[0])   # Q(flat)=300

    S = np.empty_like(kk); Q = np.empty_like(kk)
    for i, k in enumerate(kk):
        E = deam_E(H, keff_fit(k) * 1e3, phi_eff_fit(k))
        _, S[i] = frequency_response(H, E, v=0.08)
        Q[i] = 1.0 / (1.0 / Q0 + alpha_m * (E.max() - E.min()) / FEGAB_E)
    LOD = 1.0 / (Q * S); LOD /= LOD.min()
    resp = S / S[-1]                      # raw response, normalized to worst

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(kk, resp, "-", color="C2", lw=2,
                label=f"raw response spread (×{S[0]/S[-1]:.0f})")
    ax.semilogy(kk, LOD, "-", color="C3", lw=2,
                label=f"detectivity / LOD spread (×{LOD.max()/LOD.min():.0f})")
    ax.set_xlabel(r"curvature $\kappa$ (mm$^{-1}$)")
    ax.set_ylabel("normalized spread (a.u.)")
    ax.set_title("Innovation 3b: loss-limited detectivity is curvature-robust")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_detectivity.png", dpi=140)
    plt.close(fig)
    return S[0] / S[-1], Q[-1] / Q[0], LOD.max() / LOD.min()


def fig_design():
    d = design_compensation_layer(SIGMA0_REF, t_fegab=T_FEGAB)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    labels = ["as-fabricated", "with compensation\nlayer"]
    vals = [abs(d.kappa_before)/1e3, abs(d.kappa_after)/1e3 + 1e-4]
    ax.bar(labels, vals, color=["C3", "C0"])
    ax.set_ylabel(r"|curvature| $\kappa$ (mm$^{-1}$)")
    ax.set_title("Innovation 4: inverse-designed stress compensation")
    ax.text(1, vals[1], f"  $\\sigma_{{comp}}$={d.comp_stress_biaxial/1e6:.0f} MPa\n"
            f"  (100 nm cap)", va="bottom", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_design.png", dpi=140)
    plt.close(fig)
    return d


if __name__ == "__main__":
    layers, kap, profs = _sweep()
    fig_6_Q(kap, profs)
    cert = fig_cert(layers, kap, profs)
    det = fig_detectivity()
    d = fig_design()
    print("Certified: kappa=%.3f in [%.3f, %.3f] mm^-1  ->  response in [%.3f, %.3f]%%"
          % (cert[0]/1e3, abs(cert[1])/1e3, abs(cert[2])/1e3, cert[3], cert[4]))
    print("Detectivity: raw response x%.0f, Q x%.1f  ->  LOD spread only x%.1f"
          % det)
    print("Inverse design: sigma_comp=%.0f MPa flattens kappa %.3f -> ~0 mm^-1"
          % (d.comp_stress_biaxial/1e6, d.kappa_before/1e3))
    print("Figures: fig_6_Q.png, fig_cert_band.png, fig_detectivity.png, fig_design.png")
