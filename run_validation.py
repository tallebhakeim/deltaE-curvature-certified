"""
Quantitative validation of the model against the MEASURED device population of

  A. D. Matyushov, B. Spetzler, ... N. X. Sun, Adv. Mater. Technol. 6,
  2100294 (2021).

The measured values are read from measured_fig3a.json / measured_fig6a.json,
produced by digitize_paper_figures.py directly from the vector content of the
published figures (64 devices in Fig. 3a, 51 in Fig. 6a).

Two comparisons are made, each with a single global calibration constant, i.e.
the same number of free scalars the original work uses:

  (1) response:  Delta f_r/f_r,min versus curvature.  The energy-averaged
      domain model (DEAM) and the single-domain (coherent-rotation) model are
      both scaled by ONE global active fraction v fitted to the whole
      population in log space.  The paper instead fits v per device.

  (2) quality factor:  Q versus response.  The magnetoelastic loss model has
      two constants (Q0, alpha_m), the same count as the two damping constants
      gamma_NM, gamma_M of Eq. (4) of the paper, but here the SHAPE of Q(x) is
      predicted by the same softening compliance that produces the Delta-E
      effect rather than assumed.

Outputs: fig_validation_response.png, fig_validation_Q.png, validation.json
Run:     python3 run_validation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

from deltaE import squire_E, deam_E, frequency_response
from materials import MU0
from paper_fits import keff_fit, phi_eff_fit

HERE = Path(__file__).parent
B_SWEEP = np.linspace(0.0, 12e-3, 140)
H_SWEEP = B_SWEEP / MU0
V_REF = 0.08                      # paper's per-device active fraction


# --------------------------------------------------------------------------
def load_measured(name):
    d = json.loads((HERE / name).read_text())
    p = np.asarray(d["points"], dtype=float)
    return p[:, 0], p[:, 1], d


def model_response(kappa, model, v):
    """Delta f_r/f_r,min (%) at the given curvatures for the requested model."""
    out = np.empty_like(kappa, dtype=float)
    for i, k in enumerate(kappa):
        K = keff_fit(k) * 1e3
        phi = phi_eff_fit(k)
        E = deam_E(H_SWEEP, K, phi) if model == "deam" else squire_E(H_SWEEP, K, phi)
        _, out[i] = frequency_response(H_SWEEP, E, v=v)
    return out


def fit_global_v(kappa_meas, resp_meas, model, grid=np.linspace(0.02, 0.20, 19)):
    """One global active fraction, least squares on log10(response)."""
    best = None
    for v in grid:
        pred = model_response(kappa_meas, model, v)
        rms = float(np.sqrt(np.mean((np.log10(pred) - np.log10(resp_meas)) ** 2)))
        if best is None or rms < best[1]:
            best = (float(v), rms, pred)
    return best


def decade_spread(y):
    return float(np.max(y) / np.min(y))


# --------------------------------------------------------------------------
def fit_free_branch(kap_meas, resp_meas):
    """Through-thickness integrated DEAM: the local (Keff(z), phi_eff(z)) come
    from the laminated-plate stress field, so NO fitted anisotropy function is
    used.  Only one global scale is fitted, as for the other branches."""
    from mechanics import curvature_sweep
    from magnetoelastic import keff_phieff_vs_curvature
    from materials import default_stack
    from deltaE import frequency_response_profile

    layers = default_stack(500e-9)
    sweep = curvature_sweep(layers, {"FeGaB": (45e6, 75e6, 5e6)},
                            np.linspace(0.02, 11.0, 24))
    kap, _, _, profiles = keff_phieff_vs_curvature(sweep)
    dfr = np.array([frequency_response_profile(H_SWEEP, p, 500e-9, model="deam")[1]
                    for p in profiles])
    inside = (kap >= kap_meas.min()) & (kap <= kap_meas.max())
    logi = np.interp(np.log(kap_meas), np.log(kap[inside]), np.log(dfr[inside]))
    scale = float(np.exp(np.mean(np.log(resp_meas) - logi)))
    rms = float(np.sqrt(np.mean((np.log10(np.exp(logi) * scale)
                                 - np.log10(resp_meas)) ** 2)))
    return kap[inside], dfr[inside] * scale, rms


def validate_response():
    kap, resp, meta = load_measured("measured_fig3a.json")
    order = np.argsort(kap)
    kap, resp = kap[order], resp[order]

    v_deam, rms_deam, pred_deam = fit_global_v(kap, resp, "deam")
    v_sq, rms_sq, pred_sq = fit_global_v(kap, resp, "squire")
    k_ff, dfr_ff, rms_ff = fit_free_branch(kap, resp)

    # the model curve published in Fig. 10 of [1], scored on the same 64 points
    k10, v10, _ = load_measured("model_fig10.json")
    pred10 = np.exp(np.interp(np.log(kap), np.log(k10), np.log(v10)))
    rms10 = float(np.sqrt(np.mean((np.log10(pred10) - np.log10(resp)) ** 2)))
    med10 = float(np.median(np.abs(pred10 / resp - 1.0))) * 100

    # smooth curves for the figure, over the measured curvature range
    kk = np.linspace(kap.min(), kap.max(), 60)
    curve_deam = model_response(kk, "deam", v_deam)
    curve_sq = model_response(kk, "squire", v_sq)

    med_deam = float(np.median(np.abs(pred_deam / resp - 1.0))) * 100
    med_sq = float(np.median(np.abs(pred_sq / resp - 1.0))) * 100

    stats = {
        "n_devices": int(kap.size),
        "kappa_min": float(kap.min()), "kappa_max": float(kap.max()),
        "measured_spread": decade_spread(resp),
        "measured_min": float(resp.min()), "measured_max": float(resp.max()),
        "deam": {"v_global_pct": 100 * v_deam,
                 "rms_log10": rms_deam,
                 "rms_factor": float(10 ** rms_deam),
                 "median_abs_dev_pct": med_deam,
                 "spread": decade_spread(curve_deam)},
        "squire": {"v_global_pct": 100 * v_sq,
                   "rms_log10": rms_sq,
                   "rms_factor": float(10 ** rms_sq),
                   "median_abs_dev_pct": med_sq,
                   "spread": decade_spread(curve_sq)},
        "deam_fit_free": {"rms_log10": rms_ff,
                          "rms_factor": float(10 ** rms_ff),
                          "spread": decade_spread(dfr_ff),
                          "kappa_max": float(k_ff.max())},
        "published_model": {"rms_factor": float(10 ** rms10),
                            "median_abs_dev_pct": med10,
                            "spread": decade_spread(
                                v10[(k10 >= kap.min()) & (k10 <= kap.max())]),
                            "range": [float(v10[0]), float(v10[-1])]},
        "curve_range_deam": [float(curve_deam[0]), float(curve_deam[-1])],
        "curve_range_squire": [float(curve_sq[0]), float(curve_sq[-1])],
    }

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.semilogy(kap, resp, "o", ms=5, mfc="none", mec="0.35", mew=1.0,
                label=f"measured, {kap.size} devices [1]")
    ax.semilogy(kk, curve_deam, "-", color="C2", lw=2.2,
                label=(r"DEAM, single global $v=%.0f\%%$ (%s%.0f)"
                       % (100 * v_deam, r"$\times$", stats["deam"]["spread"])))
    ax.semilogy(kk, curve_sq, "--", color="C1", lw=2.2,
                label=(r"single domain, $v=%.0f\%%$ (%s%.1f, saturates)"
                       % (100 * v_sq, r"$\times$", stats["squire"]["spread"])))
    ax.semilogy(k_ff, dfr_ff, "s", color="C0", ms=4.5, alpha=0.85,
                label="DEAM, thickness-integrated (no fitted anisotropy)")
    ax.semilogy(k10, v10, ":", color="C3", lw=2.0,
                label=r"two-domain model of [1], $v$ fitted per device")
    ax.set_xlabel(r"curvature $\kappa$ (mm$^{-1}$)")
    ax.set_ylabel(r"$\Delta f_r/f_{r,\min}$ (%)")
    ax.set_ylim(0.008, 3.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "fig_validation_response.png", dpi=160)
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------
SUBSETS = (("measured_fig6c.json", 246e6, "designs 3, 4"),
           ("measured_fig6d.json", 138e6, "designs 7, 8"))
GAMMA_TABLE2 = {"measured_fig6c.json": (0.027, 0.150),   # 1e8 Hz, Table 2 of [1]
                "measured_fig6d.json": (0.017, 0.041)}


def _fit_eq4(x, q, f_r):
    """Equation (4) of [1]: Q = 2 pi f_r / (gamma_NM + gamma_M x/x_max)."""
    x_max = x.max()

    def model(xx, g_nm, g_m):
        return 2 * np.pi * f_r / (g_nm * 1e8 + g_m * 1e8 * xx / x_max)

    (g_nm, g_m), _ = curve_fit(model, x, q, p0=[0.02, 0.05], maxfev=20000)
    r2 = 1.0 - np.sum((q - model(x, g_nm, g_m)) ** 2) / np.sum((q - q.mean()) ** 2)
    return float(g_nm), float(g_m), float(r2)


def validate_q():
    """Quality factor versus response.

    Because the magnetoelastic loss compliance is proportional to the same
    softening that produces the Delta-E effect, the model gives

        1/Q = 1/Q0 + 2 pi f_r tau x ,        x = Delta f_r/f_r,min ,

    which is algebraically Eq. (4) of [1] but with the two damping constants
    replaced by a non-magnetic Q0 and a single magnetoelastic relaxation time
    tau, both shared by every device family.  The paper fits gamma_NM and
    gamma_M separately for each of its two fixed-f_r subsets (four constants).
    """
    data, per_subset = [], []
    for fname, f_r, tag in SUBSETS:
        x, q, _ = load_measured(fname)
        data.append((x, q, f_r, tag))
        g_nm, g_m, r2 = _fit_eq4(x, q, f_r)
        ref = GAMMA_TABLE2[fname]
        per_subset.append({"subset": tag, "f_r_MHz": f_r / 1e6, "n": int(x.size),
                           "gamma_NM_refit": g_nm, "gamma_M_refit": g_m,
                           "gamma_NM_table2": ref[0], "gamma_M_table2": ref[1],
                           "R2_paper_form": r2})

    def resid(p):
        inv_q0, tau = p
        return np.concatenate([q - 1.0 / (inv_q0 + 2 * np.pi * f_r * tau * x)
                               for x, q, f_r, _ in data])

    sol = least_squares(resid, [1.0 / 540.0, 4e-12])
    inv_q0, tau = float(sol.x[0]), float(sol.x[1])
    q_all = np.concatenate([q for _, q, _, _ in data])
    r2_joint = 1.0 - float(np.sum(sol.fun ** 2)) / float(np.sum((q_all - q_all.mean()) ** 2))

    for entry, (x, q, f_r, _) in zip(per_subset, data):
        pred = 1.0 / (inv_q0 + 2 * np.pi * f_r * tau * x)
        entry["R2_shared_constants"] = float(
            1.0 - np.sum((q - pred) ** 2) / np.sum((q - q.mean()) ** 2))

    xa, qa, _ = load_measured("measured_fig6a.json")
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 7.4))
    ax = axes[0]
    ax.plot(xa, qa, "o", ms=4.5, mfc="none", mec="0.55", mew=0.9,
            label=f"measured ensemble, {xa.size} devices [1]")
    xx = np.linspace(1e-3, 1.25, 200)
    for (x, q, f_r, tag), col in zip(data, ("C2", "C0")):
        ax.plot(xx, 1.0 / (inv_q0 + 2 * np.pi * f_r * tau * xx), "-", color=col,
                lw=2.0, label=r"model, $f_r=%.0f$ MHz" % (f_r / 1e6))
    ax.set_xlabel(r"$\Delta f_r/f_{r,\min}$ (%)")
    ax.set_ylabel(r"quality factor $Q$")
    ax.set_xlim(0, 1.25); ax.set_ylim(0, 780)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7.5)
    ax.set_title("(a) full ensemble", fontsize=9, loc="left")

    ax = axes[1]
    for (x, q, f_r, tag), col, mk in zip(data, ("C2", "C0"), ("s", "^")):
        ax.plot(x, q, mk, ms=5, mfc="none", mec=col, mew=1.2,
                label=f"{tag}, {f_r/1e6:.0f} MHz")
        ax.plot(xx, 1.0 / (inv_q0 + 2 * np.pi * f_r * tau * xx), "-", color=col, lw=2.0)
    ax.set_xlabel(r"$\Delta f_r/f_{r,\min}$ (%)")
    ax.set_xlim(0, 1.25); ax.set_ylim(0, 780)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7.5)
    ax.set_title(r"(b) fixed-$f_r$ subsets, $Q_0=%.0f$, $\tau=%.1f$ ps/%%"
                 % (1.0 / inv_q0, tau * 1e12), fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_validation_Q.png", dpi=160)
    plt.close(fig)

    return {"n_ensemble": int(xa.size), "Q_min": float(qa.min()), "Q_max": float(qa.max()),
            "pearson_r_ensemble": float(np.corrcoef(xa, qa)[0, 1]),
            "Q0_shared": 1.0 / inv_q0, "tau_ps_per_pct": tau * 1e12,
            "R2_joint": r2_joint, "subsets": per_subset}


# --------------------------------------------------------------------------
def validate_detectivity(q0, tau):
    """Loss-limited detectivity, computed device by device from the MEASURED
    response and quality factor.

    For a frequency-readout resonator the noise floor scales with the linewidth
    f_r/Q and the sensitivity with the response, so the limit of detection
    scales as (f_r/Q)/x.  Only the two fixed-frequency subsets can be used,
    since f_r must be known per device.
    """
    per, pooled_x, pooled_d = [], [], []
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for (fname, f_r, tag), col, mk in zip(SUBSETS, ("C2", "C0"), ("s", "^")):
        x, q, _ = load_measured(fname)
        order = np.argsort(x)
        x, q = x[order], q[order]
        d = (f_r / q) / x
        d = d / d.min()
        pooled_x.append(x); pooled_d.append((f_r / q) / x)

        # what the loss model predicts over the same response range
        q_mod = 1.0 / (1.0 / q0 + 2 * np.pi * f_r * tau * x)
        d_mod = (f_r / q_mod) / x
        per.append({"subset": tag, "f_r_MHz": f_r / 1e6, "n": int(x.size),
                    "response_spread": float(x.max() / x.min()),
                    "Q_spread": float(q.max() / q.min()),
                    "detectivity_spread_measured": float(d.max() / d.min()),
                    "detectivity_spread_model": float(d_mod.max() / d_mod.min()),
                    "compression_measured": float((x.max() / x.min()) / (d.max() / d.min())),
                    "compression_model": float((x.max() / x.min()) / (d_mod.max() / d_mod.min()))})

        ax.loglog(x, d, mk, ms=6, mfc="none", mec=col, mew=1.3,
                  label=f"measured, {tag} ({f_r/1e6:.0f} MHz)")
        ax.loglog(x, d_mod / d_mod.min(), "-", color=col, lw=1.8)

    xr = np.array([min(np.concatenate(pooled_x)), max(np.concatenate(pooled_x))])
    ax.loglog(xr, (xr.max() / xr) / (xr.max() / xr).min(), "k:", lw=1.5,
              label=r"constant $Q$ (no loss compensation)")
    ax.set_xlabel(r"$\Delta f_r/f_{r,\min}$ (%)")
    ax.set_ylabel("loss-limited detectivity (normalised)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_validation_detectivity.png", dpi=160)
    plt.close(fig)

    X = np.concatenate(pooled_x); D = np.concatenate(pooled_d)
    return {"subsets": per, "n_pooled": int(X.size),
            "response_spread_pooled": float(X.max() / X.min()),
            "detectivity_spread_pooled": float(D.max() / D.min()),
            "compression_pooled": float((X.max() / X.min()) / (D.max() / D.min())),
            "log_correlation": float(np.corrcoef(np.log(X), np.log(D))[0, 1])}


if __name__ == "__main__":
    r = validate_response()
    q = validate_q()
    d = validate_detectivity(q["Q0_shared"], q["tau_ps_per_pct"] * 1e-12)
    out = {"response": r, "quality_factor": q, "detectivity": d}
    (HERE / "validation.json").write_text(json.dumps(out, indent=1))

    print("RESPONSE  (%d measured devices, kappa %.2f-%.2f 1/mm)"
          % (r["n_devices"], r["kappa_min"], r["kappa_max"]))
    print("  measured  : %.3f-%.3f %%  spread x%.0f"
          % (r["measured_min"], r["measured_max"], r["measured_spread"]))
    for tag in ("deam", "squire"):
        s = r[tag]
        print("  %-7s : v=%.0f%%  spread x%-4.0f  RMS factor %.2f  median dev %.0f%%"
              % (tag, s["v_global_pct"], s["spread"], s["rms_factor"],
                 s["median_abs_dev_pct"]))
    s = r["deam_fit_free"]
    print("  fit-free: no fitted anisotropy, spread x%.0f up to kappa=%.1f, "
          "RMS factor %.2f" % (s["spread"], s["kappa_max"], s["rms_factor"]))
    s = r["published_model"]
    print("  [1] Fig10: published two-domain model, spread x%.0f, RMS factor %.2f, "
          "median dev %.0f%%" % (s["spread"], s["rms_factor"], s["median_abs_dev_pct"]))
    print("  curve ranges: DEAM %.3f-%.3f %%, single domain %.3f-%.3f %%"
          % (*r["curve_range_deam"], *r["curve_range_squire"]))
    print("QUALITY FACTOR  (%d measured devices in the ensemble)" % q["n_ensemble"])
    print("  Q measured %.0f-%.0f, Pearson r = %.2f"
          % (q["Q_min"], q["Q_max"], q["pearson_r_ensemble"]))
    print("  shared constants: Q0 = %.0f, tau = %.2f ps per %% of response"
          % (q["Q0_shared"], q["tau_ps_per_pct"]))
    for s in q["subsets"]:
        print("  %-11s f_r=%3.0f MHz  N=%2d | Eq.(4) refit gNM=%.3f gM=%.3f "
              "(Table II of [1]: %.3f / %.3f), R2 %.2f -> %.2f with shared constants"
              % (s["subset"], s["f_r_MHz"], s["n"], s["gamma_NM_refit"],
                 s["gamma_M_refit"], s["gamma_NM_table2"], s["gamma_M_table2"],
                 s["R2_paper_form"], s["R2_shared_constants"]))
    print("DETECTIVITY  (%d devices with known f_r)" % d["n_pooled"])
    print("  pooled: response x%.0f -> loss-limited detectivity x%.0f (compression x%.1f)"
          % (d["response_spread_pooled"], d["detectivity_spread_pooled"],
             d["compression_pooled"]))
    for s in d["subsets"]:
        print("  %-11s response x%-4.0f Q x%-5.1f detectivity x%-5.1f measured "
              "(x%.1f predicted), compression x%.1f measured vs x%.1f predicted"
              % (s["subset"], s["response_spread"], s["Q_spread"],
                 s["detectivity_spread_measured"], s["detectivity_spread_model"],
                 s["compression_measured"], s["compression_model"]))
    print("Figures: fig_validation_response.png, fig_validation_Q.png, "
          "fig_validation_detectivity.png")
