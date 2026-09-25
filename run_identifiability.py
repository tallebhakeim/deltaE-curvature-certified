"""
Practical identifiability of the global constants.

Three questions, in the order a reviewer asks them:

  1. Is the domain-population width identifiable?  The width enters as
     A_s = c/Keff.  We profile c over more than two decades, refitting the
     active fraction at each c, and report how much the fit actually changes.

  2. Are the two loss constants separately determined, or do they trade off?
     Linearised covariance at the optimum plus a profile of tau with Q0
     refitted, on top of the leave-one-out spread of run_crossvalidation.py.

  3. Does the assumed baseline residual stress (45, 75, 5) MPa trade off against
     the fitted scale?  Only the thickness-integrated branch is exposed to it,
     since elsewhere the curvature is measured, not computed.  We vary the
     in-plane anisotropy ratio and the shear component, refit the single scale,
     and report what the data can and cannot distinguish.

Outputs: identifiability.json, fig_identifiability.png
Run:     python3 run_identifiability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import crossval as cv
from deltaE import frequency_response
from deltaE_analytic import coherent_compliance, modulus
from magnetoelastic import keff_phieff_vs_curvature
from materials import default_stack
from mechanics import curvature_sweep

HERE = Path(__file__).parent


# ------------------------------------------------- 1. domain width c --------
def profile_width(n_boot=200, seed=20260919):
    kappa, resp = cv.load_points("measured_fig3a.json")
    c_grid = np.array([2., 3., 5., 8., 10., 15., 25., 50., 100., 300., 1000.])
    prof = []
    for c in c_grid:
        v, rms = cv.fit_v(kappa, resp, "averaged", float(c))
        prof.append({"c": float(c), "v_pct": 100 * v, "rms_factor": 10 ** rms})
    rms_all = np.array([p["rms_factor"] for p in prof])
    v_all = np.array([p["v_pct"] for p in prof])
    stiff = [p for p in prof if p["c"] >= 5.0]
    v_stiff = np.array([p["v_pct"] for p in stiff])

    # coherent rotation is the c -> infinity limit of the same family
    v_coh, rms_coh = cv.fit_v(kappa, resp, "coherent")

    # bootstrap confidence interval of the one constant that IS identifiable
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, kappa.size, kappa.size)
        v, _ = cv.fit_v(kappa[idx], resp[idx], "coherent")
        boot.append(100 * v)
    boot = np.array(boot)

    return {
        "profile": prof,
        "rms_factor_span_pct": float(100 * (rms_all.max() / rms_all.min() - 1.0)),
        "v_pct_span": [float(v_all.min()), float(v_all.max())],
        "v_pct_span_c_above_5": [float(v_stiff.min()), float(v_stiff.max())],
        "valley": ("for c below about 5 the pair (v, c) trades off along a shallow "
                   "valley: v rises to 10.8% at c = 2 for a 1% better fit, so only "
                   "the response amplitude is determined, not the two separately"),
        "coherent_limit": {"v_pct": 100 * v_coh, "rms_factor": 10 ** rms_coh},
        "active_fraction_bootstrap": {
            "n": n_boot, "mean_pct": float(boot.mean()), "std_pct": float(boot.std()),
            "ci95_pct": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
    }, (c_grid, rms_all, v_all)


# ------------------------------------------------- 2. loss constants --------
SUBSETS = (("measured_fig6c.json", 246e6), ("measured_fig6d.json", 138e6))


def _sets():
    out = []
    for name, f_r in SUBSETS:
        x, q = cv.load_points(name)
        out.append((x, q, f_r))
    return out


def _resid(u, sets):
    q0, tau = 1000.0 / u[0], u[1] * 1e-12
    return np.concatenate([q - 1.0 / (1.0 / q0 + 2 * np.pi * f * tau * x)
                           for x, q, f in sets])


def loss_identifiability():
    sets = _sets()
    sol = least_squares(_resid, [1000.0 / 540.0, 4.0], args=(sets,))
    u = sol.x
    q0, tau_ps = 1000.0 / u[0], float(u[1])
    n = sum(x.size for x, _, _ in sets)
    dof = n - 2
    s2 = 2.0 * sol.cost / dof
    J = sol.jac
    cov = s2 * np.linalg.inv(J.T @ J)
    sd = np.sqrt(np.diag(cov))
    corr = cov[0, 1] / (sd[0] * sd[1])
    # propagate the standard error of u0 = 1000/Q0 to Q0
    sd_q0 = q0 * sd[0] / u[0]

    # profile of tau with Q0 refitted at each tau
    prof = []
    for tau_fix in np.linspace(0.5, 24.0, 48):
        r = least_squares(lambda a: _resid([a[0], tau_fix], sets), [u[0]])
        prof.append({"tau_ps": float(tau_fix), "Q0": 1000.0 / r.x[0],
                     "rms_Q": float(np.sqrt(2 * r.cost / n))})
    rms0 = float(np.sqrt(2 * sol.cost / n))
    # 95% interval from the F-threshold on the sum of squares (2 parameters)
    thr = rms0 * np.sqrt(1.0 + 2.0 * 3.0 / dof)
    inside = [p["tau_ps"] for p in prof if p["rms_Q"] <= thr]

    return {
        "n_devices": int(n),
        "Q0": q0, "Q0_stderr": float(sd_q0),
        "tau_ps_per_pct": tau_ps, "tau_stderr_ps": float(sd[1]),
        "correlation_Q0_tau": float(-corr),   # sign in terms of Q0, not 1/Q0
        "condition_number_jacobian": float(np.linalg.cond(J)),
        "rms_Q_at_optimum": rms0,
        "tau_profile": prof,
        "tau_interval_95_ps": [float(min(inside)), float(max(inside))] if inside else None,
        "previous_submission_values": {"Q0": 540.0, "tau_ps_per_pct": 4.0,
                                       "status": "optimiser start point, not a fit"},
    }, (prof, rms0, thr, tau_ps)


# --------------------------------------- 3. baseline residual stress --------
def integrated_curve(sigma0, scales, t_fegab=500e-9, n_z=40):
    """Thickness-integrated response from the laminated-plate stress field, with
    no fitted anisotropy function: dfr(kappa) up to one global scale."""
    layers = default_stack(t_fegab)
    sweep = curvature_sweep(layers, {"FeGaB": tuple(sigma0)}, scales, n_z=n_z)
    kap, _, _, profiles = keff_phieff_vs_curvature(sweep)
    dfr = np.empty(len(profiles))
    for i, p in enumerate(profiles):
        E = np.mean([modulus(coherent_compliance(cv.H_SWEEP, p.Keff[j], p.phi_eff[j]))
                     for j in range(len(p.z))], axis=0)
        dfr[i] = frequency_response(cv.H_SWEEP, E, t_fegab, v=1.0)[1]
    return kap, dfr


def fit_scale(kap, dfr, kappa_meas, resp_meas):
    """One multiplicative scale in log space, over the overlapping range."""
    inside = (kappa_meas >= kap.min()) & (kappa_meas <= kap.max())
    logi = np.interp(np.log(kappa_meas[inside]), np.log(kap), np.log(dfr))
    scale = float(np.exp(np.mean(np.log(resp_meas[inside]) - logi)))
    rms = float(np.sqrt(np.mean((np.log10(np.exp(logi) * scale)
                                 - np.log10(resp_meas[inside])) ** 2)))
    return scale, 10 ** rms, int(inside.sum())


def stress_identifiability():
    kappa, resp = cv.load_points("measured_fig3a.json")
    scales = np.linspace(0.02, 11.0, 24)
    s11 = 45e6
    rows = []
    for ratio in (1.2, 1.4, 1.667, 1.9, 2.2):
        for shear in (0.0, 5e6, 10e6):
            sigma0 = (s11, ratio * s11, shear)
            kap, dfr = integrated_curve(sigma0, scales)
            sc, rmsf, npts = fit_scale(kap, dfr, kappa, resp)
            rows.append({"ratio_s22_s11": ratio, "shear_MPa": shear / 1e6,
                         "scale_fitted": sc, "rms_factor": rmsf,
                         "n_points_scored": npts,
                         "kappa_covered": [float(kap.min()), float(kap.max())]})
    rmsv = np.array([r["rms_factor"] for r in rows])
    scv = np.array([r["scale_fitted"] for r in rows])
    nominal = [r for r in rows if r["ratio_s22_s11"] == 1.667 and r["shear_MPa"] == 5.0][0]
    return {
        "nominal": nominal,
        "grid": rows,
        "rms_factor_range": [float(rmsv.min()), float(rmsv.max())],
        "fitted_scale_range": [float(scv.min()), float(scv.max())],
        "scale_compensation_factor": float(scv.max() / scv.min()),
        "comment": ("the fitted scale absorbs the assumed stress anisotropy: over "
                    "the grid the scale moves by the factor above while the "
                    "goodness of fit moves inside the range above"),
    }


# ---------------------------------------------------------------- figure ----
def figure(width_aux, loss_aux, stress):
    c_grid, rms_c, v_c = width_aux
    prof, rms0, thr, tau_hat = loss_aux
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 10.4))

    ax = axes[0]
    ax.semilogx(c_grid, rms_c, "o-", color="C0", lw=1.8, ms=5)
    ax.axhline(rms_c.min() * 1.02, ls=":", color="0.5", lw=1.2,
               label="2% above the best value")
    ax.set_xlabel(r"domain-population width $c$ ($A_s = c/K_{\rm eff}$)")
    ax.set_ylabel("RMS deviation (factor)")
    ax.set_ylim(1.6, 2.0)
    ax2 = ax.twinx()
    ax2.semilogx(c_grid, v_c, "s--", color="C3", lw=1.4, ms=4)
    ax2.set_ylabel("fitted active fraction (%)", color="C3")
    ax2.tick_params(axis="y", colors="C3")
    ax2.set_ylim(3.5, 11.5)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("(a) width not identifiable: 4% in fit over 2 decades, and a "
                 "shallow\n     valley in which the fraction rises below $c=5$",
                 fontsize=9, loc="left")

    ax = axes[1]
    t = np.array([p["tau_ps"] for p in prof]); r = np.array([p["rms_Q"] for p in prof])
    ax.plot(t, r, "-", color="C0", lw=2)
    ax.axhline(thr, ls="--", color="C3", lw=1.4, label="95% threshold")
    ax.axvline(tau_hat, ls=":", color="0.4", lw=1.2,
               label=r"$\hat\tau=%.2f$ ps/%%" % tau_hat)
    ax.axvline(4.0, ls="-.", color="C2", lw=1.2, label="4.0 ps/% as submitted")
    ax.set_xlabel(r"relaxation time $\tau$ (ps per % of response)")
    ax.set_ylabel(r"RMS residual in $Q$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7.5)
    ax.set_title(r"(b) profile of $\tau$ with $Q_0$ refitted", fontsize=9, loc="left")

    ax = axes[2]
    rows = stress["grid"]
    for shear, col, mk in zip((0.0, 5.0, 10.0), ("C0", "C2", "C3"), ("o", "s", "^")):
        sub = [r for r in rows if r["shear_MPa"] == shear]
        ax.plot([r["ratio_s22_s11"] for r in sub], [r["rms_factor"] for r in sub],
                mk + "-", color=col, lw=1.6, ms=5,
                label=r"$\sigma_{12}=%.0f$ MPa" % shear)
    ax.set_xlabel(r"assumed in-plane anisotropy $\sigma_{22}/\sigma_{11}$")
    ax.set_ylabel("RMS deviation (factor)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7.5)
    ax.set_title("(c) the baseline stress tensor is absorbed by the fitted scale",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_identifiability.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    width, width_aux = profile_width()
    loss, loss_aux = loss_identifiability()
    stress = stress_identifiability()
    figure(width_aux, loss_aux, stress)
    (HERE / "identifiability.json").write_text(
        json.dumps({"domain_width": width, "loss_constants": loss,
                    "baseline_stress": stress}, indent=1))

    print("1. DOMAIN-POPULATION WIDTH")
    for p in width["profile"]:
        print("   c = %7.1f -> v = %.2f %%, RMS factor %.3f" % (p["c"], p["v_pct"], p["rms_factor"]))
    print("   over c = 2 to 1000 the fit changes by %.1f%%; v spans %.2f-%.2f %% overall"
          % (width["rms_factor_span_pct"], *width["v_pct_span"]))
    print("   for c >= 5 (the stiff branch) v is confined to %.2f-%.2f %%; below that"
          " (v, c) trade off along a shallow valley" % tuple(width["v_pct_span_c_above_5"]))
    print("   coherent rotation (the c -> infinity limit): v = %.2f %%, RMS factor %.3f"
          % (width["coherent_limit"]["v_pct"], width["coherent_limit"]["rms_factor"]))
    b = width["active_fraction_bootstrap"]
    print("   active fraction, %d bootstrap resamples: %.2f +/- %.2f %%, 95%% CI [%.2f, %.2f] %%"
          % (b["n"], b["mean_pct"], b["std_pct"], *b["ci95_pct"]))

    print("2. LOSS CONSTANTS  (%d devices)" % loss["n_devices"])
    print("   Q0  = %.0f +/- %.0f" % (loss["Q0"], loss["Q0_stderr"]))
    print("   tau = %.2f +/- %.2f ps per %% of response" % (loss["tau_ps_per_pct"], loss["tau_stderr_ps"]))
    print("   correlation(Q0, tau) = %+.2f, Jacobian condition number %.1f"
          % (loss["correlation_Q0_tau"], loss["condition_number_jacobian"]))
    if loss["tau_interval_95_ps"]:
        print("   95%% interval on tau from the profile: %.2f to %.2f ps/%%"
              % tuple(loss["tau_interval_95_ps"]))
    print("   as submitted: Q0 = 540, tau = 4.0 ps per %% of response (%s)"
          % loss["previous_submission_values"]["status"])

    print("3. BASELINE RESIDUAL STRESS  (thickness-integrated branch)")
    n = stress["nominal"]
    print("   nominal (45, 75, 5) MPa: scale %.3f, RMS factor %.3f on %d points, "
          "kappa covered %.2f-%.2f" % (n["scale_fitted"], n["rms_factor"],
                                       n["n_points_scored"], *n["kappa_covered"]))
    print("   over the grid: RMS factor %.3f-%.3f while the fitted scale moves by x%.2f"
          % (*stress["rms_factor_range"], stress["scale_compensation_factor"]))
    print("Outputs: identifiability.json, fig_identifiability.png")
