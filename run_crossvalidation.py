"""
Held-out validation of the global constants against the measured population.

Three questions are answered, each with the parameters estimated on devices the
prediction never saw:

  1. RESPONSE, interpolation.  Fit the single global active fraction on half the
     devices, predict the other half.  Repeated over a deterministic interleaved
     split, 5 folds, and 200 stratified random halves.

  2. RESPONSE, extrapolation.  Fit on the low-curvature half and predict the
     high-curvature half, then the reverse.  The fitted fraction is a pure
     vertical offset in log-log, so this split tests the predicted SHAPE of
     dfr(kappa), which no scale factor can repair.

  3. LOSS, transfer across device families.  Fit the two shared constants
     (Q0, tau) on one fixed-frequency family and predict the other, which sits
     at a different resonance frequency (246 MHz against 138 MHz).  The same
     transfer is applied to the four-constant form of Eq. (4) of [1], whose
     damping constants absorb f_r and therefore cannot travel between families.

Reference point throughout: the two-domain model published in [1], recovered
from its Fig. 6, which places one fitted active fraction per device.  Its score
is in-sample by construction, including on our held-out devices.

Outputs: crossvalidation.json, fig_crossvalidation.png
Run:     python3 run_crossvalidation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

import crossval as cv

HERE = Path(__file__).parent
MODELS = ("coherent", "averaged")
LABEL = {"coherent": "coherent rotation", "averaged": "energy-averaged domains"}


# ------------------------------------------------------------- response -----
def published_model_on(kappa):
    """The model curve of [1] (its Fig. 6), interpolated at our curvatures."""
    k10, v10 = cv.load_points("model_fig10.json")
    return np.exp(np.interp(np.log(kappa), np.log(k10), np.log(v10)))


def score_split(kappa, resp, tr, te, model):
    v, rms_tr = cv.fit_v(kappa[tr], resp[tr], model)
    pred_te = cv.predict(kappa[te], model, v)
    return {"v_pct": 100 * v,
            "rms_factor_train": 10 ** rms_tr,
            "rms_factor_test": 10 ** cv.rms_log(pred_te, resp[te]),
            "median_dev_test_pct": 100 * float(np.median(np.abs(pred_te / resp[te] - 1)))}


def response_crossvalidation(seed=20260919, n_random=200):
    kappa, resp = cv.load_points("measured_fig3a.json")
    n = kappa.size
    rng = np.random.default_rng(seed)
    out = {"n_devices": n, "models": {}}

    # reference: the published per-device-fitted model, scored everywhere
    pub = published_model_on(kappa)
    out["published_model_in_sample_rms_factor"] = 10 ** cv.rms_log(pub, resp)

    for model in MODELS:
        m = {}
        # in-sample, for comparison
        v_all, rms_all = cv.fit_v(kappa, resp, model)
        m["in_sample"] = {"v_pct": 100 * v_all, "rms_factor": 10 ** rms_all}

        tr, te = cv.interleaved(n)
        m["interleaved"] = score_split(kappa, resp, tr, te, model)
        m["interleaved"]["published_rms_factor_on_test"] = \
            10 ** cv.rms_log(pub[te], resp[te])

        folds = cv.kfold(n, 5, np.random.default_rng(seed))
        fold_scores = []
        for i in range(5):
            te_f = folds[i]
            tr_f = np.sort(np.concatenate([folds[j] for j in range(5) if j != i]))
            fold_scores.append(score_split(kappa, resp, tr_f, te_f, model))
        m["kfold5"] = {
            "folds": fold_scores,
            "mean_rms_factor_test": float(np.mean([f["rms_factor_test"] for f in fold_scores])),
            "v_pct_mean": float(np.mean([f["v_pct"] for f in fold_scores])),
            "v_pct_std": float(np.std([f["v_pct"] for f in fold_scores])),
        }

        rand = []
        for _ in range(n_random):
            tr_r, te_r = cv.stratified_random(n, rng)
            rand.append(score_split(kappa, resp, tr_r, te_r, model))
        rt = np.array([s["rms_factor_test"] for s in rand])
        vv = np.array([s["v_pct"] for s in rand])
        m["random_halves"] = {
            "n_repeats": n_random,
            "rms_factor_test_mean": float(rt.mean()),
            "rms_factor_test_p05": float(np.percentile(rt, 5)),
            "rms_factor_test_p95": float(np.percentile(rt, 95)),
            "v_pct_mean": float(vv.mean()), "v_pct_std": float(vv.std()),
            "v_pct_min": float(vv.min()), "v_pct_max": float(vv.max()),
        }

        ex = {}
        for tag, low_train in (("train_low_kappa", True), ("train_high_kappa", False)):
            tr_e, te_e = cv.extrapolation(n, low_train)
            s = score_split(kappa, resp, tr_e, te_e, model)
            s["kappa_train"] = [float(kappa[tr_e].min()), float(kappa[tr_e].max())]
            s["kappa_test"] = [float(kappa[te_e].min()), float(kappa[te_e].max())]
            s["published_rms_factor_on_test"] = 10 ** cv.rms_log(pub[te_e], resp[te_e])
            ex[tag] = s
        m["extrapolation"] = ex
        out["models"][model] = m
    return out, (kappa, resp, pub)


# ----------------------------------------------------------------- loss -----
SUBSETS = (("measured_fig6c.json", 246e6, "designs 3, 4"),
           ("measured_fig6d.json", 138e6, "designs 7, 8"))


def _fit_shared(sets, start=(1000.0 / 540.0, 4.0)):
    """1/Q = 1/Q0 + 2 pi f_r tau x, fitted jointly on the given families.

    The two unknowns are carried as u = (1000/Q0, tau in ps) so that both are of
    order unity.  With the raw (1/Q0, tau in s) pair the numerical Jacobian of
    least_squares is degenerate across the nine decades that separate them and
    the solver returns its own starting point unchanged, which is how the values
    Q0 = 540 and tau = 4.0 ps of the first submission arose: they were the
    initial guess, not a fit.  Rescaled, the fit converges to the same optimum
    from any start (verified over Q0 in 300-2000, tau in 1-20 ps).
    """
    def resid(u):
        q0, tau = 1000.0 / u[0], u[1] * 1e-12
        return np.concatenate([q - 1.0 / (1.0 / q0 + 2 * np.pi * f * tau * x)
                               for x, q, f in sets])
    sol = least_squares(resid, list(start))
    return 1000.0 / sol.x[0], float(sol.x[1] * 1e-12)


def _predict_shared(x, f_r, q0, tau):
    return 1.0 / (1.0 / q0 + 2 * np.pi * f_r * tau * x)


def _fit_eq4(x, q, f_r):
    """Eq. (4) of [1]: Q = 2 pi f_r / (gamma_NM + gamma_M x/x_max)."""
    xm = x.max()

    def model(xx, g_nm, g_m):
        return 2 * np.pi * f_r / (g_nm * 1e8 + g_m * 1e8 * xx / xm)
    (g_nm, g_m), _ = curve_fit(model, x, q, p0=[0.02, 0.05], maxfev=40000)
    return float(g_nm), float(g_m)


def _predict_eq4(x, f_r, g_nm, g_m, x_max):
    return 2 * np.pi * f_r / (g_nm * 1e8 + g_m * 1e8 * x / x_max)


def _r2(q, pred):
    return float(1.0 - np.sum((q - pred) ** 2) / np.sum((q - q.mean()) ** 2))


def loss_transfer():
    fam = []
    for fname, f_r, tag in SUBSETS:
        x, q = cv.load_points(fname)
        fam.append({"tag": tag, "f_r": f_r, "x": x, "q": q})

    out = {"families": [{"tag": f["tag"], "f_r_MHz": f["f_r"] / 1e6,
                         "n": int(f["x"].size)} for f in fam]}

    q0_all, tau_all = _fit_shared([(f["x"], f["q"], f["f_r"]) for f in fam])
    out["joint_fit"] = {"Q0": q0_all, "tau_ps_per_pct": tau_all * 1e12,
                        "R2": _r2(np.concatenate([f["q"] for f in fam]),
                                  np.concatenate([_predict_shared(f["x"], f["f_r"],
                                                                  q0_all, tau_all)
                                                  for f in fam]))}

    trans = []
    for i, j in ((0, 1), (1, 0)):
        src, tgt = fam[i], fam[j]
        q0, tau = _fit_shared([(src["x"], src["q"], src["f_r"])])
        pred_shared = _predict_shared(tgt["x"], tgt["f_r"], q0, tau)
        g_nm, g_m = _fit_eq4(src["x"], src["q"], src["f_r"])
        pred_eq4 = _predict_eq4(tgt["x"], tgt["f_r"], g_nm, g_m, tgt["x"].max())
        trans.append({
            "fitted_on": src["tag"], "predicted": tgt["tag"],
            "f_r_source_MHz": src["f_r"] / 1e6, "f_r_target_MHz": tgt["f_r"] / 1e6,
            "shared_two_constants": {
                "Q0": q0, "tau_ps_per_pct": tau * 1e12,
                "R2_on_target": _r2(tgt["q"], pred_shared),
                "median_abs_dev_pct": 100 * float(np.median(np.abs(pred_shared / tgt["q"] - 1)))},
            "paper_eq4_four_constants": {
                "gamma_NM": g_nm, "gamma_M": g_m,
                "R2_on_target": _r2(tgt["q"], pred_eq4),
                "median_abs_dev_pct": 100 * float(np.median(np.abs(pred_eq4 / tgt["q"] - 1)))},
        })
    out["transfer"] = trans

    # leave-one-out stability of the two shared constants on the pooled families
    xs = np.concatenate([f["x"] for f in fam])
    qs = np.concatenate([f["q"] for f in fam])
    fs = np.concatenate([np.full(f["x"].size, f["f_r"]) for f in fam])
    q0s, taus = [], []
    for k in range(xs.size):
        m = np.ones(xs.size, bool); m[k] = False
        sets = [(xs[m & (fs == f)], qs[m & (fs == f)], f) for f in np.unique(fs)]
        q0k, tauk = _fit_shared(sets)
        q0s.append(q0k); taus.append(tauk * 1e12)
    out["leave_one_out"] = {
        "n": int(xs.size),
        "Q0_mean": float(np.mean(q0s)), "Q0_std": float(np.std(q0s)),
        "Q0_min": float(np.min(q0s)), "Q0_max": float(np.max(q0s)),
        "tau_mean_ps": float(np.mean(taus)), "tau_std_ps": float(np.std(taus)),
        "tau_min_ps": float(np.min(taus)), "tau_max_ps": float(np.max(taus)),
    }
    return out, fam


# ---------------------------------------------------------------- figure ----
def figure(resp_data, resp_res, loss_res, fam):
    """Panel (a) of the original two-panel figure only.

    The loss-transfer panel moved to the supplementary material once
    run_loss_scaling.py replaced it in the main text: comparing the loss across
    platforms and frequencies says more than comparing two families 1.8 apart
    in frequency.
    """
    kappa, meas, pub = resp_data
    n = kappa.size
    fig, ax = plt.subplots(figsize=(5.4, 3.9))

    tr, te = cv.extrapolation(n, True)
    v_low, _ = cv.fit_v(kappa[tr], meas[tr], "coherent")
    tr2, te2 = cv.extrapolation(n, False)
    v_high, _ = cv.fit_v(kappa[tr2], meas[tr2], "coherent")
    kk = np.linspace(kappa.min(), kappa.max(), 50)
    ax.semilogy(kappa[tr], meas[tr], "o", ms=5, mfc="C0", mec="C0", alpha=0.75,
                label=r"low-$\kappa$ half")
    ax.semilogy(kappa[te], meas[te], "s", ms=5, mfc="none", mec="C3", mew=1.2,
                label=r"high-$\kappa$ half")
    ax.semilogy(kk, cv.predict(kk, "coherent", v_low), "-", color="C0", lw=2,
                label=r"calibrated on low $\kappa$ ($v=%.1f\%%$)" % (100 * v_low))
    ax.semilogy(kk, cv.predict(kk, "coherent", v_high), "--", color="C3", lw=2,
                label=r"calibrated on high $\kappa$ ($v=%.1f\%%$)" % (100 * v_high))
    ax.set_xlabel(r"curvature $\kappa$ (mm$^{-1}$)")
    ax.set_ylabel(r"$\Delta f_r/f_{r,\min}$ (%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_crossvalidation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    resp, resp_data = response_crossvalidation()
    loss, fam = loss_transfer()
    figure(resp_data, resp, loss, fam)
    (HERE / "crossvalidation.json").write_text(
        json.dumps({"response": resp, "loss": loss}, indent=1))

    print("HELD-OUT RESPONSE  (%d measured devices)" % resp["n_devices"])
    print("  reference: model of [1], one fraction fitted per device, "
          "in-sample RMS factor %.2f" % resp["published_model_in_sample_rms_factor"])
    for model in MODELS:
        m = resp["models"][model]
        print("  %s" % LABEL[model])
        print("    in sample          : v=%.2f%%  RMS factor %.3f"
              % (m["in_sample"]["v_pct"], m["in_sample"]["rms_factor"]))
        s = m["interleaved"]
        print("    interleaved halves : v=%.2f%%  train %.3f  TEST %.3f  "
              "([1] on same test points: %.3f)"
              % (s["v_pct"], s["rms_factor_train"], s["rms_factor_test"],
                 s["published_rms_factor_on_test"]))
        s = m["kfold5"]
        print("    5-fold             : mean TEST %.3f   v = %.2f +/- %.2f %%"
              % (s["mean_rms_factor_test"], s["v_pct_mean"], s["v_pct_std"]))
        s = m["random_halves"]
        print("    200 random halves  : TEST %.3f [%.3f, %.3f] (5-95%%)  "
              "v = %.2f +/- %.2f %% (range %.2f-%.2f)"
              % (s["rms_factor_test_mean"], s["rms_factor_test_p05"],
                 s["rms_factor_test_p95"], s["v_pct_mean"], s["v_pct_std"],
                 s["v_pct_min"], s["v_pct_max"]))
        for tag, s in m["extrapolation"].items():
            print("    %-18s : v=%.2f%%  train %.3f  TEST %.3f  "
                  "(kappa %.2f-%.2f -> %.2f-%.2f, [1]: %.3f)"
                  % (tag, s["v_pct"], s["rms_factor_train"], s["rms_factor_test"],
                     *s["kappa_train"], *s["kappa_test"],
                     s["published_rms_factor_on_test"]))

    print("LOSS CONSTANTS TRANSFERRED ACROSS DEVICE FAMILIES")
    j = loss["joint_fit"]
    print("  joint fit: Q0 = %.0f, tau = %.2f ps/%%, R2 = %.2f"
          % (j["Q0"], j["tau_ps_per_pct"], j["R2"]))
    for t in loss["transfer"]:
        s, p = t["shared_two_constants"], t["paper_eq4_four_constants"]
        print("  fit on %-11s (%3.0f MHz) -> predict %-11s (%3.0f MHz)"
              % (t["fitted_on"], t["f_r_source_MHz"], t["predicted"], t["f_r_target_MHz"]))
        print("      two shared constants : Q0=%.0f tau=%.2f ps/%%  R2 = %+.2f  "
              "median dev %3.0f%%" % (s["Q0"], s["tau_ps_per_pct"], s["R2_on_target"],
                                      s["median_abs_dev_pct"]))
        print("      Eq. (4) of [1]       : gNM=%.3f gM=%.3f        R2 = %+.2f  "
              "median dev %3.0f%%" % (p["gamma_NM"], p["gamma_M"], p["R2_on_target"],
                                      p["median_abs_dev_pct"]))
    l = loss["leave_one_out"]
    print("  leave-one-out over %d devices: Q0 = %.0f +/- %.0f (%.0f-%.0f), "
          "tau = %.2f +/- %.2f ps/%% (%.2f-%.2f)"
          % (l["n"], l["Q0_mean"], l["Q0_std"], l["Q0_min"], l["Q0_max"],
             l["tau_mean_ps"], l["tau_std_ps"], l["tau_min_ps"], l["tau_max_ps"]))
    print("Outputs: crossvalidation.json, fig_crossvalidation.png")
