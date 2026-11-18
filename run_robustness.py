"""
Uncertainty and robustness of the loss-scaling and detectivity conclusions.

Written for the review of Sensors-114910-2026 (Reviewer 1, comments 2, 4, 5).
Nothing in the models changes; this script only asks how much the published
numbers move when the inputs are allowed to move.

  1. Loss per unit softening, Lambda = [1/Q - 1/Q_sat]/x, with an uncertainty
     on every point:
       - the two device families of [1]: bootstrap over devices (Q0 and Lambda
         refitted on each resample);
       - Nan 2013 [33]: x uniform between the two readings of its frequency
         swing, each quoted Q uniform within +/-10 % (the values are rounded in
         the text);
       - Durdaut 2020 [34]: each digitised Q uniform within +/-5 %.
     The exponent n of Lambda ~ f^n is refitted on every Monte Carlo draw, for
     the full set, without the 7.4 kHz point, and leaving out one platform.

  2. The size of a platform effect that would be needed to rescue the
     relaxation form (n = 1) or the inverse form of Eq. (4) of [1] (n = -1),
     compared with the platform effects that can actually be observed.

  3. Loss-limited detectivity: bootstrap over devices of the measured and
     predicted compression, and of the log-log slope of detectivity against
     response, which is less sensitive to the two extreme devices than a
     max/min ratio.  A +/-5 % uncertainty on every measured Q is propagated.
     The resonance frequency cancels in the normalised detectivity of a family
     and therefore carries no uncertainty into it.

  4. Joint 95 % confidence region of (Q0, Lambda) from the F-test on the sum of
     squared residuals, against the leave-one-out intervals of the paper.

Outputs: robustness.json, fig_loss_scaling_ci.png (Fig. 3), fig_joint_region.png (Fig. S9)
Run:     python3 run_robustness.py     (deterministic, fixed seed)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import f as fdist

import crossval as cv

HERE = Path(__file__).parent
SEED = 20261118
N_MC = 20000
N_BOOT = 2000
FAMILIES = (("measured_fig6c.json", 246e6, "[1], 246 MHz"),
            ("measured_fig6d.json", 138e6, "[1], 138 MHz"))
NAN = {"f": 215e6, "Q_wp": 250.0, "Q_sat": 1400.0, "x": (0.74, 1.10), "dQ": 0.10}
DUR = {"f": 7422.0, "Q_wp": 825.0, "Q_sat": 1071.0, "x": 0.500, "dQ": 0.05}
DQ_MEAS = 0.05


def fit_q0_lambda(x, q):
    def resid(u):
        return q - 1.0 / (1.0 / (1000.0 / u[0]) + u[1] * 1e-3 * x)
    s = least_squares(resid, [1000 / 523.0, 5.0])
    return 1000.0 / s.x[0], s.x[1] * 1e-3, s


def slope(f, lam):
    return np.polyfit(np.log(f), np.log(lam), 1)[0]


# ------------------------------------------------------ 1. loss scaling ----
def loss_scaling(rng):
    fam = [cv.load_points(n) + (fr, tag) for n, fr, tag in FAMILIES]
    boot = []
    for x, q, fr, tag in fam:
        lam = np.empty(N_MC)
        for i in range(N_MC):
            j = rng.integers(0, x.size, x.size)
            lam[i] = fit_q0_lambda(x[j], q[j])[1]
        boot.append(lam)

    def ext_draw(e, n):
        qw = e["Q_wp"] * rng.uniform(1 - e["dQ"], 1 + e["dQ"], n)
        qs = e["Q_sat"] * rng.uniform(1 - e["dQ"], 1 + e["dQ"], n)
        x = (rng.uniform(*e["x"], n) if isinstance(e["x"], tuple)
             else np.full(n, e["x"]))
        return (1 / qw - 1 / qs) / x

    lam_nan = ext_draw(NAN, N_MC)
    lam_dur = ext_draw(DUR, N_MC)
    # a resample can return a non-positive Lambda (no loss resolved): such draws
    # carry no exponent and are counted, then dropped jointly from every set
    ok = (lam_nan > 0) & (lam_dur > 0) & (boot[0] > 0) & (boot[1] > 0)
    n_dropped = int((~ok).sum())

    F = {"246": FAMILIES[0][1], "138": FAMILIES[1][1], "nan": NAN["f"], "dur": DUR["f"]}
    L = {"246": boot[0], "138": boot[1], "nan": lam_nan, "dur": lam_dur}
    sets = {
        "all four points": ("246", "138", "nan", "dur"),
        "without the 7.4 kHz point": ("246", "138", "nan"),
        "without [33]": ("246", "138", "dur"),
        "without [1]": ("nan", "dur"),
        "[1] alone (same wafer, same process)": ("246", "138"),
    }
    out = {}
    for name, keys in sets.items():
        f = np.array([F[k] for k in keys])
        lam = np.vstack([L[k][ok] for k in keys])
        n = np.array([slope(f, lam[:, i]) for i in range(lam.shape[1])])
        out[name] = {"points": list(keys), "frequency_span": float(f.max() / f.min()),
                     "n_median": float(np.median(n)),
                     "n_ci95": [float(np.percentile(n, 2.5)), float(np.percentile(n, 97.5))],
                     "excludes_n_eq_1": bool(np.percentile(n, 97.5) < 1.0),
                     "excludes_n_eq_minus_1": bool(np.percentile(n, 2.5) > -1.0)}

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)),
                float(np.percentile(a, 97.5))]
    points = {"[1], 246 MHz": ci(boot[0]), "[1], 138 MHz": ci(boot[1]),
              "Nan 2013, 215 MHz": ci(lam_nan), "Durdaut 2020, 7.4 kHz": ci(lam_dur)}

    # 2. platform effect needed to rescue n = 1 or n = -1
    ratio = L["246"][ok] / L["dur"][ok]          # observed growth, 7.4 kHz -> 246 MHz
    fspan = F["246"] / F["dur"]
    need_relax = fspan / ratio                    # factor a platform effect must supply
    need_inverse = ratio * fspan
    observed_platform = {
        "between the two families of [1] (same wafer)":
            float(np.median(L["246"] / L["138"])),
        "[33] against the nearer family of [1]":
            float(np.median(np.maximum(lam_nan / L["138"][:lam_nan.size],
                                       L["138"][:lam_nan.size] / lam_nan)))}
    rescue = {"observed_growth_ci95": ci(ratio), "frequency_span": float(fspan),
              "factor_needed_for_n_eq_1_ci95": ci(need_relax),
              "factor_needed_for_n_eq_minus_1_ci95": ci(need_inverse),
              "observed_platform_factors": observed_platform}
    rescue["draws_dropped_nonpositive_Lambda"] = n_dropped
    return out, points, rescue, (F, L, ok)


# -------------------------------------------------------- 3. detectivity ----
def detectivity(rng):
    """Bootstrap the 25 devices family by family, refit the two SHARED loss
    constants on the joint resample (as in the paper), and compare measured and
    predicted compression in each family."""
    data = [cv.load_points(n) for n, fr, tag in FAMILIES]

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)),
                float(np.percentile(a, 97.5))]

    def stats(xb, qb, q0, lam):
        d = (1.0 / qb) / xb
        dm = (1.0 / (1.0 / (1.0 / q0 + lam * xb))) / xb
        return (np.exp(np.ptp(np.log(xb)) - np.ptp(np.log(d))),
                np.exp(np.ptp(np.log(xb)) - np.ptp(np.log(dm))),
                -np.polyfit(np.log(xb), np.log(d), 1)[0],
                -np.polyfit(np.log(xb), np.log(dm), 1)[0])

    q0, lam, _ = fit_q0_lambda(np.concatenate([d[0] for d in data]),
                               np.concatenate([d[1] for d in data]))
    point = [stats(x, q, q0, lam) for x, q in data]
    acc = [[] for _ in data]
    for _ in range(N_BOOT):
        res = []
        for x, q in data:
            j = rng.integers(0, x.size, x.size)
            res.append((x[j], q[j] * rng.uniform(1 - DQ_MEAS, 1 + DQ_MEAS, x.size)))
        if any(np.ptp(xb) <= 0 for xb, _ in res):
            continue
        q0b, lamb, _ = fit_q0_lambda(np.concatenate([r[0] for r in res]),
                                     np.concatenate([r[1] for r in res]))
        for a, (xb, qb) in zip(acc, res):
            a.append(stats(xb, qb, q0b, lamb))
    out = []
    for (name, fr, tag), (x, q), pt, a in zip(FAMILIES, data, point, acc):
        a = np.array(a)
        out.append({"family": tag, "n": int(x.size),
                    "compression_measured_point": float(pt[0]),
                    "compression_predicted_point": float(pt[1]),
                    "compression_measured_ci95": ci(a[:, 0]),
                    "compression_predicted_ci95": ci(a[:, 1]),
                    "overstatement_ratio_ci95": ci(a[:, 1] / a[:, 0]),
                    "slope_measured_ci95": ci(a[:, 2]),
                    "slope_predicted_ci95": ci(a[:, 3]),
                    "slope_if_Q_constant": 1.0,
                    "measured_slope_below_1": bool(np.percentile(a[:, 2], 97.5) < 1.0)})
    return out


# ------------------------------------------------- 4. joint (Q0, Lambda) ----
def joint_region():
    xs, qs = [], []
    for name, fr, tag in FAMILIES:
        x, q = cv.load_points(name)
        xs.append(x); qs.append(q)
    x = np.concatenate(xs); q = np.concatenate(qs)
    q0, lam, sol = fit_q0_lambda(x, q)
    sse0 = float(np.sum(sol.fun ** 2))
    n, p = x.size, 2
    thr = sse0 * (1 + p / (n - p) * fdist.ppf(0.95, p, n - p))
    Q0g = np.linspace(330, 900, 240)
    Lg = np.linspace(0.5e-3, 14e-3, 240)
    S = np.empty((Lg.size, Q0g.size))
    for i, L in enumerate(Lg):
        for k, Q0 in enumerate(Q0g):
            S[i, k] = np.sum((q - 1 / (1 / Q0 + L * x)) ** 2)
    inside = S <= thr
    li = np.where(inside.any(axis=1))[0]
    ki = np.where(inside.any(axis=0))[0]
    return {"Q0": q0, "Lambda": lam, "n": n, "sse_min": sse0, "sse_95": thr,
            "Q0_range_95": [float(Q0g[ki.min()]), float(Q0g[ki.max()])],
            "Lambda_range_95_permille": [float(1e3 * Lg[li.min()]), float(1e3 * Lg[li.max()])],
            "edge_of_grid": bool(inside[0].any() or inside[-1].any()
                                 or inside[:, 0].any() or inside[:, -1].any())}, (Q0g, Lg, S, thr)


def figure(ls, points, F_L, joint_aux, jr):
    F, L, ok = F_L
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    lab = {"246": "[1], 246 MHz", "138": "[1], 138 MHz",
           "nan": "Nan 2013, 215 MHz", "dur": "Durdaut 2020, 7.4 kHz"}
    mk = {"246": "o", "138": "o", "nan": "s", "dur": "^"}
    for k in ("246", "138", "nan", "dur"):
        lo, med, hi = np.percentile(L[k][ok], [2.5, 50, 97.5])
        ax.errorbar(F[k], med, yerr=[[med - lo], [hi - med]], fmt=mk[k], ms=7,
                    mfc="C0" if k in ("246", "138") else "none", mec="k",
                    ecolor="k", capsize=3, label=lab[k])
    ff = np.geomspace(3e3, 5e8, 50)
    a = ls["all four points"]
    ref = np.median(L["246"][ok])
    keys = ("246", "138", "nan", "dur")
    lf = np.log([F[k] for k in keys]); ll = np.log([np.median(L[k][ok]) for k in keys])
    c_fit = np.mean(ll - a["n_median"] * lf)        # intercept at the median exponent
    ax.loglog(ff, np.exp(c_fit) * ff ** a["n_median"], "-", color="C0", lw=1.5,
              label="fit, n = %.2f [%.2f, %.2f]" % (a["n_median"], *a["n_ci95"]))
    for n_, st, c, lb in ((1.0, "--", "C3", "relaxation, n = 1"),
                          (-1.0, ":", "C2", "Eq. (4) of [1], n = -1")):
        ax.loglog(ff, ref * (ff / F["246"]) ** n_, st, color=c, lw=1.5, label=lb)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e-1)
    ax.set_xlabel(r"resonance frequency $f_r$ (Hz)")
    ax.set_ylabel(r"$\Lambda$ (per %), median and 95% interval")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_loss_scaling_ci.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    Q0g, Lg, S, thr = joint_aux
    ax.contourf(Q0g, 1e3 * Lg, S <= thr, levels=[0.5, 1.5], colors=["C0"], alpha=0.25)
    ax.contour(Q0g, 1e3 * Lg, S, levels=[thr], colors="C0", linewidths=1.5)
    ax.plot(jr["Q0"], 1e3 * jr["Lambda"], "k+", ms=10, mew=1.8, label="best fit")
    ax.errorbar(523, 4.81, xerr=16, yerr=0.20, fmt="none", ecolor="C3", capsize=3,
                label="leave-one-out spread")
    ax.set_xlabel(r"$Q_0$")
    ax.set_ylabel(r"$\Lambda$ (per mille per %)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("joint 95% region, 25 devices of [1]", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_joint_region.png", dpi=160)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    ls, points, rescue, F_L = loss_scaling(rng)
    det = detectivity(rng)
    jr, jaux = joint_region()
    out = {"loss_scaling_exponent": ls, "Lambda_points_ci95_median": points,
           "platform_effect_needed": rescue, "detectivity": det,
           "joint_region_Q0_Lambda": jr,
           "assumptions": {"Nan_Q_uniform_rel": NAN["dQ"], "Durdaut_Q_uniform_rel": DUR["dQ"],
                           "measured_Q_uniform_rel": DQ_MEAS, "seed": SEED,
                           "n_monte_carlo": N_MC, "n_bootstrap": N_BOOT}}
    (HERE / "robustness.json").write_text(json.dumps(out, indent=1))
    figure(ls, points, F_L, jaux, jr)

    print("1. EXPONENT OF Lambda ~ f^n (median, 95% interval)")
    for k, v in ls.items():
        print("   %-40s span x%-8.0f n = %+.2f [%+.2f, %+.2f]  excl. n=1: %s  excl. n=-1: %s"
              % (k, v["frequency_span"], v["n_median"], *v["n_ci95"],
                 v["excludes_n_eq_1"], v["excludes_n_eq_minus_1"]))
    print("   Lambda per point (2.5/50/97.5 %, per %):")
    for k, v in points.items():
        print("     %-24s %.2e  %.2e  %.2e" % (k, *v))
    print("2. PLATFORM EFFECT NEEDED")
    print("   observed growth 7.4 kHz -> 246 MHz: x%.1f [%.1f, %.1f] over f x%.0f"
          % (rescue["observed_growth_ci95"][1], rescue["observed_growth_ci95"][0],
             rescue["observed_growth_ci95"][2], rescue["frequency_span"]))
    print("   to rescue n=1 : factor %.0f [%.0f, %.0f]" % tuple(rescue["factor_needed_for_n_eq_1_ci95"][i] for i in (1, 0, 2)))
    print("   to rescue n=-1: factor %.2e [%.2e, %.2e]" % tuple(rescue["factor_needed_for_n_eq_minus_1_ci95"][i] for i in (1, 0, 2)))
    for k, v in rescue["observed_platform_factors"].items():
        print("   observed, %s: x%.2f" % (k, v))
    print("   draws dropped (non-positive Lambda): %d of %d" % (rescue["draws_dropped_nonpositive_Lambda"], N_MC))
    print("3. DETECTIVITY (bootstrap, +/-5% on Q, shared constants)")
    for d in det:
        print("   %s point values: measured %.2f, predicted %.2f" % (d["family"], d["compression_measured_point"], d["compression_predicted_point"]))
        print("   %s: compression measured %.2f [%.2f, %.2f], predicted %.2f [%.2f, %.2f], "
              "over-statement x%.2f [%.2f, %.2f]"
              % (d["family"], d["compression_measured_ci95"][1], d["compression_measured_ci95"][0],
                 d["compression_measured_ci95"][2], d["compression_predicted_ci95"][1],
                 d["compression_predicted_ci95"][0], d["compression_predicted_ci95"][2],
                 d["overstatement_ratio_ci95"][1], d["overstatement_ratio_ci95"][0],
                 d["overstatement_ratio_ci95"][2]))
        print("      slope of log D vs log x: measured %.2f [%.2f, %.2f], predicted %.2f [%.2f, %.2f] (1 if Q constant)"
              % (d["slope_measured_ci95"][1], d["slope_measured_ci95"][0], d["slope_measured_ci95"][2],
                 d["slope_predicted_ci95"][1], d["slope_predicted_ci95"][0], d["slope_predicted_ci95"][2]))
    print("4. JOINT 95%% REGION: Q0 %.0f-%.0f, Lambda %.1f-%.1f per mille per %%, edge of grid: %s"
          % (*jr["Q0_range_95"], *jr["Lambda_range_95_permille"], jr["edge_of_grid"]))


if __name__ == "__main__":
    main()
