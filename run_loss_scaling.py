"""
What actually transfers in the magnetoelastic loss, and how it scales with
frequency.

The first submission wrote the loss as a relaxation,

    1/Q = 1/Q0 + 2 pi f_r tau x ,     x = Delta f_r/f_r,min in percent,

and presented the transfer of (Q0, tau) between the two device families of [1]
as evidence for that form.  Two checks reported here show that the evidence was
weaker than claimed, and that the frequency factor is wrong.

  1. Within [1] the frequency exponent is not identifiable.  Profiling n in
     1/Q = 1/Q0 + A (f_r/f0)^n x over the 25 devices moves the residual by 3%
     between n = 0 and n = 1.  The two families sit at 138 and 246 MHz, a lever
     of 1.8, while the fitted loss coefficient differs between them by 2.1.  The
     lever is smaller than the scatter, so the exponent cannot be read off.

  2. Published devices at other frequencies settle it.  The quantity that can be
     compared across laboratories is the magnetoelastic loss per unit softening,

         Lambda = [1/Q(x) - 1/Q_sat] / x ,

     in which the non-magnetic loss cancels, so nothing needs to be known about
     the anchor or thermoelastic budget of someone else's device.  Measured on
     three platforms from 7.4 kHz to 246 MHz, Lambda grows by a factor of about
     12 while the frequency grows by a factor of 33000.

The conclusion is that the magnetoelastic loss is nearly rate-independent, as a
hysteretic domain-wall mechanism would be, and that both the relaxation form
assumed initially here and the inverse-frequency form of Eq. (4) of [1] are
excluded.  Note that the whole of the frequency lever comes from the single
low-frequency point: removing it returns an apparent exponent of 0.9 built from
points that span a factor of 1.8 in frequency, which means nothing.

External values, each traceable:

  Nan 2013, Sci. Rep. 3, 1985, doi 10.1038/srep01985, 215 MHz contour mode,
  AlN/(FeGaB/Al2O3)x10/Pt.  Q quoted verbatim in the text: 735 at zero bias,
  250 at the transition field, 1400 in saturation.  The frequency swing is not
  in the text, only in Fig. 2(d); both plausible readings are carried.

  Durdaut 2020, arXiv:2003.01085 / J. Microelectromech. Syst. 29, 1347, 7.42 kHz
  bending cantilever, poly-Si/AlN/FeCoSiB.  Q and f_res digitised here from the
  vector content of Fig. 2(a) and Fig. 3(b) of the preprint, calibrated on the
  printed axis ticks with residuals of 0.18 in Q and 0.000 in Hz.

Outputs: loss_scaling.json, fig_loss_scaling.png
Run:     python3 run_loss_scaling.py
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

HERE = Path(__file__).parent
F0 = 200e6
FAMILIES = (("measured_fig6c.json", 246e6, "[1], designs 3, 4"),
            ("measured_fig6d.json", 138e6, "[1], designs 7, 8"))


# ---------------------------------------------------- 1. exponent profile ---
def profile_exponent():
    xs, qs, frs = [], [], []
    for name, f_r, _ in FAMILIES:
        x, q = cv.load_points(name)
        xs.append(x); qs.append(q); frs.append(np.full(x.size, f_r))
    x = np.concatenate(xs); q = np.concatenate(qs); f = np.concatenate(frs)

    rows = []
    for n in (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3):
        def resid(u):
            return q - 1.0 / (1.0 / (1000.0 / u[0]) + u[1] * 1e-3 * (f / F0) ** n * x)
        s = least_squares(resid, [1000 / 523.0, 5.0])
        r = s.fun
        rows.append({"n": float(n), "Q0": 1000.0 / s.x[0], "A": float(s.x[1] * 1e-3),
                     "R2": float(1 - np.sum(r ** 2) / np.sum((q - q.mean()) ** 2)),
                     "rms_Q": float(np.sqrt(np.mean(r ** 2)))})
    best = min(rows, key=lambda r: r["rms_Q"])
    flat = [r for r in rows if r["rms_Q"] <= best["rms_Q"] * 1.05]
    return {"rows": rows, "best_n": best["n"],
            "n_within_5pct_of_best": [r["n"] for r in flat]}


# -------------------------------------------- 2. loss per unit softening ----
def lambda_per_family():
    out = []
    for name, f_r, tag in FAMILIES:
        x, q = cv.load_points(name)

        def resid(u):
            return q - 1.0 / (1.0 / (1000.0 / u[0]) + u[1] * 1e-3 * x)
        s = least_squares(resid, [1000 / 523.0, 5.0])
        out.append({"label": tag, "f_r_Hz": f_r, "Lambda": float(s.x[1] * 1e-3),
                    "Q0": 1000.0 / s.x[0], "n_devices": int(x.size),
                    "kind": "fitted over the population", "source": "[1]"})
    return out


EXTERNAL = [
    {"label": "Nan 2013, 215 MHz (upper reading)", "f_r_Hz": 215e6,
     "Q_wp": 250.0, "Q_sat": 1400.0, "x_pct": 1.10,
     "source": "doi 10.1038/srep01985", "kind": "increment, one device"},
    {"label": "Nan 2013, 215 MHz (lower reading)", "f_r_Hz": 215e6,
     "Q_wp": 250.0, "Q_sat": 1400.0, "x_pct": 0.74,
     "source": "doi 10.1038/srep01985", "kind": "increment, one device"},
    {"label": "Durdaut 2020, 7.4 kHz", "f_r_Hz": 7422.0,
     "Q_wp": 825.0, "Q_sat": 1071.0, "x_pct": 0.500,
     "source": "arXiv:2003.01085, digitised here", "kind": "increment, one device"},
]


def lambda_external():
    out = []
    for e in EXTERNAL:
        lam = (1.0 / e["Q_wp"] - 1.0 / e["Q_sat"]) / e["x_pct"]
        d = dict(e); d["Lambda"] = float(lam)
        out.append(d)
    return out


def fit_scaling(points):
    f = np.array([p["f_r_Hz"] for p in points])
    L = np.array([p["Lambda"] for p in points])
    n, c = np.polyfit(np.log(f), np.log(L), 1)
    res = np.log(L) - (n * np.log(f) + c)
    se = np.sqrt(np.sum(res ** 2) / max(len(points) - 2, 1)
                 / np.sum((np.log(f) - np.log(f).mean()) ** 2))
    return float(n), float(se), float(np.exp(c)), float(L.max() / L.min()), \
        float(f.max() / f.min())


def main():
    prof = profile_exponent()
    fam = lambda_per_family()
    ext = lambda_external()

    # reference set: one reading of Nan, both families, the low-frequency point
    ref = fam + [e for e in ext if "upper" in e["label"] or "Durdaut" in e["label"]]
    n, se, lam0, span_L, span_f = fit_scaling(ref)
    alt = fam + [e for e in ext if "lower" in e["label"] or "Durdaut" in e["label"]]
    n_alt, se_alt, _, _, _ = fit_scaling(alt)
    no_low = [p for p in ref if p["f_r_Hz"] > 1e6]
    n_no_low, _, _, _, span_f_no_low = fit_scaling(no_low)

    out = {
        "exponent_profile_within_ref1": prof,
        "lambda_families": fam, "lambda_external": ext,
        "scaling": {"n": n, "stderr": se, "Lambda_at_1Hz": lam0,
                    "Lambda_span": span_L, "frequency_span": span_f,
                    "n_alternative_reading": n_alt,
                    "n_without_low_frequency_point": n_no_low,
                    "frequency_span_without_low_point": span_f_no_low},
        "scatter_floor": {
            "Lambda_ratio_between_our_two_families":
                fam[0]["Lambda"] / fam[1]["Lambda"],
            "frequency_ratio_between_them": fam[0]["f_r_Hz"] / fam[1]["f_r_Hz"]},
    }
    (HERE / "loss_scaling.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ff = np.geomspace(3e3, 5e8, 50)
    ax.loglog(ff, lam0 * ff ** n, "-", color="C0", lw=2,
              label=r"fit, $n=%.2f\pm%.2f$" % (n, se))
    ref_lam = ref[0]["Lambda"]; ref_f = ref[0]["f_r_Hz"]
    ax.loglog(ff, ref_lam * (ff / ref_f) ** 1.0, "--", color="C3", lw=1.6,
              label=r"relaxation, $n=1$ (assumed initially)")
    ax.loglog(ff, ref_lam * (ff / ref_f) ** -1.0, ":", color="C2", lw=1.6,
              label=r"Eq. (4) of [1], $n=-1$")
    for p, mk, col in zip(fam, ("o", "o"), ("C0", "C0")):
        ax.loglog(p["f_r_Hz"], p["Lambda"], mk, ms=8, mfc=col, mec="k", mew=0.8)
    for p in ext:
        mk = "s" if "Nan" in p["label"] else "^"
        ax.loglog(p["f_r_Hz"], p["Lambda"], mk, ms=9, mfc="none", mec="k", mew=1.4)
    ax.loglog([], [], "o", ms=8, mfc="C0", mec="k", label="this work, from [1]")
    ax.loglog([], [], "s", ms=9, mfc="none", mec="k", label="Nan 2013, 215 MHz")
    ax.loglog([], [], "^", ms=9, mfc="none", mec="k", label="Durdaut 2020, 7.4 kHz")
    ax.set_xlabel(r"resonance frequency $f_r$ (Hz)")
    ax.set_ylabel(r"$\Lambda=[1/Q-1/Q_{\rm sat}]/x$  (per %)")
    ax.set_ylim(1e-4, 1e-1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_loss_scaling.png", dpi=160)
    plt.close(fig)

    print("1. FREQUENCY EXPONENT INSIDE [1] ALONE")
    for r in prof["rows"]:
        mark = {1.0: "  <- assumed initially", -1.0: "  <- Eq. (4) of [1]",
                0.0: "  <- rate independent"}.get(r["n"], "")
        print("   n = %+4.1f  Q0 = %4.0f  Lambda = %.2e  R2 = %+.3f  rms(Q) = %5.1f%s"
              % (r["n"], r["Q0"], r["A"], r["R2"], r["rms_Q"], mark))
    print("   within 5%% of the best: n from %.1f to %.1f -> not identifiable\n"
          % (min(prof["n_within_5pct_of_best"]), max(prof["n_within_5pct_of_best"])))

    print("2. LOSS PER UNIT SOFTENING ACROSS PLATFORMS")
    for p in fam + ext:
        print("   %-36s f_r = %10.4f MHz   Lambda = %.3e   (%s)"
              % (p["label"], p["f_r_Hz"] / 1e6, p["Lambda"], p["kind"]))
    s = out["scaling"]
    print("\n   fit over %.1f decades: n = %.2f +/- %.2f"
          % (np.log10(s["frequency_span"]), s["n"], s["stderr"]))
    print("   other reading of Nan: n = %.2f" % s["n_alternative_reading"])
    print("   Lambda spans x%.1f while the frequency spans x%.0f"
          % (s["Lambda_span"], s["frequency_span"]))
    print("   n = 1 would require x%.0f" % s["frequency_span"])
    print("\n   WITHOUT the low-frequency point: n = %.2f, but over a frequency span"
          " of only x%.1f," % (s["n_without_low_frequency_point"],
                               s["frequency_span_without_low_point"]))
    c = out["scatter_floor"]
    print("   against a scatter of x%.1f between our own two families, whose"
          " frequencies\n   differ by only x%.1f. The high-frequency cluster alone"
          " determines nothing."
          % (c["Lambda_ratio_between_our_two_families"], c["frequency_ratio_between_them"]))
    print("\nOutputs: loss_scaling.json, fig_loss_scaling.png")


if __name__ == "__main__":
    main()
