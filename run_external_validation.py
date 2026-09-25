"""
Test the loss law on an EXTERNAL device family, with nothing refitted.

The law calibrated in this work on the population of [1],

    1/Q = 1/Q0 + 2 pi f_r tau x ,   x = Delta f_r/f_r,min in percent,
    Q0 = 523 +/- 64 ,   tau = 4.07 +/- 1.41 ps per percent,

contains the resonance frequency explicitly, so it can be applied to devices it
was never fitted to.  Within [1] that transfer works between the 246 MHz and the
138 MHz families (coefficients of determination 0.37 and 0.70, against -0.26 and
-1.97 for the four-constant form of its Eq. 4).  The honest next step is a family
from another laboratory or another fabrication run, ideally at a third resonance
frequency.

This script is the harness for that step.  Drop a JSON file in external_data/
with the shape below and run it; nothing else has to be touched.

    {
      "source": "Author et al., Journal vol, page (year)",
      "doi": "10.xxxx/yyyy",
      "note": "which figure or table the values come from, and how",
      "mode": "contour | bending | torsion | SAW | ...",
      "stack": "FeGaB/AlN/Pt, 500 nm magnetic",
      "devices": [
        {"f_r_Hz": 1.0e8, "Q": 400, "response_pct": 0.25},
        ...
      ]
    }

Three questions are answered for each file:

  1. Does the law predict this family with NO free parameter?  Coefficient of
     determination and median deviation, using Q0 = 523 and tau = 4.07 ps.
  2. How much of the answer is the explicit f_r scaling?  The same prediction is
     repeated with the frequency of our own families, which is what a law
     without explicit f_r would effectively assume.
  3. What would this family alone say?  (Q0, tau) refitted on it, to be compared
     with our values and their intervals.  If the refitted tau lands inside
     2.0 to 8.0 ps, the relaxation time is consistent across laboratories, which
     is the strongest statement this work can reach without new fabrication.

Run:  python3 run_external_validation.py [file.json ...]
      with no argument it runs the two families of [1] as a self-test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import crossval as cv

HERE = Path(__file__).parent
EXT = HERE / "external_data"

Q0_REF = 523.0          # this work, fitted on [1]
TAU_REF = 4.07e-12      # s per percent of response
TAU_INTERVAL_PS = (2.0, 8.0)    # 95% profile interval, run_identifiability.py
OUR_FREQS = (246e6, 138e6)


def predict(x, f_r, q0=Q0_REF, tau=TAU_REF):
    return 1.0 / (1.0 / q0 + 2 * np.pi * f_r * tau * x)


def r2(q, pred):
    q = np.asarray(q, float)
    return float(1.0 - np.sum((q - pred) ** 2) / np.sum((q - q.mean()) ** 2))


def fit(x, q, f_r, start=(1000.0 / 523.0, 4.07)):
    """(Q0, tau) on one family, in the scaled parameterisation that converges."""
    def resid(u):
        return q - predict(x, f_r, 1000.0 / u[0], u[1] * 1e-12)
    sol = least_squares(resid, list(start))
    return 1000.0 / sol.x[0], float(sol.x[1])


def evaluate(name, x, q, f_r, meta=None):
    x = np.asarray(x, float); q = np.asarray(q, float)
    pred = predict(x, f_r, )
    out = {
        "name": name, "n": int(x.size), "f_r_MHz": f_r / 1e6,
        "meta": meta or {},
        "blind": {"R2": r2(q, pred),
                  "median_abs_dev_pct": 100 * float(np.median(np.abs(pred / q - 1)))},
        "wrong_frequency_control": {
            "%.0f MHz" % (f / 1e6): r2(q, predict(x, f)) for f in OUR_FREQS},
    }
    if x.size >= 3:
        q0_loc, tau_loc = fit(x, q, f_r)
        out["refitted_on_this_family"] = {
            "Q0": q0_loc, "tau_ps": tau_loc,
            "tau_inside_our_interval": bool(TAU_INTERVAL_PS[0] <= tau_loc <= TAU_INTERVAL_PS[1]),
            "R2": r2(q, predict(x, f_r, q0_loc, tau_loc * 1e-12))}
    else:
        out["refitted_on_this_family"] = None
    return out


# --------------------------------------------------------------------------
#  Increment test: the magnetic part of the loss alone
# --------------------------------------------------------------------------
def increment_test(f_r, q_wp, q_sat, x, tau=TAU_REF):
    """Compare the MEASURED magnetic loss increment with the predicted one.

    Between the magnetic working point and magnetic saturation the non-magnetic
    loss is unchanged, so

        1/Q(x) - 1/Q_sat = 2 pi f_r tau x

    and Q0 cancels.  This is the sharpest form of the test, because it isolates
    the magnetic mechanism and needs no shared non-magnetic constant: a device
    from another laboratory can be used without knowing anything about its
    anchor or thermoelastic losses.
    """
    measured = 1.0 / q_wp - 1.0 / q_sat
    predicted = 2 * np.pi * f_r * tau * x
    tau_implied = measured / (2 * np.pi * f_r * x)
    return {"f_r_MHz": f_r / 1e6, "response_pct": x,
            "measured_increment": measured, "predicted_increment": predicted,
            "ratio_predicted_over_measured": predicted / measured,
            "tau_implied_ps": tau_implied * 1e12,
            "tau_inside_our_interval": bool(TAU_INTERVAL_PS[0] <= tau_implied * 1e12
                                            <= TAU_INTERVAL_PS[1])}


def load_increments():
    rows = []
    if not EXT.is_dir():
        return rows
    for f in sorted(EXT.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("kind") != "increment":
            continue
        for e in d["increments"]:
            r = increment_test(float(e["f_r_Hz"]), float(e["Q_working_point"]),
                               float(e["Q_saturation"]), float(e["response_pct"]))
            r["source"] = d.get("source", f.stem)
            r["doi"] = d.get("doi")
            r["independent"] = d.get("independent_of_matyushov_2021")
            r["reading"] = e.get("reading", "")
            rows.append(r)
    return rows


def report_increments(rows):
    if not rows:
        return
    print("MAGNETIC LOSS INCREMENT, 1/Q(x) - 1/Q_sat, where Q0 cancels")
    print("our value: tau = 4.07 ps per percent, 95%% interval %.1f to %.1f\n"
          % TAU_INTERVAL_PS)
    for r in rows:
        print("%s" % r["source"][:96])
        print("   f_r = %.3f MHz, x = %.2f%%  (%s)%s"
              % (r["f_r_MHz"], r["response_pct"], r["reading"],
                 "" if r["independent"] else "   [authors overlap with [1]]"))
        print("   measured %.2e, predicted %.2e, ratio %.0f"
              % (r["measured_increment"], r["predicted_increment"],
                 r["ratio_predicted_over_measured"]))
        print("   implied tau = %.2f ps per percent -> %s our interval\n"
              % (r["tau_implied_ps"],
                 "INSIDE" if r["tau_inside_our_interval"] else "OUTSIDE"))


def self_test():
    """The two families of [1], to show the harness reproduces known numbers."""
    res = []
    for fname, f_r, tag in (("measured_fig6c.json", 246e6, "[1], designs 3, 4"),
                            ("measured_fig6d.json", 138e6, "[1], designs 7, 8")):
        x, q = cv.load_points(fname)
        res.append(evaluate(tag, x, q, f_r, {"self_test": True}))
    return res


def load_external():
    if not EXT.is_dir():
        return []
    res = []
    for f in sorted(EXT.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("kind") == "increment" or "devices" not in d:
            continue
        dev = d["devices"]
        by_freq = {}
        for entry in dev:
            by_freq.setdefault(float(entry["f_r_Hz"]), []).append(entry)
        for f_r, group in sorted(by_freq.items()):
            x = [g["response_pct"] for g in group]
            q = [g["Q"] for g in group]
            res.append(evaluate("%s [%s]" % (d.get("source", f.stem), f.name),
                                x, q, f_r,
                                {k: d.get(k) for k in ("doi", "note", "mode", "stack")}))
    return res


def report(rows):
    for r in rows:
        print("%s  (%d devices at %.0f MHz)" % (r["name"], r["n"], r["f_r_MHz"]))
        if r["meta"].get("doi"):
            print("   doi %s" % r["meta"]["doi"])
        b = r["blind"]
        print("   predicted with Q0 = 523 and tau = 4.07 ps, nothing refitted:"
              "  R2 = %+.2f, median deviation %.0f%%"
              % (b["R2"], b["median_abs_dev_pct"]))
        ctrl = ", ".join("%s: %+.2f" % (k, v) for k, v in r["wrong_frequency_control"].items())
        print("   same law forced to our own frequencies (control): %s" % ctrl)
        g = r["refitted_on_this_family"]
        if g:
            print("   refitted on this family alone: Q0 = %.0f, tau = %.2f ps"
                  " (R2 = %+.2f); tau %s our 95%% interval %.1f to %.1f ps"
                  % (g["Q0"], g["tau_ps"], g["R2"],
                     "INSIDE" if g["tau_inside_our_interval"] else "OUTSIDE",
                     *TAU_INTERVAL_PS))
        print()


if __name__ == "__main__":
    rows = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            d = json.loads(Path(path).read_text())
            by_freq = {}
            for entry in d["devices"]:
                by_freq.setdefault(float(entry["f_r_Hz"]), []).append(entry)
            for f_r, group in sorted(by_freq.items()):
                rows.append(evaluate(d.get("source", path),
                                     [g["response_pct"] for g in group],
                                     [g["Q"] for g in group], f_r,
                                     {k: d.get(k) for k in ("doi", "note", "mode", "stack")}))
    else:
        print("SELF-TEST on the two families of [1] (no external file given)\n")
        rows = self_test() + load_external()
    report(rows)
    inc = load_increments()
    report_increments(inc)
    (HERE / "external_validation.json").write_text(
        json.dumps({"families": rows, "increments": inc}, indent=1))
    print("Written: external_validation.json")
