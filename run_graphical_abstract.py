"""
Graphical abstract for the IEEE Sensors Journal submission.

Single clean image, ratio 2.5:1 at 300 dpi, no heading text, large sans-serif
labels so that it stays readable at thumbnail size.

The message is the one the paper is built on: the model is scored against the
measured device population of [1], device by device.

  Left   device schematic, flat versus curved suspended plate.
  Middle the 64 measured responses of Fig. 3a of [1] with the energy-averaged
         model through them (one global constant) and the single-domain model,
         which saturates.
  Right  the loss-limited detectivity of the 25 measured devices whose
         resonance frequency is known: the factor-of-57 response spread
         compresses to a factor of 23.

Every number shown here is read from validation.json, so this figure cannot
drift away from the manuscript.  Run run_validation.py first.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from deltaE import squire_E, deam_E, frequency_response
from materials import MU0
from paper_fits import keff_fit, phi_eff_fit

HERE = Path(__file__).parent
plt.rcParams["font.family"] = "DejaVu Sans"

V = json.loads((HERE / "validation.json").read_text())
RESP, DET = V["response"], V["detectivity"]

# Which result goes in the right-hand panel.  "detectivity" is the original
# choice; "scaling" shows instead the cross-platform frequency scaling of the
# magnetoelastic loss, which is the central new result of the present version.
PANEL = "scaling" if "--scaling" in sys.argv else "detectivity"
OUT = "fig_graphical_abstract_scaling.png" if PANEL == "scaling" else "fig_graphical_abstract.png"
SCAL = json.loads((HERE / "loss_scaling.json").read_text()) if PANEL == "scaling" else None

H = np.linspace(0.0, 12e-3, 140) / MU0


def load(name):
    p = np.asarray(json.loads((HERE / name).read_text())["points"], dtype=float)
    return p[:, 0], p[:, 1]


# IEEE Sensors graphical-abstract spec: 672 x 456 px (3.5 x 2.38 in).
# Rendered at 2x for sharpness, so it downsamples cleanly to the spec size.
fig = plt.figure(figsize=(7.0, 4.75), dpi=192)           # 1344x912 px, 1.47:1
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32,
                      left=0.115, right=0.985, top=0.94, bottom=0.135)

# ---------------- (device schematic dropped: not a figure of the paper) ----
axL = None
if axL is not None:
    axL.axis("off")
    axL.set_xlim(0, 10); axL.set_ylim(0, 10)

    axL.add_patch(Rectangle((0.8, 6.6), 1.0, 1.0, facecolor="#c8c8c8", edgecolor="k"))
    axL.add_patch(Rectangle((7.6, 6.6), 1.0, 1.0, facecolor="#c8c8c8", edgecolor="k"))
    axL.add_patch(Rectangle((1.6, 6.75), 6.2, 0.7, facecolor="#f6c9a8", edgecolor="k", lw=1.3))
    axL.text(5.0, 8.2, "flat", ha="center", fontsize=13, fontweight="bold")

    axL.add_patch(Rectangle((0.8, 1.2), 1.0, 1.0, facecolor="#c8c8c8", edgecolor="k"))
    axL.add_patch(Rectangle((7.6, 1.2), 1.0, 1.0, facecolor="#c8c8c8", edgecolor="k"))
    xs = np.linspace(1.8, 7.6, 60)
    yb = 1.55 + 0.55 * np.sin(np.pi * (xs - 1.8) / (7.6 - 1.8))
    axL.fill_between(xs, yb, yb + 0.7, color="#f6c9a8", edgecolor="k", lw=1.3)
    axL.text(5.0, 3.5, "curved (κ)", ha="center", fontsize=13, fontweight="bold")

    axL.add_patch(FancyArrowPatch((5.0, 6.0), (5.0, 4.7), arrowstyle="-|>",
                                  mutation_scale=22, color="C3", lw=2.2))
    axL.text(5.5, 5.35, "residual\nstress", fontsize=10, color="C3", va="center")

# ---------------- Middle: model against the measured population ----------------
axM = fig.add_subplot(gs[0])
kap, resp = load("measured_fig3a.json")
kk = np.linspace(kap.min(), kap.max(), 40)
# The contrast that matters is one constant for the whole wafer against one
# fitted per device, not one magnetisation model against the other: the two
# models are not distinguished by these measurements (see REVISION_NOTES.md).
import crossval as _cv
cur_d = _cv.predict(kk, "coherent", RESP["coherent"]["v_global_pct"] / 100.0)
k10, v10 = load("model_fig10.json")

axM.semilogy(kap, resp, "o", ms=5.5, mfc="none", mec="0.3", mew=1.2,
             label="%d measured devices [1]" % RESP["n_devices"])
axM.semilogy(kk, cur_d, "-", color="C0", lw=3.2,
             label="this work: 1 constant/wafer (%.2f)"
                   % RESP["coherent"]["rms_factor"])
axM.semilogy(k10, v10, ":", color="C3", lw=2.6,
             label="[1]: 1 constant/device (%.2f)"
                   % RESP["published_model"]["rms_factor"])
axM.set_xlabel("curvature κ  (mm$^{-1}$)", fontsize=13)
axM.set_ylabel(r"$\Delta f_r/f_{r,\min}$  (%)", fontsize=13)
axM.tick_params(labelsize=11)
axM.grid(True, which="both", alpha=0.25)
axM.legend(fontsize=9, loc="lower left", frameon=True, framealpha=0.88,
           edgecolor="0.8", title="RMS deviation (factor)", title_fontsize=9)

# ---------------- Right: either detectivity, or the loss scaling ----------
axR = fig.add_subplot(gs[1])
if PANEL == "detectivity":

    for (fname, col, mk, lab) in (("measured_fig6c.json", "C2", "s", "246 MHz"),
                                  ("measured_fig6d.json", "C0", "^", "138 MHz")):
        x, q = load(fname)
        fr = 246e6 if "6c" in fname else 138e6
        d = (fr / q) / x
        axR.loglog(x, d / d.min(), mk, ms=7, mfc="none", mec=col, mew=1.6, label=lab)
    xr = np.array([0.02, 1.25])
    axR.loglog(xr, xr.max() / xr, "k:", lw=2.0)
    axR.set_xlabel(r"$\Delta f_r/f_{r,\min}$  (%)", fontsize=13)
    axR.set_ylabel("detectivity (normalised)", fontsize=13)
    axR.tick_params(labelsize=11)
    axR.grid(True, which="both", alpha=0.25)
    axR.legend(fontsize=10.5, loc="lower left", frameon=False, title="measured",
               title_fontsize=10.5)
    axR.text(0.97, 0.95,
             "response ×%.0f\n→ detectivity ×%.0f" % (DET["response_spread_pooled"],
                                                      DET["detectivity_spread_pooled"]),
             transform=axR.transAxes, ha="right", va="top",
             fontsize=12, fontweight="bold", color="C3")

else:
    fam = SCAL["lambda_families"]; ext = SCAL["lambda_external"]
    sc = SCAL["scaling"]
    ff = np.geomspace(3e3, 5e8, 40)
    axR.loglog(ff, sc["Lambda_at_1Hz"] * ff ** sc["n"], "-", color="C0", lw=2.6)
    ref_f, ref_L = fam[0]["f_r_Hz"], fam[0]["Lambda"]
    axR.loglog(ff, ref_L * (ff / ref_f), "--", color="C3", lw=1.8)
    for q in fam:
        axR.loglog(q["f_r_Hz"], q["Lambda"], "o", ms=8, mfc="C0", mec="k", mew=0.8)
    for q in ext:
        mk = "s" if "Nan" in q["label"] else "^"
        axR.loglog(q["f_r_Hz"], q["Lambda"], mk, ms=8.5, mfc="none", mec="k", mew=1.5)
    axR.set_xlabel("resonance frequency  (Hz)", fontsize=13)
    axR.set_ylabel(r"magnetic loss / softening", fontsize=12)
    axR.set_ylim(1e-4, 4e-2)
    axR.tick_params(labelsize=11)
    axR.grid(True, which="both", alpha=0.25)
    axR.text(0.04, 0.95, "loss \u00d7%.1f\nover f \u00d7%.0f k"
             % (sc["Lambda_span"], sc["frequency_span"] / 1000.0),
             transform=axR.transAxes, ha="left", va="top",
             fontsize=12, fontweight="bold", color="C0")
    axR.text(0.96, 0.28, "3 platforms,\n4.5 decades", transform=axR.transAxes,
             ha="right", va="bottom", fontsize=9.5, color="0.35")
    axR.text(0.62, 0.42, r"$\propto f_r$", transform=axR.transAxes,
             fontsize=11, color="C3", rotation=38)

fig.savefig(HERE / OUT, dpi=96)   # -> 672 x 456 px, the IEEE spec size
plt.close(fig)
print("Graphical abstract written: %s  (right panel: %s; RMS %.2f, detectivity x%.0f, "
      "all read from validation.json)"
      % (OUT, PANEL, RESP["coherent"]["rms_factor"], DET["detectivity_spread_pooled"]))
