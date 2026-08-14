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
v_d = RESP["deam"]["v_global_pct"] / 100.0
v_s = RESP["squire"]["v_global_pct"] / 100.0
cur_d = np.array([frequency_response(H, deam_E(H, keff_fit(k) * 1e3, phi_eff_fit(k)),
                                     v=v_d)[1] for k in kk])
cur_s = np.array([frequency_response(H, squire_E(H, keff_fit(k) * 1e3, phi_eff_fit(k)),
                                     v=v_s)[1] for k in kk])

axM.semilogy(kap, resp, "o", ms=5.5, mfc="none", mec="0.3", mew=1.2,
             label="%d measured devices" % RESP["n_devices"])
axM.semilogy(kk, cur_d, "-", color="C2", lw=3.2,
             label="this work (RMS %.2f)" % RESP["deam"]["rms_factor"])
axM.semilogy(kk, cur_s, "--", color="C1", lw=3.0,
             label="single domain (RMS %.2f)" % RESP["squire"]["rms_factor"])
axM.set_xlabel("curvature κ  (mm$^{-1}$)", fontsize=13)
axM.set_ylabel(r"$\Delta f_r/f_{r,\min}$  (%)", fontsize=13)
axM.tick_params(labelsize=11)
axM.grid(True, which="both", alpha=0.25)
axM.legend(fontsize=10.5, loc="lower left", frameon=False)

# ---------------- Right: measured detectivity compression ----------------
axR = fig.add_subplot(gs[1])
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

fig.savefig(HERE / "fig_graphical_abstract.png", dpi=96)   # -> 672 x 456 px, the IEEE spec size
plt.close(fig)
print("Graphical abstract written: fig_graphical_abstract.png "
      "(RMS %.2f, detectivity x%.0f from validation.json)"
      % (RESP["deam"]["rms_factor"], DET["detectivity_spread_pooled"]))
