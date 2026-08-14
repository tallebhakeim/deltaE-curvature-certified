"""Figure: anchor-induced retention of residual in-plane anisotropy (2D FE)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fe2d import relax_plate
from materials import LAMBDA_S

r = relax_plate(Lx=200e-6, Ly=70e-6, nx=48, ny=18,
                sigma0=(45e6, 75e6, 5e6), E=150e9, nu=0.3, anchor_frac=0.12)
c = r["cent"] * 1e6                      # um
d = r["dsig"] / 1e6                      # MPa retained sxx - syy

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [2.3, 1]})

# left: spatial map of retained anisotropy
sc = a1.tripcolor(c[:, 0], c[:, 1], d, shading="gouraud", cmap="RdBu_r",
                  vmin=-abs(d).max(), vmax=abs(d).max())
xa = r["xa"] * 1e6; Lx = r["Lx"] * 1e6; Ly = r["Ly"] * 1e6
for x0 in (0, Lx - xa):
    a1.add_patch(plt.Rectangle((x0, 0), xa, Ly, fill=False, ec="k", lw=1.2, ls="--"))
a1.set_xlabel("length x (um)"); a1.set_ylabel("width y (um)")
a1.set_title("Retained $s_{xx}-s_{yy}$ (MPa): blue core = easy axis along y")
fig.colorbar(sc, ax=a1, shrink=0.85)

# right: retention fraction vs free-plate CLT, and resulting Ksigma
labels = ["free plate\n(CLT, ~0.13)", "anchored\n(2D FE, core)"]
fret = [0.13, r["f_ret_core"]]
Ksig = [1.5 * LAMBDA_S * abs(f * r["ds0"]) / 1e3 for f in fret]   # kJ/m^3
b = a2.bar(labels, fret, color=["C7", "C0"])
a2.set_ylabel("retained anisotropy fraction")
a2.set_title("Anchors retain ~%.0f%% of $\\sigma_0$ anisotropy" % (r["f_ret_core"]*100))
for rect, k in zip(b, Ksig):
    a2.text(rect.get_x()+rect.get_width()/2, rect.get_height(),
            f"$K_\\sigma\\approx${k:.1f} kJ/m$^3$", ha="center", va="bottom", fontsize=8)
a2.set_ylim(0, max(fret)*1.4)

fig.tight_layout()
fig.savefig("fig_fe2d.png", dpi=140)
plt.close(fig)
print("f_ret core=%.3f  Ksigma_core~%.2f kJ/m3 (paper range 3.5-18)" %
      (r["f_ret_core"], Ksig[1]))
print("Figure: fig_fe2d.png")
