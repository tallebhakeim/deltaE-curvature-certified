"""Field cartographies:
   fig_fields_z.png  -- through-thickness stress and magnetoelastic anisotropy
                        (reproduction of Fig. 8b / 9c of [1])
   fig_fields_xy.png -- 2D xy maps of stress-induced easy axis and anisotropy
                        (reproduction of Fig. 9a,b of [1]), from the 2D FE.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materials import default_stack, LAMBDA_S, KU
from mechanics import curvature_sweep
from magnetoelastic import anisotropy_profile, _energy_me, _ALPHA
from fe2d import relax_plate

# ============================================================
# 1) Through-thickness field profiles (z)
# ============================================================
layers = default_stack(500e-9)
# pick a curved device: scale so kappa_y ~ 2.5 mm^-1
sweep = curvature_sweep(layers, {"FeGaB": (45e6, 75e6, 5e6)}, [5.0])
k_y, st, sc = sweep[0]
p = anisotropy_profile(st)
z_nm = (st.z - st.z.min()) * 1e9        # nm from bottom of FeGaB
zN = None if st.z_neutral is None else (st.z_neutral - st.z.min()) * 1e9

fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 7.4))
a1.plot(z_nm, st.sxx / 1e6, label=r"$\sigma_{xx}$", color="C0")
a1.plot(z_nm, st.syy / 1e6, label=r"$\sigma_{yy}$", color="C3")
a1.plot(z_nm, st.sxy / 1e6, label=r"$\sigma_{xy}$", color="C2")
a1.plot(z_nm, (st.sxx - st.syy) / 1e6, "--", color="k", lw=1,
        label=r"$\sigma_{xx}-\sigma_{yy}$")
if zN is not None:
    a1.axvline(zN, color="gray", ls=":", lw=1.2)
    a1.text(zN, a1.get_ylim()[1], " neutral axis", color="gray", fontsize=8, va="top")
a1.axhline(0, color="0.7", lw=0.6)
a1.set_xlabel("z through FeGaB thickness (nm)")
a1.set_ylabel("stress (MPa)")
a1.set_title(r"(a) Through-thickness stress ($\kappa\approx$%.1f mm$^{-1}$)" % (k_y/1e3))
a1.legend(fontsize=8); a1.grid(alpha=0.3)

a2b = a2.twinx()
a2.plot(z_nm, p.Ksigma / 1e3, color="C0", label=r"$K_\sigma$")
a2b.plot(z_nm, p.phi_sigma, color="C3", ls="--", label=r"$\varphi_\sigma$")
a2.set_xlabel("z through FeGaB thickness (nm)")
a2.set_ylabel(r"$K_\sigma$ (kJ m$^{-3}$)", color="C0")
a2b.set_ylabel(r"$\varphi_\sigma$ (deg from $x$)", color="C3")
a2.tick_params(axis="y", labelcolor="C0"); a2b.tick_params(axis="y", labelcolor="C3")
if zN is not None:
    a2.axvline(zN, color="gray", ls=":", lw=1.2)
a2.set_title("(b) Stress-induced anisotropy vs depth")
fig.suptitle("Through-thickness field profiles: the easy axis flips across the "
             "neutral axis (cf. Fig. 8b/9c of [1])", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_fields_z.png", dpi=140)
plt.close(fig)

# ============================================================
# 2) 2D xy cartographies of phi_sigma and K_sigma (from 2D FE)
# ============================================================
r = relax_plate(Lx=200e-6, Ly=70e-6, nx=48, ny=18,
                sigma0=(45e6, 75e6, 5e6), E=150e9, nu=0.3, anchor_frac=0.12)
cent = r["cent"] * 1e6                    # um
sret = r["sret"]                          # per-element (sxx,syy,sxy)

phis = np.empty(len(sret)); Ksig = np.empty(len(sret))
for i, (sx, sy, sxy) in enumerate(sret):
    e = _energy_me(_ALPHA, sx, sy, sxy) + KU * np.cos(_ALPHA) ** 2
    Ksig[i] = (e.max() - e.min()) / 1e3   # kJ/m^3
    phis[i] = np.rad2deg(_ALPHA[np.argmin(e)])

fig, (b1, b2) = plt.subplots(2, 1, figsize=(5.4, 6.8))
sc1 = b1.tripcolor(cent[:, 0], cent[:, 1], phis, shading="gouraud",
                   cmap="twilight", vmin=0, vmax=180)
b1.set_title(r"(a) Easy-axis angle $\varphi_\sigma$ (deg)")
b1.set_xlabel("x (um)"); b1.set_ylabel("y (um)")
fig.colorbar(sc1, ax=b1, shrink=0.85)
sc2 = b2.tripcolor(cent[:, 0], cent[:, 1], Ksig, shading="gouraud", cmap="viridis")
b2.set_title(r"(b) Anisotropy density $K_\sigma$ (kJ m$^{-3}$)")
b2.set_xlabel("x (um)"); b2.set_ylabel("y (um)")
fig.colorbar(sc2, ax=b2, shrink=0.85)
for ax in (b1, b2):
    xa = r["xa"] * 1e6; Lx = r["Lx"] * 1e6; Ly = r["Ly"] * 1e6
    for x0 in (0, Lx - xa):
        ax.add_patch(plt.Rectangle((x0, 0), xa, Ly, fill=False, ec="w", lw=1.0, ls="--"))
fig.suptitle("In-plane cartographies from the 2D FE: easy axis and anisotropy vary "
             "from centre to edges/anchors (cf. Fig. 9a,b of [1])", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_fields_xy.png", dpi=140)
plt.close(fig)
print("Figures: fig_fields_z.png, fig_fields_xy.png")
