"""Schematic illustration of the contour-mode Delta-E magnetometer."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyArrowPatch, Arc

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2),
                               gridspec_kw={"width_ratios": [1, 1.25]})

# ---------- Panel A: layer stack cross-section ----------
axA.set_title("(a) Layer stack (cross-section)", fontsize=11)
layers = [("Pt IDE (50 nm)", 0.6, "#b0b0b0"),
          ("AlN piezoelectric (250 nm)", 2.6, "#cfe3f7"),
          ("FeGaB magnetostrictive (250-1000 nm)", 2.2, "#f6c9a8")]
y = 0.0
w = 6.0
for name, h, col in layers:
    axA.add_patch(Rectangle((0.5, y), w, h, facecolor=col, edgecolor="k", lw=1.2))
    axA.text(0.5 + w / 2, y + h / 2, name, ha="center", va="center", fontsize=9)
    y += h
# IDE fingers (teeth) inside Pt
for xf in [1.2, 2.2, 3.2, 4.2, 5.2]:
    axA.add_patch(Rectangle((xf, 0.0), 0.35, 0.6, facecolor="#777", edgecolor="k", lw=0.6))
# AC drive
axA.annotate("AC drive $V\\sim$", xy=(0.5, 0.3), xytext=(-1.6, 0.3),
             fontsize=9, va="center",
             arrowprops=dict(arrowstyle="->", lw=1.4, color="C3"))
# suspended (air) below
axA.text(3.5, -0.6, "released: suspended in air", ha="center", fontsize=8, style="italic")
axA.annotate("", xy=(6.6, 0), xytext=(6.6, y),
             arrowprops=dict(arrowstyle="->", lw=1.2))
axA.text(6.9, y / 2, "z", fontsize=11, va="center")
axA.set_xlim(-2.2, 7.6); axA.set_ylim(-1.1, y + 0.6); axA.axis("off")

# ---------- Panel B: top view of suspended plate ----------
axB.set_title("(b) Suspended resonator plate (top view)", fontsize=11)
L, W = 8.0, 3.2
x0, y0 = 1.0, 1.0
axB.add_patch(Rectangle((x0, y0), L, W, facecolor="#f6c9a8", edgecolor="k", lw=1.4))
# anchors
axB.add_patch(Rectangle((x0 - 1.0, y0 + W/2 - 0.5), 1.0, 1.0, facecolor="#c8c8c8", edgecolor="k"))
axB.add_patch(Rectangle((x0 + L, y0 + W/2 - 0.5), 1.0, 1.0, facecolor="#c8c8c8", edgecolor="k"))
axB.text(x0 - 0.5, y0 + W/2 + 0.9, "anchor", ha="center", fontsize=8)
axB.text(x0 + L + 0.5, y0 + W/2 + 0.9, "anchor", ha="center", fontsize=8)
# IDE fingers
for xf in range(1, 8):
    axB.plot([x0 + xf, x0 + xf], [y0 + 0.15, y0 + W - 0.15], color="#777", lw=1.0)
# easy axis (deposition field), along width y
axB.annotate("", xy=(x0 + L/2, y0 + W + 0.7), xytext=(x0 + L/2, y0 - 0.9),
             arrowprops=dict(arrowstyle="<->", lw=2.0, color="C0"))
axB.text(x0 + L/2 + 0.25, y0 + W + 1.35,
         "easy axis $K_u$ (deposition field, $y$)",
         color="C0", fontsize=8.5, va="center", ha="left")
# DC test field H along length x
axB.annotate("", xy=(x0 + L + 0.2, y0 - 0.6), xytext=(x0 - 0.2, y0 - 0.6),
             arrowprops=dict(arrowstyle="->", lw=2.0, color="C3"))
axB.text(x0 + L/2, y0 - 1.15, "DC test field $H$ (length, $x$)",
         color="C3", ha="center", fontsize=9)
# curvature note
axB.text(x0 + L/2, y0 + W/2, "residual stress\n$\\rightarrow$ curvature $\\kappa$",
         ha="center", va="center", fontsize=9, style="italic")
# axes
axB.annotate("", xy=(x0 + 1.6, y0 + W + 1.6), xytext=(x0, y0 + W + 1.6),
             arrowprops=dict(arrowstyle="->", lw=1.0))
axB.annotate("", xy=(x0, y0 + W + 2.6), xytext=(x0, y0 + W + 1.6),
             arrowprops=dict(arrowstyle="->", lw=1.0))
axB.text(x0 + 1.7, y0 + W + 1.6, "x", fontsize=10, va="center")
axB.text(x0 - 0.05, y0 + W + 2.7, "y", fontsize=10, ha="center")
axB.set_xlim(x0 - 1.6, x0 + L + 1.6); axB.set_ylim(y0 - 1.6, y0 + W + 3.0)
axB.set_aspect("equal"); axB.axis("off")

fig.suptitle("Contour-mode Delta-E effect magnetometer: the AC drive excites a "
             "contour-mode resonance whose frequency shifts with the DC field via "
             "the Delta-E effect", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig_schematic.png", dpi=140)
plt.close(fig)
print("Figure: fig_schematic.png")
