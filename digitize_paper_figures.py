"""Exact vector digitization of the measured data published in

  A. D. Matyushov, B. Spetzler, ... N. X. Sun,
  "Curvature and stress effects on the performance of contour-mode resonant
   Delta-E effect magnetometers", Adv. Mater. Technol. 6, 2100294 (2021).

The published figures are stored as vector graphics, so the marker centroids
are read directly from the PDF content stream and mapped to data coordinates
through the axis tick marks. No image-based picking, no manual point clicking:
the calibration residual on the tick positions is reported and is below 1e-3
in data units.

Only the *data values* are recovered (data are not copyrightable); the figures
of the present work replot them, they do not reproduce the published artwork.

Usage:  python digitize_paper_figures.py [path/to/paper.pdf]
Output: measured_fig3a.json   Delta f_r/f_r,min (%) versus curvature (1/mm)
        measured_fig6a.json   Q versus Delta f_r/f_r,min (%)
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    raise SystemExit("PyMuPDF is required: pip install pymupdf")


def _same(colour, ref, tol=1e-3):
    return colour is not None and all(abs(a - b) < tol for a, b in zip(colour, ref))


def _calibrate(ticks, values):
    """Affine map from page coordinate to data coordinate."""
    ticks = np.asarray(sorted(ticks), dtype=float)
    values = np.asarray(values, dtype=float)
    if ticks.size != values.size:
        raise RuntimeError(f"found {ticks.size} ticks for {values.size} labels")
    coef = np.polyfit(ticks, values, 1)
    residual = float(np.max(np.abs(np.polyval(coef, ticks) - values)))
    return coef, residual


def _dedupe(points, tol_x, tol_y):
    """Vector markers are painted twice (outline then fill); keep one copy."""
    kept = []
    for x, y in sorted(points):
        if any(abs(x - u) < tol_x and abs(y - v) < tol_y for u, v in kept):
            continue
        kept.append((x, y))
    return kept


def _extract(page, frame, marker_colour, tick_colour, xticks, yticks,
             tick_len, tick_bar=0.5, exclude=(), tol=(5e-3, 2e-3)):
    x0, y0, x1, y1 = frame
    drawings = page.get_drawings()

    xt, yt = [], []
    for g in drawings:
        r = g["rect"]
        if not _same(g.get("fill"), tick_colour):
            continue
        if not (x0 - 4 < r.x0 and r.x1 < x1 + 4 and y0 - 4 < r.y0 and r.y1 < y1 + 4):
            continue
        if abs(r.width - tick_bar) < 0.08 and abs(r.height - tick_len) < 0.12 \
                and abs(r.y1 - y1) < 2.0:
            xt.append(r.x0 + r.width / 2)
        if abs(r.height - tick_bar) < 0.08 and abs(r.width - tick_len) < 0.12 \
                and abs(r.x0 - x0) < 2.0:
            yt.append(r.y0 + r.height / 2)

    cx_map, rx = _calibrate(xt, sorted(xticks))
    cy_map, ry = _calibrate(yt, sorted(yticks, reverse=True))

    raw = []
    for g in drawings:
        r = g["rect"]
        if not _same(g.get("fill"), marker_colour):
            continue
        if not (r.x0 > x0 - 3 and r.x1 < x1 + 3 and r.y0 > y0 - 3 and r.y1 < y1 + 3):
            continue
        if max(r.width, r.height) > 6.0:  # fitted curves, frame, legend box
            continue
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if any(ex[0] < cx < ex[2] and ex[1] < cy < ex[3] for ex in exclude):
            continue
        raw.append((float(np.polyval(cx_map, cx)), float(np.polyval(cy_map, cy))))

    pts = _dedupe(raw, *tol)
    return pts, {"tick_residual_x": rx, "tick_residual_y": ry,
                 "raw_shapes": len(raw), "unique_markers": len(pts)}


def _extract_curve(page, frame, tick_colour, xticks, yticks, tick_len, tick_bar,
                   is_curve_colour, exclude=(), nbins=90):
    """Recover a plotted model curve.

    The path is a stroked outline, so both edges of the drawn line are present;
    binning in x and taking the median recovers the centre line.  Regions listed
    in `exclude` (typically the legend, which repeats the line style) are dropped.
    """
    x0, y0, x1, y1 = frame
    xt, yt = [], []
    verts = []
    for g in page.get_drawings():
        r = g["rect"]
        fill = g.get("fill")
        if fill and _same(fill, tick_colour) and \
                x0 - 4 < r.x0 and r.x1 < x1 + 4 and y0 - 4 < r.y0 and r.y1 < y1 + 4:
            if abs(r.width - tick_bar) < 0.08 and abs(r.height - tick_len) < 0.5 \
                    and abs(r.y1 - y1) < 2.5:
                xt.append(r.x0 + r.width / 2)
            if abs(r.height - tick_bar) < 0.08 and abs(r.width - tick_len) < 0.5 \
                    and abs(r.x0 - x0) < 2.5:
                yt.append(r.y0 + r.height / 2)
        if fill and is_curve_colour(fill):
            for item in g["items"]:
                for el in item[1:]:
                    if not (hasattr(el, "x") and x0 <= el.x <= x1 and y0 <= el.y <= y1):
                        continue
                    if any(ex[0] < el.x < ex[2] and ex[1] < el.y < ex[3]
                           for ex in exclude):
                        continue
                    verts.append((el.x, el.y))

    cx, rx = _calibrate(xt, sorted(xticks))
    cy, ry = _calibrate(yt, sorted(yticks, reverse=True))
    arr = np.asarray(verts)
    kx = np.polyval(cx, arr[:, 0])
    vy = np.maximum(np.polyval(cy, arr[:, 1]), 1e-4)

    edges = np.linspace(kx.min(), kx.max(), nbins + 1)
    idx = np.clip(np.digitize(kx, edges) - 1, 0, nbins - 1)
    pts = []
    for b in range(nbins):
        sel = idx == b
        if sel.any():
            pts.append((float(np.median(kx[sel])), float(np.median(vy[sel]))))
    return pts, {"tick_residual_x": rx, "tick_residual_y": ry,
                 "raw_shapes": len(verts), "unique_markers": len(pts)}


NAVY_F3 = (0.18242160975933075, 0.26105135679244995, 0.6079041957855225)
BLACK_F3 = (0.01112382672727108, 0.015518425032496452, 0.01744106225669384)
DARK_F6 = (0.13669031858444214, 0.12195010483264923, 0.1252918243408203)


def main(pdf_path):
    doc = fitz.open(pdf_path)

    # ---- Figure 3a: Delta f_r/f_r,min (%) versus curvature kappa (1/mm) -----
    pts3, info3 = _extract(
        doc[4],
        frame=(144.899, 91.020, 275.078, 195.811),
        marker_colour=NAVY_F3, tick_colour=BLACK_F3,
        xticks=[0, 2, 4, 6], yticks=[0.0, 0.4, 0.8, 1.2], tick_len=3.04,
        exclude=[(242.7, 92.4, 273.4, 163.7)],          # "Sensor ID" legend box
    )

    # ---- Figure 6a: quality factor Q versus Delta f_r/f_r,min (%) ----------
    navy6 = _dominant_marker_colour(doc[7], (143.86, 90.23, 282.43, 201.84))
    pts6, info6 = _extract(
        doc[7],
        frame=(143.86, 90.23, 282.43, 201.84),
        marker_colour=navy6, tick_colour=DARK_F6,
        xticks=[0.0, 0.4, 0.8, 1.2], yticks=[0, 200, 400, 600],
        tick_len=3.81, tick_bar=0.57,
        exclude=[(224.0, 91.0, 281.0, 152.0)],          # "Sensor ID" legend box
        tol=(3e-3, 2.0),
    )

    # ---- Figures 6c, 6d: the two fixed-f_r subsets the paper fits Eq. (4) to --
    subsets = {}
    for tag, frame, legend in (
        ("fig6c", (143.96, 252.43, 282.52, 364.11), (248.0, 253.8, 281.5, 281.2)),
        ("fig6d", (337.58, 252.34, 476.21, 364.02), (441.8, 253.6, 475.2, 280.9)),
    ):
        subsets[tag] = _extract(
            doc[7], frame=frame, marker_colour=navy6, tick_colour=DARK_F6,
            xticks=[0.0, 0.4, 0.8, 1.2], yticks=[0, 200, 400, 600],
            tick_len=3.81, tick_bar=0.57, exclude=[legend], tol=(3e-3, 2.0),
        )

    # ---- Figure 10: the two-domain model curve the paper fits to Fig. 3a -----
    pts10, info10 = _extract_curve(
        doc[11], frame=(107.14, 517.23, 258.66, 639.28), tick_colour=DARK_F6,
        xticks=[0, 2, 4, 6], yticks=[0.0, 0.4, 0.8, 1.2],
        tick_len=3.81, tick_bar=0.57,
        is_curve_colour=lambda f: f[0] > 0.85 and f[1] < 0.3 and f[2] > 0.3,
        exclude=[(201.4, 518.5, 257.1, 620.2)],          # legend box, same line style
    )

    out = Path(__file__).parent
    for name, pts, info, meta in (
        ("model_fig10", pts10, info10,
         {"figure": "Fig. 10", "x": "curvature kappa (1/mm)",
          "y": "Delta f_r / f_r,min (%)",
          "content": "two-domain model of [1] fitted with v = 8% per device"}),
        ("measured_fig3a", pts3, info3,
         {"figure": "Fig. 3a", "x": "curvature kappa (1/mm)",
          "y": "Delta f_r / f_r,min (%)"}),
        ("measured_fig6a", pts6, info6,
         {"figure": "Fig. 6a", "x": "Delta f_r / f_r,min (%)", "y": "quality factor Q"}),
        ("measured_fig6c", *subsets["fig6c"],
         {"figure": "Fig. 6c", "x": "Delta f_r / f_r,min (%)", "y": "quality factor Q",
          "subset": "sensor designs 3 and 4, mean f_r = 246 MHz"}),
        ("measured_fig6d", *subsets["fig6d"],
         {"figure": "Fig. 6d", "x": "Delta f_r / f_r,min (%)", "y": "quality factor Q",
          "subset": "sensor designs 7 and 8, mean f_r = 138 MHz"}),
    ):
        payload = dict(meta)
        payload["source"] = ("A. D. Matyushov et al., Adv. Mater. Technol. 6, "
                             "2100294 (2021); values digitized from the vector "
                             "content of the published figure")
        payload["calibration"] = info
        payload["points"] = [[round(a, 5), round(b, 5)] for a, b in pts]
        (out / f"{name}.json").write_text(json.dumps(payload, indent=1))
        print(f"{name}: {info['unique_markers']} markers "
              f"(tick residual {info['tick_residual_x']:.2e} / "
              f"{info['tick_residual_y']:.2e})")

    k = np.array([p[0] for p in pts3])
    r = np.array([p[1] for p in pts3])
    print(f"  Fig 3a: kappa {k.min():.2f}-{k.max():.2f} 1/mm, "
          f"response {r.min():.3f}-{r.max():.3f} %, spread x{r.max()/r.min():.0f}")
    q = np.array([p[1] for p in pts6])
    x = np.array([p[0] for p in pts6])
    print(f"  Fig 6a: response {x.min():.3f}-{x.max():.3f} %, Q {q.min():.0f}-{q.max():.0f}, "
          f"Pearson r = {np.corrcoef(x, q)[0,1]:.3f}")


def _dominant_marker_colour(page, frame):
    """The scatter colour is the most frequent fill among small shapes."""
    from collections import Counter
    x0, y0, x1, y1 = frame
    c = Counter()
    for g in page.get_drawings():
        r = g["rect"]
        if g.get("fill") is None or max(r.width, r.height) > 6.0:
            continue
        if r.x0 > x0 - 3 and r.x1 < x1 + 3 and r.y0 > y0 - 3 and r.y1 < y1 + 3:
            c[tuple(round(v, 6) for v in g["fill"])] += 1
    return c.most_common(1)[0][0]


if __name__ == "__main__":
    default = Path(__file__).parent / "matyushov2021.pdf"
    main(sys.argv[1] if len(sys.argv) > 1 else default)
