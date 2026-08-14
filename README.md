# Certified, fit-free modeling of curvature and stress effects in contour-mode Delta-E magnetometers

Reproducibility package for the manuscript

> H. Talleb, *Certified, fit-free modeling of curvature and stress effects in
> contour-mode Delta-E effect magnetometers: an energy-averaged domain model and
> loss-limited detectivity*, submitted to IEEE Sensors Journal.

Every figure, every table and every number quoted in the paper is produced by a
deterministic script in this repository. There is no random number generation
and no manual data entry, so a single run reproduces the paper exactly.

## Measured reference data

The measurements the model is scored against are those published in

> A. D. Matyushov, B. Spetzler, M. Zaeimbashi, J. Zhou, Z. Qian, E. V. Golubeva,
> C. Tu, Y. Guo, B. F. Chen, D. Wang, A. Will-Cole, H. Chen, M. Rinaldi,
> J. McCord, F. Faupel and N. X. Sun, *Curvature and stress effects on the
> performance of contour-mode resonant Delta-E effect magnetometers*,
> Adv. Mater. Technol. **6**, 2100294 (2021). doi:10.1002/admt.202100294

No new device was fabricated for the present work. Because the published figures
are vector graphics, the individual device values are recovered exactly rather
than estimated from a raster image: `digitize_paper_figures.py` reads the
centroid of every marker from the graphics content stream of the article and
maps it to data coordinates through the axis tick marks. The affine calibration
residual on the ticks is below 1e-3 in data units for every panel.

The recovered values are shipped here as plain JSON (`measured_*.json`,
`model_fig10.json`) so that the rest of the package runs without the source PDF.
They remain the intellectual property of the authors above and are redistributed
only as the coordinates read from their published figures, with attribution.

**Verification of the digitization.** Refitting Eq. (4) of the reference work to
the recovered fixed-frequency subsets returns non-magnetic damping constants of
0.027 and 0.017 (units of 1e8 Hz), the values listed in its Table 2. This is run
automatically by `run_validation.py`.

| file | content | N |
| --- | --- | --- |
| `measured_fig3a.json` | response vs curvature, Fig. 3a | 64 devices |
| `measured_fig6a.json` | quality factor vs response, Fig. 6a | 51 devices |
| `measured_fig6c.json` | subset at 246 MHz, Fig. 6c | 12 devices |
| `measured_fig6d.json` | subset at 138 MHz, Fig. 6d | 13 devices |
| `model_fig10.json` | the two-domain model curve of Fig. 10 | 47 samples |

## Install and run

```bash
pip install -r requirements.txt
./run                        # everything, output collected in results/
```

`./run` is also the Code Ocean entry point. It takes about a minute on one core.
The individual drivers can be called separately:

```bash
python run_validation.py     # Figs. 4, 6, 7 and Table I, plus validation.json
python run_repro.py          # Figs. 3 and 5
python run_extensions.py     # Figs. 8 and 9, certified bounds and inverse design
python run_fe2d.py           # Fig. 10, 2D relaxation with clamped anchors
python run_fieldmaps.py      # Figs. 2 and 11, through-thickness and xy fields
python run_graphical_abstract.py   # graphical abstract, numbers read from validation.json
```

### Figure map

| manuscript | file |
| --- | --- |
| Fig. 2 | `fig_fields_z.png` |
| Fig. 3 | `fig_9d_aniso.png` |
| Fig. 4 | `fig_validation_response.png` |
| Fig. 5 | `fig_10_frH.png` |
| Fig. 6 | `fig_validation_Q.png` |
| Fig. 7 | `fig_validation_detectivity.png` |
| Fig. 8 | `fig_cert_band.png` |
| Fig. 9 | `fig_design.png` |
| Fig. 10 | `fig_fe2d.png` |
| Fig. 11 | `fig_fields_xy.png` |
| Table I | printed by `run_validation.py`, stored in `validation.json` |

`run_repro.py` and `run_extensions.py` also emit `fig_3a_response.png`,
`fig_6_Q.png` and `fig_detectivity.png`. These are the earlier model-only
versions of Figs. 4, 6 and 7, kept because they show the model without the
measured overlay; the manuscript uses the `fig_validation_*` versions, which
are scored against the measurements.

To regenerate the JSON data files from the source article instead of using the
shipped copies, place the published PDF next to the scripts and run

```bash
python digitize_paper_figures.py matyushov2021.pdf
```

The PDF itself is not redistributed here.

## Modules

| module | role |
| --- | --- |
| `materials.py` | material constants of the FeGaB / AlN / Pt stack, taken from the reference work |
| `mechanics.py` | classical lamination theory: initial stress to curvature and through-thickness stress |
| `magnetoelastic.py` | stress-induced anisotropy `(K_eff, phi_eff)`, per slice and volume-effective |
| `deltaE.py` | Delta-E effect: coherent rotation, energy-averaged domain model (DEAM), response |
| `paper_fits.py` | the fitted `K_eff(kappa)` and `phi_eff(kappa)` of the reference work, used only for the comparison branch |
| `q_factor.py` | quality factor from the magnetoelastic softening compliance |
| `extensions.py` | certified two-sided bounds and closed-form stress compensation |
| `fe2d.py` | 2D plane-stress relaxation with clamped anchors |

## Expected output of `run_validation.py`

```
RESPONSE  (64 measured devices, kappa 0.22-7.33 1/mm)
  measured  : 0.013-1.309 %  spread x101
  deam    : v=5%  spread x34    RMS factor 1.85  median dev 41%
  squire  : v=2%  spread x2     RMS factor 3.56  median dev 72%
  fit-free: no fitted anisotropy, spread x20 up to kappa=5.8, RMS factor 1.92
  [1] Fig10: published two-domain model, spread x45, RMS factor 2.11, median dev 36%
QUALITY FACTOR  (51 measured devices in the ensemble)
  shared constants: Q0 = 540, tau = 4.00 ps per % of response
DETECTIVITY  (25 devices with known f_r)
  pooled: response x57 -> loss-limited detectivity x23 (compression x2.5)
```

## Contact

H. Talleb, Sorbonne Universite, Universite Paris-Saclay, CNRS, CentraleSupelec,
Laboratoire de Genie Electrique et Electronique de Paris (GeePs),
hakeim.talleb@sorbonne-universite.fr
