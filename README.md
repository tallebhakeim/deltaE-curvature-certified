# Reproduction + innovation : courbure & contrainte dans les magnétomètres à effet ΔE

Reproduction Python autonome de :

> A. D. Matyushov, B. Spetzler, ... N. X. Sun,
> *Curvature and Stress Effects on the Performance of Contour-Mode Resonant
> ΔE Effect Magnetometers*, Adv. Mater. Technol. **6**, 2100294 (2021).

Résultat central du papier : dans des résonateurs NEMS FeGaB/AlN/Pt (~130-250 MHz),
la **courbure résiduelle κ = 1/R** de la plaque (manifestation de la contrainte de
dépôt) contrôle la réponse fréquentielle Δf_r/f_r,min sur ~2 ordres de grandeur,
et anti-corrèle avec le facteur de qualité Q (mécanisme de perte magnétique).

## Chaîne de modèle (fidèle au papier)

1. `mechanics.py` — plaque laminée (CLT/ABD) avec contrainte initiale par couche ;
   relaxation libre → courbure κ et champ de contrainte σ(z) à travers l'épaisseur,
   avec **axe neutre** (position fixée par les rapports d'épaisseur, invariante vs κ,
   comme observé). Remplace le FEM COMSOL du papier.
2. `magnetoelastic.py` — énergie magnéto-élastique (éq. 5-6) → anisotropie induite
   K_σ(z), angle φ_σ(z) ; superposition de l'anisotropie de champ K_u ; moyennes
   volumiques → K_eff(κ), φ_eff(κ). **Reproduit Fig 9d.**
3. `deltaE.py` — modulus E(H) via ramollissement magnéto-élastique
   1/E = 1/E0 + dλ/dσ, puis f_r(H) et Δf_r/f_r,min. **Reproduit Fig 3a et Fig 10.**

## Fidélité (constantes verbatim du papier)

| Grandeur | Papier | Reproduction |
|---|---|---|
| κ (plage dispositifs) | ~0 – 7 mm⁻¹ | 0 – 8 mm⁻¹ |
| K_eff(κ) | linéaire, ~12 kJ/m³ à κ=7 | linéaire (fit), forme reproduite par FEM |
| φ_eff | 90° → plateau ~60° | 90° → plateau 60° (fit) ; FEM plateaue aussi |
| Δf_r/f_r,min plat | ~2.5 % | 2.5 % (calibré) |
| Plage dynamique Δf_r | ~×100 (2 ordres) | **×33 (moyenné)** ; **×33 aussi en rotation cohérente** (voir avertissement) |
| Q vs réponse | anti-corrélé (Fig 6) | anti-corrélé, ×7 |

Notes :
- **φ_eff** : la moyenne circulaire des angles par tranche est dégénérée pour une
  distribution bimodale 0°/90° ; la bonne méthode (papier p.12) est d'extraire l'axe
  facile du paysage d'énergie moyenné en z, qui **plateaue**. Le papier alimente
  d'ailleurs le modèle domaine avec des fonctions **fittées** φ_eff(κ) (loi puissance)
  et K_eff(κ) (linéaire), reproduites dans `paper_fits.py`. Notre FEM en donne la forme
  (départ 90°, plateau), à magnitude réduite/miroir car une plaque libre sur-relaxe la
  contrainte biaxiale (seul le désaccord inter-couches subsiste).
- ⚠ **AVERTISSEMENT (19/09/2026), défaut numérique corrigé** : l'affirmation « le
  mono-domaine sature à ×2 (puis ×1,5), donc le modèle moyenné est nécessaire » était
  un **artefact de discrétisation**, pas un résultat. `squire_E` et `deam_E` évaluent
  dλ/dσ par différence finie sur une sonde dσ = 0,2 MPa, après avoir cherché l'angle
  d'équilibre sur la grille fixe de 721 points de `deltaE.py`. Or la sonde déplace
  l'angle d'équilibre de beaucoup moins que le pas de grille (8,7e-3 rad), donc la
  dérivée du mono-domaine est quantifiée, et effondrée. En raffinant la grille elle
  remonte à ×22 (n = 21601) et continue de monter. `deltaE_analytic.py` donne les deux
  compliances **en forme fermée** (différentiation implicite de g'(θ) = 0 pour la
  rotation cohérente, identité fluctuation-réponse dλ/dσ = A_s (3λ_s/2)² Var(cos²θ)
  pour la population). Résultat vérifié et indépendant de toute grille : rotation
  cohérente ×28,9 / RMS ×1,818 et moyenné ×30,3 / RMS ×1,851, avec le **même** v
  global ≈ 4,8 %. Les deux modèles partagent la limite raide (A_s → ∞), ce que la
  forme fermée démontre. **Les données ne discriminent donc pas les deux modèles** et
  le résultat robuste est ailleurs : une seule constante d'échelle à l'échelle du
  wafer suffit, là où [1] en ajuste une par dispositif.

## Innovations ajoutées (au-delà du papier)

1. **Une seule constante globale** à la place du paramètre d'ajustement *par
   dispositif* v = 8 % de [1] : v ≈ 4,8 % (IC 95 % bootstrap [4,2 ; 5,4] %) pour les
   64 dispositifs. Vaut aussi bien pour la rotation cohérente que pour le modèle
   énergie-moyenné (`deltaE_analytic.py`), qui ne sont pas discriminés par ces
   mesures. Le paramètre de largeur de population A_s = c/K_eff n'est **pas**
   identifiable (4 % de variation du score pour c de 2 à 1000).
2. **ΔE intégré en épaisseur** (`deltaE.py::frequency_response_profile`) : la
   compensation entre tranches de part et d'autre de l'axe neutre reproduit
   *physiquement* la forte chute de réponse — origine microscopique du petit v du papier.
3. **Q(κ) par perte physique** (`q_factor.py`) : le facteur de qualité découle du même
   ramollissement ΔE (perte magnéto-élastique 1/Q_mag ∝ α_m·E0·dλ/dσ), transformant
   l'éq. 4 phénoménologique en relation prédictive. **Reproduit Fig 6.**
   Corollaire nouveau (`run_extensions.py::fig_detectivity`) : en couplant Q dans la
   **détectivité** (LOD ∝ (f_r/Q)/S), l'écart ×46 de la réponse brute se réduit à **×6,5**
   en LOD, car le faible Q des dispositifs plats compense leur sensibilité élevée. La
   dispersion « ×100 » que le papier voit comme un problème de rendement est donc bien
   moins sévère sur la détectivité réelle — conclusion absente du papier.
4. **Certification** (`extensions.py`) : bornes garanties (enclosure exacte, car la
   mécanique est linéaire) sur κ puis sur Δf_r à partir d'une *boîte d'incertitude de
   contrainte* → bande de performance certifiée (style Prager-Synge).
5. **Design inverse** (`extensions.py`) : couche de compensation de contrainte
   (solution close, linéaire) qui annule κ → réponse ΔE maximale.
6. **FE 2D rétention par ancrage** (`fe2d.py`) : relaxation plane-stress Q4 de
   l'empreinte de la plaque avec les 2 anchors encastrés. Montre que l'ancrage
   retient ~64 % de l'anisotropie résiduelle au centre (vs ~13 % plaque libre CLT)
   → Kσ ~2 kJ/m³ dans la gamme du papier, et reproduit la non-uniformité MOKE
   centre/bords. Ferme l'écart de magnitude de la Fig 1.

## Manuscrit + cover letter

- `build_manuscript.js` → `Article_DeltaE_curvature_certified_v1.docx` (IOP/SMS,
  anglais, 11 figures dont schéma d'illustration + cartographies de champs,
  Table 1 de constantes, sans em-dash).
- Figures d'illustration/champs : `run_illustration.py` (schéma capteur →
  fig_schematic), `run_fieldmaps.py` (profils σ(z)/Kσ(z)/φσ(z) → fig_fields_z ;
  cartographies 2D φσ/Kσ → fig_fields_xy, repro Fig 9a,b du papier).
- `build_cover_letter.js` → `Cover_letter_DeltaE_curvature_v1.docx`.
Régénérer : `node build_manuscript.js && node build_cover_letter.js`.

## Exécution

```bash
python3 run_repro.py        # Fig 9d, Fig 10, Fig 3a
python3 run_extensions.py    # Fig 6 (Q), certification, design inverse
```

Figures produites : `fig_9d_aniso.png`, `fig_10_frH.png`, `fig_3a_response.png`,
`fig_6_Q.png`, `fig_cert_band.png`, `fig_detectivity.png`, `fig_design.png`.

## Fichiers

- `materials.py` — constantes FeGaB/AlN/Pt (verbatim papier)
- `mechanics.py` — plaque laminée, κ et σ(z)
- `magnetoelastic.py` — K_σ, φ_σ, K_eff(κ), φ_eff(κ)
- `deltaE.py` — Squire, DEAM, intégration en épaisseur, f_r(H)
- `paper_fits.py` — fonctions fittées K_eff(κ) (linéaire) et φ_eff(κ) (puissance, plateau 60°), comme le papier
- `q_factor.py` — Q(κ) par perte physique
- `extensions.py` — certification + design inverse
- `run_repro.py`, `run_extensions.py` — pilotes / figures (dont détectivité)

## Réponse aux relecteurs IEEE Sensors (19/09/2026)

Voir `REVISION_NOTES.md` pour les chiffres et les deux défauts numériques corrigés.

| Script | Objet | Sorties |
|---|---|---|
| `deltaE_analytic.py` | compliances magnéto-élastiques en forme fermée, sans grille | (bibliothèque) |
| `crossval.py` | modèles, ajustements continus, découpages de population | (bibliothèque) |
| `run_crossvalidation.py` | validation hors échantillon, transfert entre familles | `crossvalidation.json`, `fig_crossvalidation.png` |
| `run_identifiability.py` | identifiabilité pratique des constantes globales | `identifiability.json`, `fig_identifiability.png` |
| `run_bounds_population.py` | enveloppe garantie et couverture de la population | `bounds_population.json`, `fig_bounds_population.png` |

`run_validation.py` a été porté sur les modèles analytiques et son ajustement des
constantes de perte reparamétré ; `run_graphical_abstract.py` suit automatiquement,
puisqu'il lit `validation.json`.

Ordre d'exécution : `run_validation.py`, `run_crossvalidation.py`,
`run_identifiability.py`, `run_bounds_population.py`, `run_graphical_abstract.py`,
puis `node build_manuscript.js`, `node build_supplementary.js`,
`node build_cover_letter.js`.

### Brancher un second jeu de mesures

`run_external_validation.py` teste la loi de perte sur une famille de dispositifs
extérieure, **sans rien réajuster** : Q0 = 523 et tau = 4,07 ps sont imposés et on
regarde ce que ça donne. Déposer un fichier JSON dans `external_data/` au format de
`external_data/_TEMPLATE.json.example`, puis lancer le script. Sans argument il fait
son auto-test sur les deux familles de [1] et doit redonner R² = +0,70 et +0,39.

Trois sorties par famille : la prédiction à l'aveugle, un contrôle où la même loi est
forcée aux fréquences de nos propres familles (ce que ferait une loi sans f_r
explicite), et le refit local pour voir si tau tombe dans notre intervalle à 95 %,
2,0 à 8,0 ps.

⚠ Le contrôle en fréquence est FAIBLE entre nos deux familles, 246 et 138 MHz ne
différant que d'un facteur 1,8 (0,70 contre 0,56 quand on se trompe de fréquence).
C'est précisément pourquoi une troisième famille, à une fréquence nettement
différente, vaudrait beaucoup plus que dix dispositifs de plus au même endroit.
