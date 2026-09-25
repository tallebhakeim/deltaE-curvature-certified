# Notes de révision, article ΔE courbure (Sensors-112417-2026)

Rédigé le 19/09/2026, en réponse au rapport de relecture IEEE Sensors Journal
(décision : reject ; R1 reject, R2 « review again after resubmission »).

Scripts ajoutés dans ce dossier :

| Script | Objet |
|---|---|
| `deltaE_analytic.py` | compliances magnéto-élastiques en forme fermée, sans grille |
| `crossval.py` | bibliothèque : modèles, ajustements, découpages de population |
| `run_crossvalidation.py` | validation hors échantillon, transfert entre familles |
| `run_identifiability.py` | identifiabilité pratique des constantes globales |
| `run_bounds_population.py` | enveloppe garantie et couverture de la population |

Sorties : `crossvalidation.json`, `identifiability.json`, `bounds_population.json`,
`fig_crossvalidation.png`, `fig_identifiability.png`, `fig_bounds_population.png`.

---

## 1. Deux défauts numériques trouvés en route

Ils ne viennent pas des relecteurs. Ils sont apparus en instrumentant le code pour
répondre au point sur la circularité de la calibration.

### 1.1 Le « mono-domaine sature » était un artefact de grille

`squire_E` et `deam_E` évaluent dλ/dσ par différence finie sur une sonde
dσ = 0,2 MPa, après avoir localisé l'angle d'équilibre sur la grille fixe de 721
points de `deltaE.py`. La sonde déplace l'angle de bien moins que le pas de grille
(8,7·10⁻³ rad), donc la dérivée du mono-domaine est quantifiée et s'effondre.

En raffinant la grille, la gamme dynamique du mono-domaine passe de ×1,5 (n = 721)
à ×3,5 (n = 2161), ×10,8 (n = 7201), ×21,8 (n = 21601) et continue de monter.

`deltaE_analytic.py` supprime le problème en donnant les deux compliances en forme
fermée :

- rotation cohérente, par différentiation implicite de g'(θ) = 0 :
  dλ_x/dσ = (3λ_s/2)² sin²(2θ) / g''(θ) ;
- population moyennée, par identité fluctuation-réponse :
  dλ_x/dσ = A_s (3λ_s/2)² Var_p(cos²θ).

Vérifications : la branche moyennée analytique redonne exactement les valeurs du
code d'origine (0,71870 / 0,24848 / 0,02370 aux courbures 0,3 / 2 / 7) ; la
compliance varie bien en λ_s² (facteur 1,2100 mesuré pour λ_s ×1,1) ; les deux
expressions sont indépendantes de toute grille à 4 chiffres près.

La seconde identité montre aussi pourquoi les deux modèles se rejoignent : quand
A_s croît, la variance décroît en 1/(A_s g''), donc le produit tend vers la valeur
de rotation cohérente. Les deux modèles partagent la limite raide.

**Conséquence** : sur les 64 dispositifs mesurés, avec une seule constante globale,

| Modèle | v | RMS (facteur) | gamme dynamique |
|---|---|---|---|
| rotation cohérente (forme fermée) | 4,77 % | **1,818** | ×32,8 |
| population moyennée (forme fermée) | 4,87 % | **1,851** | ×34,4 |
| mono-domaine tel que soumis (grille 721) | 2,07 % | *3,556* | *×1,5* |
| modèle publié de [1], un v par dispositif | par dispositif | 2,11 | ×45 |

L'affirmation centrale du manuscrit soumis, « le mono-domaine sature et ne peut pas
reproduire la dispersion, donc la moyenne d'énergie est nécessaire », est fausse.
Bien calculé, le mono-domaine fait marginalement **mieux**. Les mesures ne
discriminent pas les deux modèles.

Ce qui survit, et qui est plus solide : **une seule constante à l'échelle du wafer
suffit** là où [1] en ajuste une par dispositif, et le résultat ne dépend pas du
choix du modèle d'aimantation.

### 1.2 Les constantes de perte publiées n'étaient pas un ajustement

`run_validation.py` appelle `least_squares(resid, [1/540, 4e-12])`. Les deux
inconnues sont séparées par neuf décades, le jacobien numérique par défaut est
dégénéré, et le solveur **rend son point de départ inchangé**. Vérifié : en partant
de (1/300, 8e-12) il rend exactement Q0 = 300 et τ = 8 ps.

Les valeurs Q0 = 540 et τ = 4,0 ps du manuscrit sont donc le point de départ, pas un
résultat. Reparamétré en u = (1000/Q0, τ en ps), l'ajustement converge vers le même
optimum depuis n'importe quel départ testé (Q0 de 300 à 2000, τ de 1 à 20 ps) :

**Q0 = 523 ± 64 et τ = 4,07 ± 1,41 ps par % de réponse.**

Les valeurs publiées tombent dans l'intervalle, la conclusion qualitative tient,
mais les chiffres et leur statut doivent être corrigés.

---

## 2. Réponse au point de circularité (R2 nº 2)

Toutes les constantes sont désormais estimées sur des dispositifs que la prédiction
n'a pas vus.

### 2.1 Réponse, interpolation

| Découpage | RMS apprentissage | RMS **test** |
|---|---|---|
| moitiés alternées | 1,650 | **1,980** |
| 5 plis | | **1,826** en moyenne |
| 200 moitiés aléatoires stratifiées | | **1,840** (5-95 % : 1,659 à 2,010) |

Référence : le modèle publié de [1], qui ajuste une fraction **par dispositif**,
obtient 2,11 en échantillon, et 2,305 sur les mêmes points de test.

Autrement dit : une constante unique estimée sur la moitié de la population prédit
l'autre moitié mieux que le modèle publié ne décrit les dispositifs qu'il a ajustés.

### 2.2 Réponse, extrapolation

Apprentissage sur la moitié basse courbure (0,22 à 2,13 mm⁻¹), prédiction sur la
moitié haute (2,29 à 7,33 mm⁻¹) : RMS test **1,989** (contre 2,351 pour [1] sur ces
points). Sens inverse : **2,121** (contre 1,860 pour [1]).

La fraction ajustée dérive de 6,0 % à 3,8 % selon le sens, signe honnête d'une
erreur de forme résiduelle, mais le score de test reste au niveau du modèle publié
noté en échantillon.

### 2.3 Stabilité de la constante

v = 4,78 ± 0,17 % sur les 5 plis, 4,78 ± 0,33 % sur 200 moitiés aléatoires
(étendue 4,01 à 5,94 %). Intervalle de confiance bootstrap à 95 % : **[4,23 ; 5,39] %**.

### 2.4 Pertes, transfert entre familles de dispositifs

C'est le test le plus probant, et il répond aussi à la demande de R1 d'un résultat
physique nouveau : la loi 1/Q = 1/Q0 + 2π f_r τ x fait apparaître f_r explicitement,
donc ses deux constantes peuvent voyager d'une famille à l'autre. Celles de
l'éq. (4) de [1], qui absorbent f_r, ne le peuvent pas.

| Ajusté sur | Prédit | 2 constantes partagées | éq. (4) de [1] transférée |
|---|---|---|---|
| designs 3, 4 (246 MHz) | designs 7, 8 (138 MHz) | R² = **+0,37** | R² = −0,26 |
| designs 7, 8 (138 MHz) | designs 3, 4 (246 MHz) | R² = **+0,70** | R² = −1,97 |

Les constantes restent stables d'une famille à l'autre : Q0 de 507 à 566, τ de 3,81
à 4,51 ps. Leave-one-out sur les 25 dispositifs : Q0 = 523 ± 17, τ = 4,07 ± 0,16 ps.

Un R² négatif signifie « pire que prédire la moyenne » : la forme de [1] ne transfère
pas, la nôtre oui.

---

## 3. Réponse au point d'identifiabilité (R2 nº 2, suite)

### 3.1 La largeur de population n'est pas identifiable

Le paramètre A_s = c/K_eff était codé en dur à c = 10, donc une constante implicite
non déclarée : R2 avait raison de la compter. Profilé de c = 2 à c = 1000, le score
ne bouge que de 4 % (1,801 à 1,874).

- pour c ≥ 5, v reste confiné à 4,78 à 5,16 % ;
- en dessous, (v, c) se compensent le long d'une vallée plate : v monte à 10,8 % à
  c = 2 pour un gain de 1 % sur le score.

Conclusion honnête : ce paramètre n'est pas déterminé par ces données, et les
prédictions y sont insensibles. Ce n'est donc pas une constante ajustée mais un
choix de régularisation, ce qui **réduit** le compte de paramètres au lieu de
l'augmenter.

### 3.2 Les deux constantes de perte sont séparément déterminées

Corrélation(Q0, τ) = +0,58, conditionnement du jacobien 7,5, donc pas de
dégénérescence. Intervalle à 95 % sur τ par profil, Q0 réajusté à chaque point :
**2,0 à 8,0 ps par %**. La valeur 4,0 ps y est confortablement.

### 3.3 La contrainte de base est bien absorbée par l'échelle ajustée

R2 soupçonnait une compensation avec le tenseur (45, 75, 5) MPa. C'est vrai, et
c'est maintenant chiffré. En balayant le rapport d'anisotropie σ₂₂/σ₁₁ de 1,2 à 2,2
et le cisaillement de 0 à 10 MPa, le score reste entre 1,731 et 2,352 tandis que
**l'échelle ajustée bouge d'un facteur 2,73**.

Ces données ne déterminent donc pas le tenseur de contrainte supposé. Ce qu'elles
déterminent, c'est la **forme** Δf_r/f_r(κ) à un facteur près. Il faut le dire.

### 3.4 Effet secondaire favorable

À périmètre égal (les 60 points couverts par la branche intégrée en épaisseur),
supprimer les deux fonctions d'anisotropie ajustées de [1] au profit du champ de
contrainte de la plaque laminée ne coûte **rien** : 1,830 avec, 1,820 sans. Le
manuscrit annonçait un coût de 4 %, qui venait du modèle en différences finies et
d'une comparaison sur des jeux de points différents.

---

## 4. Ce qui doit changer dans le manuscrit

1. **Titre** : retirer « Fit-Free » (il reste une constante globale) et retirer ou
   qualifier « Certified ». Les deux relecteurs le demandent, et ils ont raison.
2. **Abstract** : définir ce que [1] ajuste (la fraction volumique magnétiquement
   active v, un par dispositif), retirer « saturates and cannot reproduce the
   spread », corriger Q0 et τ avec leurs intervalles.
3. **§III-B et Fig. 1** : refaire la comparaison des deux modèles d'aimantation, qui
   ne sont pas discriminés ; le message devient « une constante de wafer suffit,
   quel que soit le modèle ».
4. **§III-C** : Q0 = 523 ± 64, τ = 4,07 ± 1,41 ps, et dire que ce sont des valeurs
   ajustées avec leur intervalle.
5. **Nouvelle section** : validation hors échantillon et identifiabilité (§2 et §3
   ci-dessus), avec `fig_crossvalidation.png` et `fig_identifiability.png`.
6. **§III-F** : remplacer le paragraphe de périmètre, qui a servi de munition aux
   deux relecteurs, par les chiffres de couverture de population (§5 ci-dessous).
7. **Ce qui ne bouge pas** : la détectivité, ×56,7 en réponse comprimée à ×22,8
   (facteur 2,49), calculée uniquement à partir des f_r, Q et réponses mesurés,
   donc indépendante du modèle d'aimantation.

---

## 5. Réponse au point sur les bornes (R1 et R2 nº 3)

Les deux relecteurs disent la même chose : une boîte ±20 % sur la seule contrainte
résiduelle donne un intervalle étroit par construction, qui ne dit rien de la
dispersion entre dispositifs, donc ne soutient pas une revendication de
qualification industrielle. Ils ont raison. Voici la réponse sur leur terrain.

### 5.1 D'abord, corriger un chiffre du manuscrit

Le manuscrit annonçait une dispersion « d'un facteur 3 à courbure fixée », et les
deux relecteurs l'ont reprise telle quelle. Par bandes de courbure, la mesure donne :

| κ (mm⁻¹) | dispositifs | dispersion |
|---|---|---|
| 0,22 à 0,44 | 4 | ×1,7 |
| 0,44 à 0,89 | 10 | ×1,9 |
| 0,89 à 1,80 | 15 | ×10,3 |
| 1,80 à 3,63 | 21 | ×6,9 |
| 3,63 à 7,33 | 14 | **×13,3** |

Le facteur 3 sous-estimait donc notre propre difficulté d'un facteur 4.

### 5.2 Une enveloppe réellement garantie

La courbure étant ici mesurée et non calculée, ce qui compte à courbure fixée est
l'incertitude portée par les entrées magnétiques et géométriques. `run_bounds_population.py`
encadre la réponse sur une boîte à sept entrées.

Point de méthode, qui était faux dans ma première version : la réponse **n'est pas
monotone** en angle d'axe facile, puisque la compliance porte un facteur sin²(2θ) et
possède donc un optimum intérieur, avec une pointe. Sur 420 sondages, 24 départs à
la monotonie, tous sur cet axe. L'énumération des sommets seule n'est donc pas
licite : elle rate l'optimum intérieur. Cet axe est maintenant balayé densément
(17 nœuds) puis les deux extrêmes sont raffinés localement. Le raffinement déplace
les bornes de **5,8 % au plus**, ce qui est le résidu de l'encadrement. Les cinq
autres entrées sont vérifiées monotones, avec une tolérance qui rejette le bruit
numérique, donc leurs extrêmes sont bien aux bornes d'intervalle.

### 5.3 Ce que couvre la boîte nominale

Boîte nominale (v ±12 % d'après le bootstrap, anisotropie ±25 et ±20 %, axe facile
±10°, λ_s ±10 %, M_s ±5 %, épaisseur ±5 %) : bande médiane **×3,02**, qui enclot
**49 dispositifs sur 64, soit 77 %**.

### 5.4 La question inverse, qui est la seule utile

| Boîte élargie | bande médiane | couverture |
|---|---|---|
| ×1,0 | ×3,02 | 77 % |
| ×1,5 | ×5,27 | 91 % |
| **×2,0** | **×9,3** | **97 %** |
| ×2,5 | ×16,9 | 98 % |
| ×3,0 | ×32,4 | 100 % |

Enclore 95 % de la population demande d'élargir la boîte d'un facteur 2, soit
v ±24 %, anisotropie ±50 %, axe facile ±20°, λ_s ±20 %, épaisseur ±10 %.

C'est la formulation honnête et elle reste utile : le certificat cesse d'être une
spécification pour un dispositif, une bande ×9 n'en est pas une, et devient un
**énoncé de rendement**, à savoir quelle dispersion de procédé explique la
dispersion observée. C'est ce qu'il faut écrire à la place de « certified
manufacturing qualification », que les deux relecteurs ont refusé à juste titre.

---

## 6. Validation expérimentale extérieure (19/09/2026, recherche par agents)

Trois recherches bibliographiques parallèles. Résultat : la loi de perte est **confirmée
hors échantillon à haute fréquence** et **réfutée par extrapolation au kilohertz**. Les
deux ensemble donnent un énoncé borné, ce qui vaut mieux que l'énoncé universel que le
manuscrit porte aujourd'hui.

Banc d'essai : `run_external_validation.py`, données dans `external_data/`.

### 6.1 Le test par incrément, où Q0 disparaît

Entre le point de fonctionnement magnétique et la saturation, la perte non magnétique
ne change pas, donc

    1/Q(x) − 1/Q_sat = 2π f_r τ x

et Q0 s'élimine. C'est la forme la plus nette du test : on peut utiliser un dispositif
d'un autre laboratoire sans rien savoir de ses pertes d'ancrage ou thermoélastiques.

### 6.2 Confirmation à 215 MHz, wafer et empilement différents

T. Nan, Y. Hui, M. Rinaldi, N. X. Sun, *Sci. Rep.* **3**, 1985 (2013),
doi 10.1038/srep01985, libre accès. Résonateur contour AlN 250 nm /
(FeGaB/Al₂O₃)×10 250 nm / Pt 50 nm à 215 MHz. Cité verbatim du texte : « the Q started
from 735 (0 Oe) then reached the minimum value of 250 at the transition magnetic field
(15 Oe) and finally saturated to 1400 at high magnetic fields (> 60 Oe) ».

Incrément mesuré 1/250 − 1/1400 = 3,29·10⁻³. Le balayage de fréquence n'est PAS donné
dans le texte, seulement en Fig. 2d, donc les deux lectures possibles sont propagées :

| lecture de x | τ impliqué | dans notre intervalle [2,0 ; 8,0] ps/% ? |
|---|---|---|
| 0,74 % (champ nul → creux) | **3,29 ps/%** | oui |
| 1,10 % (saturation → creux) | **2,21 ps/%** | oui |

Quelle que soit la lecture, τ tombe dans l'intervalle tiré de [1]. C'est une validation
hors échantillon sur un wafer différent, un empilement différent (multicouche
FeGaB/Al₂O₃ au lieu d'une couche unique) et une génération antérieure de huit ans.
Même laboratoire que [1] toutefois (Sun, Northeastern) : à dire.

### 6.3 Réfutation par extrapolation au kilohertz

Les dispositifs ΔE de Kiel fonctionnent à quelques kilohertz. Sur le cantilever de
Durdaut et al. (arXiv:2003.01085, publié J. Microelectromech. Syst. **29**, 1347, 2020),
Q passe d'environ 1100 à saturation à environ 800 au point de fonctionnement à 7,45 kHz.
Incrément mesuré 3,4·10⁻⁴ ; notre loi prédit 1,3·10⁻⁷, soit **2600 fois trop peu**.

⚠ Ces valeurs de Q viennent de la lecture d'une figure par un agent et **ne sont pas
vérifiées à la source** : le PDF n'a pas pu être analysé par les outils disponibles. La
conclusion résiste à une grosse erreur de lecture, puisque l'écart est de trois ordres
de grandeur, mais il faut vérifier avant de citer.

### 6.4 Et surtout : nos propres données ne contraignent pas le facteur f_r

En profilant l'exposant n dans 1/Q = 1/Q0 + A (f_r/200 MHz)ⁿ x sur les 25 dispositifs
de [1] :

| n | Q0 | R² | RMS(Q) | |
|---|---|---|---|---|
| −1 | 510 | +0,43 | 147,5 | physique de l'éq. (4) de [1] |
| 0 | 523 | +0,51 | 137,4 | perte hystérétique, sans fréquence |
| **+1** | **523** | **+0,53** | **133,6** | notre loi |
| +2 | 512 | +0,51 | 136,5 | |

n = 1 est le meilleur, mais n = 0 n'est qu'à 3 % de RMS. **Nos données ne distinguent
pas les deux.** Le transfert entre familles, R² +0,37 et +0,70, vaut donc surtout comme
démonstration que des constantes partagées valent mieux que des constantes par famille :
avec n = 0 le transfert donne encore +0,29 et +0,54, et avec n = −1 encore +0,05 et
+0,17, tous bien au-dessus des −0,26 et −1,97 de la forme à quatre constantes de [1].

Autrement dit, ce qui sauve le transfert est le **partage des constantes**, pas le
facteur f_r. C'est le point à corriger dans le manuscrit.

### 6.5 Lecture physique, qui réconcilie les deux résultats

Un mécanisme de relaxation donne une perte en ωτ, négligeable au kilohertz et dominante
à 100 MHz. Une perte hystérétique de parois donne une perte indépendante de la
fréquence. Nos données à 138-246 MHz et celles de Nan à 215 MHz sont dans le régime de
relaxation ; les dispositifs de Kiel au kilohertz sont dans le régime de parois, ce que
Nan dit d'ailleurs lui-même de ses propres pertes (« magnetic loss associated with
magnetic domain wall activities »). La loi est donc une **limite haute fréquence**, pas
une loi universelle, et le croisement entre les deux régimes est une prédiction à tester.

### 6.6 Correction d'une prémisse des deux relecteurs

R2 écrit « relying exclusively on digitizing published plots from **a single wafer** in
[1] ». C'est faux, et vérifié à la source (PMC9090164) : « Each batch of sensors used in
Investigation 1 and Investigation 2 was fabricated on a **separate** Si wafer », et le
Tableau 1 donne huit conceptions à 224,4 / 224,4 / 245,8 / 245,6 / 172,4 / 171,9 /
136,8 / 137,1 MHz, soit **deux wafers et quatre fréquences distinctes**. À dire dans la
lettre, courtoisement, car cela retire une partie du reproche d'origine unique.

### 6.7 Sources retenues pour la suite, non encore exploitées

| Source | Ce qu'elle apporte | État |
|---|---|---|
| Ilgaz et al., *Sci. Rep.* **14**, 11075 (2024), doi 10.1038/s41598-024-59015-5 | Tables S1 et S3 : 43 couples (f_r, Q) sur 12 dispositifs, 79,8 kHz à 2,79 MHz, un seul empilement. Rien à digitaliser. | libre accès, préprint arXiv:2312.02903 |
| Spetzler et al., *Sensors* **21**, 2022 (2021), doi 10.3390/s21062022 | Table A1 : 6 modes d'un même dispositif, 7,65 à 175 kHz, **Q mesuré à saturation**, donc Q0 donné et non supposé. | libre accès, XML Europe PMC |
| Ilgaz et al., *APL* **126**, 084103 (2025), supplémentaire doi 10.60893/figshare.apl.28417778 | Table S1 : Q en fonction du champ de polarisation ET du niveau d'excitation, sur un même mode. Sépare la perte magnéto-élastique d'une perte de non-linéarité. | supplémentaire CC-BY sur figshare |
| Hu et al., *APL* **123**, 012406 (2023), préprint arXiv:2305.04473 | SAW FeCoSiB, harmoniques 1 à 11 de 128 MHz : ΔG et perte d'insertion aux mêmes fréquences. Laboratoire indépendant (UESTC). Le papier affirme que l'effet est piloté par le pôle FMR, donc **test adversarial** d'un τ constant. | préprint libre |
| Turutin et al., *Materials* **16**, 484 (2023), doi 10.3390/ma16020484 | Table 2 : le seul tableau inter-laboratoires trouvé qui porte f_r, Q et détectivité dans la même ligne. 5 entrées. | libre accès |

Résultat négatif utile : aucune revue ne tabule ensemble fréquence, facteur de qualité et
détectivité entre laboratoires, et aucun jeu de données brut de magnétomètre ΔE n'existe
sur Zenodo, figshare, Dryad ou OSF.

---

## 7. Ce que la validation extérieure a changé dans le manuscrit (20/09/2026)

Le résultat de la §6 n'était pas un simple ajout : il a retourné la revendication
phare. Le manuscrit v3 dit maintenant autre chose, et de mieux étayé.

### 7.1 Le nouveau résultat central

La grandeur comparable entre laboratoires est la **perte magnéto-élastique par unité
de ramollissement**, Λ = [1/Q − 1/Q_sat]/x, dans laquelle la perte non magnétique
s'élimine. Sur trois plateformes, de 7,4 kHz à 246 MHz, **Λ ne croît que d'un facteur
12 quand la fréquence croît d'un facteur 33 000**, soit un exposant de **0,20 ± 0,05**.

La perte est donc quasi indépendante de la vitesse, ce qu'un mécanisme hystérétique de
parois donnerait, et ce que Nan et al. proposent d'ailleurs pour leurs propres
dispositifs. Cela **exclut à la fois** la forme de relaxation que nous supposions et la
forme en 1/f de l'éq. (4) de [1]. Script : `run_loss_scaling.py`, figure
`fig_loss_scaling.png`, sorties `loss_scaling.json`.

### 7.2 La loi corrigée, et ses chiffres

1/Q = 1/Q0 + Λ x, toujours deux constantes contre quatre dans [1] :

- ajustement conjoint **Q0 = 523 ± 16, Λ = (4,81 ± 0,20)·10⁻³ par %** (leave-one-out) ;
- R² de **0,67** et **0,37** sur les deux familles, contre 0,71 et 0,40 pour les quatre
  constantes de [1] : deux constantes font toujours le travail de quatre ;
- transfert entre familles **+0,29** et **+0,54**, contre −0,26 et −1,97 pour la forme
  à quatre constantes.

Les valeurs Q0 = 523 ± 64 et τ = 4,07 ± 1,41 ps du v2 **disparaissent**.

### 7.3 Les deux points extérieurs, et leur traçabilité

| Source | f_r | Q_sat | Q au point de fonctionnement | x | Λ |
|---|---|---|---|---|---|
| Nan 2013 [33] | 215 MHz | 1400 | 250 | 0,74 à 1,10 % | 3,0 à 4,4·10⁻³ |
| Durdaut 2020 [34] | 7,42 kHz | 1071 | 825 | 0,500 % | 0,56·10⁻³ |

Les Q de [33] sont **cités verbatim de son texte**. Le balayage de fréquence de [33],
ainsi que Q et le balayage de [34], ont été **digitalisés ici** depuis le contenu
vectoriel des figures publiées, étalonnés sur les graduations imprimées : résidus de
0,18 sur Q et 0,000 sur les Hz. C'est dit dans la déclaration de données.

⚠ Référence [34] : le préprint arXiv:2003.01085 que j'ai digitalisé porte un titre
**différent** de l'article JMEMS 29, 1347 (« Fundamental Noise Limits… ») que l'agent
proposait comme version publiée. Les 12 auteurs coïncident, mais Crossref ne connaît
pas le titre du préprint. Je cite donc **le préprint, qui est ce que j'ai réellement
lu**, avec son DOI DataCite vérifié 10.48550/arXiv.2003.01085.

### 7.4 Le caveat qui doit rester visible

**Tout le levier en fréquence tient au seul point bas.** En le retirant, l'exposant
remonte à 0,91, mais sur des points qui ne couvrent qu'un facteur 1,8 en fréquence,
contre un facteur 2,1 de dispersion entre nos deux familles : ce 0,91 ne veut rien
dire. Le manuscrit le dit, et nomme l'expérience qui trancherait, une famille entre
1 et 100 MHz, avec la prédiction chiffrée qui va avec.

### 7.5 État des livrables

Figures du texte principal, 5 comme avant : 1 réponse, 2 extrapolation,
**3 échelle de la perte (nouvelle)**, 4 détectivité, 5 bornes. Le panneau de transfert
des constantes de perte est passé en supplémentaire. Biblio 32 → **34**, toutes citées,
les deux nouvelles vérifiées par moi via Crossref et DataCite.

⚠ **LONGUEUR** : 7942 mots contre 6484 pour la version qui tenait en 8 pages, soit
environ 10 pages, donc ~350 $ de frais de page. Mon avis : l'article a maintenant cinq
résultats étayés au lieu de trois, dont un qui répond à la demande centrale de R1 avec
des mesures d'autres laboratoires. Payer deux pages me paraît le bon arbitrage, mais
c'est un arbitrage d'argent, donc il vous revient.
