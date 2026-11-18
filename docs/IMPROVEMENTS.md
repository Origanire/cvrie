# Amélioration initiale et extension du 10 octobre 2026

Une seconde étape a ajouté 42 patients publics LIDC-IDRI et entraîné 40 configurations
(600 ajustements CV). Les candidats n’ont pas satisfait les critères de promotion :
la version décrite ci-dessous reste active. [Bilan de l’extension](EXTENDED_DATA.md).

La version active utilise un encodeur DenseNet-121 préentraîné sur des
radiographies thoraciques (TorchXRayVision `densenet121-res224-all`), gelé,
puis une régression logistique régularisée (`C=0.001`).

## Données

Les 246 patients conservés sont les mêmes que dans la version initiale.
Leurs images passent de miniJSRT 256 × 256 à JSRT 2048 × 2048 en 8 bits,
issues du miroir public raddar sur Kaggle. Aucun nouveau patient n'a été ajouté.
Les fichiers en palette sont convertis en niveaux de gris seulement après
vérification de l'absence de couleur. L'archive et les annotations sont vérifiées
par SHA-256. Les pixels décodés sont contrôlés pour détecter les doublons exacts.
JPCNN014 reste exclu pour préserver la cohorte de comparaison.
Les quasi-doublons transformés ne sont pas audités automatiquement.

`data/supervised/manifest_2048.csv` conserve les identifiants, cibles et groupes
de la première version : 147 Train, 49 Validation, 50 Test.
Le passage à la haute résolution améliore la source disponible, mais ne
constitue pas une validation sur une nouvelle population. Le réseau retenu
ramène la vue complète à 224 × 224 : il ne traite pas les 2048 pixels nativement.

## Protocole

Six configurations fixées avant l'évaluation : vue globale ou vue globale +
deux régions pulmonaires fixes ; C dans {0.001, 0.01, 0.1}.
Les régions sont indépendantes des annotations de nodules.
Sélection par AUC moyenne de validation croisée stratifiée à 5 plis sur Train.
Le scaler et la régression sont réajustés dans chaque pli. L'encodeur reste gelé.
Aucune coordonnée de lésion, nom de fichier, âge ou sexe n'entre dans le modèle.

La configuration globale C=0.001 obtient une AUC CV de 0,710 (écart-type 0,122).
Le seuil 0,508420 est choisi pour maximiser l'exactitude équilibrée sur Validation.
Les critères de remplacement (AUC et exactitude équilibrée de validation
supérieures à la référence) sont enregistrés dans `selection_v2.json` avant
le calcul des scores de test. La décision n'utilise pas les scores du test.
L'AUC de validation passe de 0,534 à 0,579 ; elle reste faible et incertaine.

## Résultats exploratoires sur les mêmes 50 patients

| Mesure | Avant | Maintenant |
| --- | ---: | ---: |
| AUC ROC | 0,667 | 0,748 |
| Exactitude | 62,0 % | 70,0 % |
| Précision positive | 52,4 % | 59,3 % |
| Sensibilité | 55,0 % | 80,0 % |
| Spécificité | 66,7 % | 63,3 % |
| Cas malins manqués | 9 / 20 | 4 / 20 |
| Fausses alertes | 10 / 30 | 11 / 30 |

IC bootstrap 95 % de la nouvelle AUC : [0,601 ; 0,893]. Un tel effectif ne
permet pas d'affirmer un gain clinique établi. Le seuil a changé : les décisions
comparent à la fois une nouvelle représentation et un nouveau seuil. L'AUC
ne dépend pas du seuil. Les métriques au seuil 0,50 restent disponibles dans
le JSON sous `test_fixed_threshold_05`.

Le test a déjà été consulté dans le développement précédent ; il n'est donc
plus un test final vierge. Les résultats restent exploratoires, et ne justifient
pas un diagnostic, un dépistage ou une conclusion rassurante. Les faux positifs
augmentent légèrement malgré la réduction des cas manqués.

## Sources et limites du préentraînement

[TorchXRayVision](https://github.com/mlmed/torchxrayvision) et sa
[documentation des modèles](https://mlmed.org/torchxrayvision/models.html)
documentent la réutilisation des caractéristiques et les sources des poids.
Les sources déclarées sont NIH, PadChest, CheXpert, MIMIC, Google, OpenI et RSNA.
JSRT n'y est pas déclaré ; aucun audit patient par patient du préentraînement
n'a été réalisé ici. Des sorties « nodule » ou « masse » ne sont pas utilisées
comme étiquettes « cancer ». La cible malin/bénin vient exclusivement des
annotations JSRT, avec leurs limites de provenance communautaire.

Il reste nécessaire d'obtenir une cohorte externe de radiographies avec
confirmation des diagnostics, identifiants patients et autorisation d'utilisation.
Des images de pneumonie, de scanner ou de microscopie ne peuvent pas être
ajoutées comme s'il s'agissait de radios avec une vérité cancer identique.
