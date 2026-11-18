# Extension publique de la base — 10 octobre 2026

42 patients supplémentaires LIDC-IDRI ont été ajoutés aux 246 patients JSRT :
**288 patients, 288 radiographies de face**. Ce sont de nouveaux patients de la
collection, pas des augmentations artificielles ou des copies de JSRT.
L'absence de recoupement d'identité entre les collections n'est pas vérifiable.

## Sources et critères

- Collection : [LIDC-IDRI, TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/).
- Attribution : Armato SG et al., DOI 10.7937/K9/TCIA.2015.LO9QL9SX, CC BY 3.0.
- Annotations : fichier public `tcia-diagnosis-data-2012-04-20.xls`, référencé par TCIA.
- Images : API publique NBIA `getSeries` / `getImage`, modalité DX.
- Positif : diagnostic patient code 2 (cancer pulmonaire primitif), confirmé par
  biopsie ou résection (méthode 2 ou 3).
- Négatif : code 1 (bénin/non malin), méthode 1 (stabilité radiologique au moins
  deux ans), 2 ou 3. Les inconnus, métastases et progression seule sont exclus.
- 43 patients avaient un diagnostic admissible et une série DX. Un a été exclu,
  LIDC-IDRI-0187, sans projection frontale **originale** documentée.
- Les profils et les images dérivées/secondaires sont exclus. La projection est
  établie par ViewPosition PA/AP ou PatientOrientation L/F ou R/F si ViewPosition
  est absent. Une seule image originale est retenue par patient, ordre SOP UID.
- Contrôle visuel des 42 images retenues, sans modification des cibles.

Les annotations LIDC sont **au niveau du patient**, celles de JSRT concernent
un nodule. La lésion n'est pas nécessairement visible sur la radiographie et
la concordance temporelle avec le diagnostic n'est pas établie. Le mélange
est exploratoire ; il ne constitue pas une base validée pour diagnostiquer
le cancer à partir d'une radiographie.

Les DICOM restent locaux. Conversion déterministe : rescale DICOM, percentiles
0,5/99,5, inversion MONOCHROME1, PNG 8 bits, dimension maximale 2048 pixels.
Les fenêtres DICOM fournies ne sont pas utilisées : certains fichiers DX
contiennent des valeurs de fenêtre de type scanner. Les empreintes des
archives, images, annotations et manifeste sont conservées dans les rapports.

## Séparation avant entraînement

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| JSRT | 147 | 49 | 50 |
| LIDC-IDRI | 24 | 9 | 9 |
| Total | 171 | 58 | 59 |

LIDC contient 15 diagnostics malins et 27 bénins. Ses jeux Validation et Test
comprennent chacun 3 cas malins et 6 bénins. Les groupes JSRT sont inchangés.
Aucun patient ni fichier exactement identique ne traverse les groupes.
Les quasi-doublons transformés ne sont pas détectés automatiquement.

## Reproduction

Après l'installation décrite dans le README et la préparation JSRT :

```bash
python -m src.supervised.extend_dataset
python -m src.supervised.train_extended
python -m src.supervised.train_extended_512
python -m pytest -q
```

Les téléchargements, caractéristiques et modèles sont conservés localement.
Chaque entraînement compare 20 configurations (régressions logistiques et SVM
RBF), avec 5 plis répétés 3 fois : 300 ajustements de validation croisée.
La stratification croise source et classe. Chaque pli ajuste son propre scaler
et son classifieur. Le score de sélection est la moyenne des AUC calculées
séparément par source, pour ne pas récompenser simplement la reconnaissance
visuelle d'une collection. Les encodeurs restent gelés : ces ajustements ne
sont pas des époques de fine-tuning du réseau.

Le seuil maximise l'exactitude équilibrée sur Validation. La promotion exige
une hausse de l'AUC et de l'exactitude équilibrée moyennes par source sur
Validation, sans baisse d'AUC sur JSRT. Cette décision est écrite **avant** le
calcul Test. Le test JSRT était déjà connu. Après l'expérience v3, le petit
test LIDC est lui aussi consulté : v4 fournit une comparaison exploratoire,
pas une nouvelle validation indépendante. Aucun score clinique cible n'est garanti.

Les protocoles, sélections et scores complets sont dans `reports/supervised/` :
`experiment_v3/v4.json`, `selection_v3/v4.json`, `candidate_v3/v4.json`,
`test_predictions_v3/v4.csv`, `split_manifest_v3/v4.csv`, `data_quality_v3/v4.json`.
Les modèles candidats sont sauvegardés même lorsqu'ils ne sont pas promus.

## Résultat des deux expériences

Aucun candidat n'a satisfait les critères préétablis de promotion. Le modèle
actif reste donc la version précédente, entraînée sur JSRT. Les 42 nouveaux
patients sont réellement téléchargés, annotés, répartis et utilisés dans les
deux entraînements candidats ; ils ne sont pas utilisés par le modèle actif.

| Modèle | AUC JSRT Test | Exactitude JSRT | AUC LIDC Test | Exactitude LIDC | Malins LIDC manqués |
|---|---:|---:|---:|---:|---:|
| Actif antérieur | 0,748 | 70 % | 0,722 | 66,7 % | 1/3 |
| Candidat DenseNet élargi | 0,733 | 60 % | 0,500 | 66,7 % | 1/3 |
| Candidat fusion ResNet512 + DenseNet | 0,740 | 66 % | 0,500 | 44,4 % | 3/3 |

Le meilleur candidat 512 px obtient une AUC CV moyenne par source de 0,746.
Sur Validation, son AUC JSRT monte à 0,669, mais son AUC LIDC tombe à 0,444 :
la progression n'est pas cohérente entre les sources. La décision de ne pas le
promouvoir a été prise sur Validation, avant lecture de son Test. Il n'y a
pas de gain de précision démontré. Le faible effectif LIDC rend les estimations
très instables et les annotations patient/nodule ne sont pas équivalentes.
Augmenter encore le temps d'entraînement ne corrige pas ces limites.
