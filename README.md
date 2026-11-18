# CVRIE — radiographies thoraciques, prototype de recherche

Importer une radiographie PNG/JPEG dans une application française, obtenir une
classification expérimentale et consulter les résultats mesurés du modèle.
Traitement local, sans API distante et sans conservation des images importées.

**Ce prototype ne permet pas de déterminer de façon fiable si une personne a
un cancer.** Une prédiction négative ne l'exclut pas, une positive ne le confirme
pas. Aucune utilisation diagnostique, de dépistage ou de décision de soin.

## Extension publique et entraînements supplémentaires

La base disponible contient désormais **288 radiographies de 288 patients** :
246 JSRT + **42 nouveaux patients LIDC-IDRI** (+17 %). Sources publiques,
sans compte ni paiement. Voir [le protocole et les limites](docs/EXTENDED_DATA.md).

Deux expériences supplémentaires ont comparé **40 configurations / 600 ajustements
CV** : DenseNet 224 px, puis ResNet 512 px seul ou combiné à DenseNet.
Les nouvelles cohortes comptent 171 Train, 58 Validation et 59 Test.
Les poids des encodeurs restent gelés ; ce nombre désigne les ajustements des
classifieurs, pas des époques d'entraînement des réseaux.

**Aucun gain suffisamment régulier n'a été confirmé.** Les deux candidats sont
sauvegardés, mais le modèle précédent reste actif : AUC JSRT 0,748 et exactitude
70 % sur son test exploratoire de 50 images. La version 512 px obtient 66 %
sur ce même test, et manque les 3 cas malins du petit test LIDC. Il serait
incorrect de la présenter comme plus précise. Les scores par source et la
raison du maintien du modèle sont visibles dans l'application.

Pour reproduire l'extension après la préparation JSRT :

```bash
python -m src.supervised.extend_dataset
python -m src.supervised.train_extended
python -m src.supervised.train_extended_512
```

## Démarrage

Python 3.13 recommandé. Depuis la racine :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.supervised.dataset   # télécharge miniJSRT et fixe la séparation par patient
python -m src.supervised.improve   # extrait les caractéristiques et entraîne le modèle
python -m streamlit run app.py
```

Ouvrir http://127.0.0.1:8501. Les données et le modèle ne sont pas dans le dépôt :
les deux commandes d'entraînement les recréent en local. Ensuite, seule la
dernière commande est nécessaire. Le serveur écoute sur 127.0.0.1 uniquement. Arrêt : Ctrl+C.

Le premier entraînement télécharge miniJSRT pour reproduire la séparation
historique, l'archive JSRT haute résolution (environ 382 Mio) et les poids
DenseNet (environ 27 Mio). Les téléchargements suivants et les caractéristiques
sont réutilisés. Aucun paiement, compte ou clé API n'est requis actuellement.
Les données et poids sont ignorés par Git et ne sont pas distribués avec le code.

Pour reproduire seulement l'ancienne approche HOG (cela remplace le modèle actif) :

```bash
python -m src.supervised.dataset
python -m src.supervised.run_supervised
```

Pour revenir à la version améliorée, relancer `python -m src.supervised.improve`.

## Utilisation

Importer une radio thoracique **de face**, confirmer la nature de l'image puis
cliquer **Analyser la radiographie**. Des exemples du jeu de test sont disponibles.
Le résultat et l'évaluation sont exportables en JSON.

Formats : JPEG/PNG 8 bits en niveaux de gris, RGB sans couleur ou palette grise.
128 pixels minimum par côté, 25 mégapixels et 20 Mo maximum. DICOM, scanners,
IRM, photographies d'écran, couleur et transparence ne sont pas pris en charge.
Les contrôles de format, contraste et dimensions **ne reconnaissent pas
l'anatomie** : une image grise non médicale peut les passer.

## Données et modèle améliorés

246 patients de JSRT, 100 nodules malins, 54 bénins, 92 sans nodule.
Le doublon JPCNN014 identifié dans miniJSRT reste exclu, parmi les 247 cas
sources. Les nouveaux fichiers sont des PNG 2048 × 2048 sans contours ajoutés,
au lieu des miniatures 256 × 256. **Le nombre de patients n'a pas augmenté.**
Empreintes d'archives, jointures d'annotations, unicité des pixels et séparation
par patient sont vérifiées. La cible négative signifie « bénin ou sans nodule
selon JSRT », pas « personne sans cancer ».

DenseNet-121 préentraîné sur des radios thoraciques extrait les caractéristiques,
puis une régression logistique apprend les cibles JSRT. Six configurations sont
comparées par validation croisée à 5 plis sur les 147 patients Train. Le modèle
retenu utilise la vue complète redimensionnée à 224 × 224 et C=0.001. Le seuil
0,508420 est choisi sur les 49 patients Validation. Les 50 patients Test sont
exclus de ces ajustements. Les métadonnées et coordonnées des lésions ne sont
jamais utilisées comme caractéristiques. Le score n'est pas une probabilité
personnelle de cancer.

| Mesure sur les mêmes 50 images Test | Ancien modèle | Modèle amélioré |
| --- | ---: | ---: |
| AUC ROC | 0,667 | 0,748 |
| Exactitude | 62,0 % | 70,0 % |
| Précision positive | 52,4 % | 59,3 % |
| Sensibilité | 55,0 % | 80,0 % |
| Spécificité | 66,7 % | 63,3 % |
| Cas malins manqués | 9 / 20 | 4 / 20 |
| Fausses alertes | 10 / 30 | 11 / 30 |

La référence « toujours négatif » atteint 60 % d'exactitude. IC bootstrap 95 %
de l'AUC améliorée : [0,601 ; 0,893]. Le test a déjà été consulté pendant le
développement : comparaison exploratoire, **aucune validation indépendante
ou clinique**. Les fausses alertes augmentent légèrement. Sur Validation,
l'AUC ne passe que de 0,534 à 0,579 ; une nouvelle cohorte est indispensable
pour vérifier la fiabilité. Ne pas rechercher de nouvelles configurations
sur la base de ces 50 résultats Test.

[Protocole complet et comparaison](docs/IMPROVEMENTS.md) · [Fiche du modèle](docs/MODEL_CARD.md).

## Sources et attribution

- [JSRT original](http://db.jsrt.or.jp/eng.php) : Shiraishi J. et al., *Development of a digital image database for chest radiographs with and without a lung nodule*, AJR 174:71–74, 2000.
- [miniJSRT officiel](http://imgcom.jsrt.or.jp/minijsrtdb/) : archive Gender01 de la première version.
- [JSRT haute résolution et métadonnées — raddar/Kaggle](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt).
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) : encodeur et poids préentraînés. Voir aussi la [documentation](https://mlmed.org/torchxrayvision/models.html).
- [NCI : examens du diagnostic du cancer](https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis).

Le miroir Kaggle déclare une licence « Unknown ». Vérifier les droits et conditions
de chaque source avant toute redistribution ou exploitation externe. Les
annotations du miroir n'ont pas été vérifiées dossier par dossier contre les
comptes rendus cliniques. Le CSV étudiant et les travaux non supervisés du dépôt
ne servent pas au modèle d'image et sont conservés.

## Vérification et fichiers

```bash
python -m pytest -q
```

- `app.py` : interface locale et comparaison avant/après.
- `src/supervised/dataset.py` : acquisition initiale, dédoublonnage et séparation.
- `src/supervised/high_resolution.py` : qualité et remplacement par les images 2048.
- `src/supervised/deep_features.py` : encodeur gelé partagé par entraînement et inférence.
- `src/supervised/improve.py` : CV, sélection, seuil et promotion sur Validation.
- `src/supervised/inference.py` : vérification des poids et prédiction.
- `reports/supervised/metrics.json` : résultats actifs et provenance.
- `reports/supervised/baseline/` : résultats de référence conservés.
- `reports/supervised/selection_v2.json` : décision enregistrée avant l'évaluation Test.
- `reports/supervised/test_predictions.csv` : erreurs auditables par identifiant.
- `data/supervised/quality_2048.json` : audit des données.

Les tests d'intégration nécessitent données et modèle. Ne charger que les
artefacts Joblib produits localement : le format n'est pas sûr pour un modèle
fourni par un tiers. Le service ne télécharge pas de poids pendant une prédiction :
les poids manquants ou modifiés sont refusés au chargement du modèle.
