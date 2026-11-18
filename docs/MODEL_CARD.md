# Fiche du modèle CVRIE — version améliorée

**Usage :** démonstration pédagogique locale, sans diagnostic individuel,
triage, dépistage ou décision de soin. Aucune garantie d'absence de cancer.

**Mise à jour :** 42 patients LIDC-IDRI supplémentaires sont disponibles localement,
mais les deux candidats entraînés sur la cohorte élargie n’ont pas été promus.
Le modèle actif décrit ici utilise toujours les 246 patients JSRT.
Voir [le bilan détaillé](EXTENDED_DATA.md).

Cible : nodule malin versus nodule bénin ou absence de nodule dans JSRT.
Pas de localisation, segmentation, estimation de stade ni détection de tous
les cancers. La classe négative ne signifie pas « personne saine ».

246 patients JSRT : 147 Train, 49 Validation, 50 Test, groupes conservés depuis
la première version. Images PNG 2048 × 2048, 8 bits. Les pixels identiques sont
contrôlés ; JPCNN014 reste exclu pour préserver la cohorte de référence.

Encodeur DenseNet-121 TorchXRayVision gelé, puis régression logistique C=0.001.
La vue complète est redimensionnée à 224 × 224. La configuration est sélectionnée
par AUC CV à 5 plis sur Train ; le seuil 0,508420 maximise l'exactitude équilibrée
sur Validation. Aucune métadonnée ni coordonnée de lésion n'est présentée au modèle.

Test exploratoire : AUC 0,748, exactitude 70 %, précision positive 59,3 %,
sensibilité 80 %, spécificité 63,3 %. 4 faux négatifs sur 20 cas malins et
11 faux positifs sur 30 négatifs. Le modèle précédent en produisait 9 et 10.
Les critères de promotion sont enregistrés avant l'évaluation Test.
Le test a cependant déjà été observé durant le développement : aucune validation
indépendante, prospective, clinique ou par sous-groupe.

Limites : petite collection historique, diagnostics issus d'un miroir public,
prévalence artificielle, résolution réduite à l'entrée du réseau, biais
d'acquisition/démographiques non mesurés. Pas d'audit patient par patient du
recoupement des bases de préentraînement, bien que JSRT ne figure pas parmi
leurs sources déclarées. Aucune calibration pour une population clinique.
Le score n'est pas une probabilité personnelle de cancer. Le seuil est instable
car estimé sur 49 cas. Les contrôles d'image ne reconnaissent pas l'anatomie.

Les images importées restent en mémoire du serveur local. Pas de transmission
à une API de prédiction ni de dossier patient persistant. Serveur lié à 127.0.0.1,
télémétrie Streamlit désactivée. La mise en ligne nécessiterait un travail distinct
sur l'accès, la protection des données et la validation clinique.

Les chiffres et versions sont dans `reports/supervised/metrics.json` ; le
protocole, les sources et les limites détaillées sont dans `IMPROVEMENTS.md`.
