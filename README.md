# CVRIE supervised package

## Structure
- `notebooks/01_supervised_models_comparison.ipynb` : notebook principal
- `src/supervised/` : fonctions réutilisables
- `models/` : emplacement du pipeline sauvegardé
- `bonus/app.py` : app Tkinter de démonstration

## Placement conseillé dans ton projet
Copie ce dossier de travail dans ton projet réel en gardant la logique suivante :

```text
G-AIA-400-MAR-4-1-cvrie-2/
├── data/
├── models/
├── notebooks/
├── src/
│   └── supervised/
└── bonus/
```

## Exécution du notebook
Le notebook suppose qu'il est placé dans `notebooks/` et que les modules Python sont dans `src/supervised/`.

## Entraîner et sauvegarder le meilleur modèle
Depuis la racine du projet :

```bash
python -m src.supervised.training
```

## Lancer l'app bonus
Depuis la racine du projet :

```bash
python bonus/app.py
```
