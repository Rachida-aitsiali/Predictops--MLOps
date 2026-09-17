# Partie 6 - Modelisation Machine Learning

Ce dossier contient le travail ML de Maroua pour predire la variable
`maintenance_required`.

## Lancer l'entrainement

Depuis la racine du projet:

```powershell
python modeling/train_model.py
```

Le script compare quatre modeles:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Sorties generees

- `models/best_model.joblib`: meilleur modele sauvegarde.
- `outputs/metrics.json`: metriques principales.
- `outputs/model_comparison.csv`: tableau comparatif des modeles.
- `outputs/confusion_matrix.png`: matrice de confusion.
- `outputs/feature_importance.png`: importance des variables.
- `outputs/feature_importance.csv`: valeurs d'importance.

## Notes de preparation

Le script utilise d'abord la table `main.ml_features` dans DuckDB. Si elle
n'existe pas encore, il lit le fichier CSV brut pour permettre un test
independant. Les colonnes `anomaly_flag`, `machine_status`, `failure_type`,
`downtime_risk` et `predicted_remaining_life` sont retirees de l'entrainement
car elles creent une fuite d'information ou donnent une evaluation trop
optimiste. Le seuil de decision est optimise sur un ensemble de validation avec
une contrainte pour eviter une precision artificiellement parfaite.

Le script ajoute aussi des variables temporelles par machine (`lag`, `diff`,
`rolling mean`, `rolling std`) afin d'exploiter l'evolution recente des capteurs
sans utiliser d'information future.
