# Chapitre 6 - Modelisation Machine Learning

## 6.1 Objectif du modele ML

L'objectif de cette partie est de construire un modele de Machine Learning
capable de predire si une machine necessite une maintenance
(`maintenance_required = 1`) a partir des donnees capteurs et des informations
machine preparees par le pipeline DataOps.

Cette etape correspond au coeur intelligent du projet: transformer les donnees
nettoyees en un modele exploitable pour anticiper les pannes et aider les
equipes de maintenance a intervenir avant l'arret de la machine.

## 6.2 Preparation des donnees

Les donnees utilisees proviennent en priorite de la table DuckDB
`main.ml_features`, produite apres les transformations dbt. Le script peut
aussi lire le fichier CSV brut si la table DuckDB n'est pas encore disponible,
ce qui permet de tester la partie ML de maniere independante.

Les principales etapes de preparation sont:

- separation de la variable cible `maintenance_required`;
- conversion de la date en variables temporelles `recorded_hour` et
  `recorded_dayofweek`;
- encodage des variables categorielles avec One-Hot Encoding;
- normalisation des variables numeriques;
- imputation des valeurs manquantes;
- suppression des colonnes provoquant une fuite d'information ou un resultat
  trop optimiste: `anomaly_flag`, `machine_status`, `failure_type`,
  `downtime_risk` et `predicted_remaining_life`. Ces variables sont trop
  proches de la cible ou correspondent deja a des diagnostics/scores apres
  coup;
- creation de variables temporelles par machine: valeurs precedentes, ecarts
  avec la mesure precedente, moyennes glissantes et ecarts-types glissants.
  Ces variables utilisent uniquement les mesures passees pour eviter la fuite
  d'information.

## 6.3 Feature engineering

Les variables utilisees par le modele sont:

- `machine_id`;
- `temperature`;
- `vibration`;
- `humidity`;
- `pressure`;
- `energy_consumption`;
- `recorded_hour`;
- `recorded_dayofweek`.
- variables temporelles derivees: `lag_1`, `diff_1`, `roll5_mean`,
  `roll5_std`, `roll15_mean` pour les principaux capteurs.

Les variables categorielles sont transformees automatiquement, ce qui permet
aux modeles scikit-learn de les exploiter correctement.

## 6.4 Separation train/test

Le dataset contient 99 963 lignes apres nettoyage. La separation utilisee est:

- 60 % pour l'entrainement: 59 977 lignes;
- 20 % pour la validation: 19 993 lignes;
- 20 % pour le test final: 19 993 lignes.

La validation sert a choisir le meilleur seuil de decision. Le test final reste
separe afin d'obtenir une evaluation plus fiable. La separation est stratifiee
afin de conserver la meme proportion de machines necessitant une maintenance.

## 6.5 Modeles testes

Quatre modeles ont ete testes:

- Logistic Regression;
- Decision Tree;
- Random Forest;
- Gradient Boosting.

Chaque modele est integre dans un pipeline scikit-learn contenant le
preprocessing et le classifieur.

## 6.6 Metriques d'evaluation

Les metriques utilisees sont:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- matrice de confusion.

Le meilleur modele obtenu apres suppression des variables de fuite, ajout des
features temporelles et optimisation du seuil est `gradient_boosting`.

| Metrique | Valeur |
|---|---:|
| Accuracy | 0.8909 |
| Precision | 0.9531 |
| Recall | 0.4693 |
| F1-score | 0.6289 |
| ROC-AUC | 0.7369 |
| Seuil de decision | 0.18 |

## 6.7 Comparaison des modeles

| Modele | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8909 | 0.9531 | 0.4693 | 0.6289 | 0.7369 |
| Decision Tree | 0.8895 | 0.9404 | 0.4688 | 0.6257 | 0.7362 |
| Random Forest | 0.8881 | 0.9250 | 0.4700 | 0.6233 | 0.7408 |
| Logistic Regression | 0.7972 | 0.4844 | 0.4601 | 0.4719 | 0.6992 |

Gradient Boosting a ete retenu car il obtient le meilleur compromis global sur
le F1-score apres retrait des variables de fuite et ajout des variables
temporelles. Le seuil de decision final est fixe a 0.18, choisi sur l'ensemble
de validation avec une contrainte pour eviter un modele trop conservateur
donnant une precision artificiellement parfaite.

## 6.8 Choix du modele final

Le modele final est sauvegarde au format `joblib` dans:

```text
modeling/models/best_model.joblib
```

Les artefacts generes sont:

- `modeling/outputs/metrics.json`;
- `modeling/outputs/model_comparison.csv`;
- `modeling/outputs/confusion_matrix.png`;
- `modeling/outputs/feature_importance.png`;
- `modeling/models/best_model.joblib`.

## 6.9 Interpretation des resultats

Les premiers essais donnaient des scores presque parfaits, notamment une
precision de 1.0000. Apres analyse, ces resultats etaient dus a une fuite
d'information: certaines variables comme `anomaly_flag`, `machine_status` et
`failure_type` etaient trop directement liees a la cible
`maintenance_required`.

Apres suppression de ces variables, les resultats sont plus realistes. Pour
ameliorer la detection des cas positifs sans obtenir une precision parfaite
suspecte, le seuil de decision a ete optimise sur un ensemble de validation.
Le modele final atteint une precision de 0.9531, un recall de 0.4693 et un
F1-score de 0.6289. L'ajout des variables temporelles ameliore legerement le
modele par rapport a la version basee uniquement sur les mesures instantanees.

La precision reste elevee, ce qui signifie que les alertes de maintenance
declenchees sont generalement fiables. En revanche, le recall montre que le
modele ne detecte pas encore tous les cas positifs. Dans un contexte
industriel, cette limite est importante: pour aller plus loin, il faudrait
enrichir les donnees avec plus d'historique machine, des fenetres temporelles,
ou des variables de maintenance reellement disponibles avant la panne.

## Captures a fournir

- Capture de l'execution de `python modeling/train_model.py`.
- Capture du fichier `modeling/outputs/model_comparison.csv`.
- Capture du graphique `modeling/outputs/confusion_matrix.png`.
- Capture du graphique `modeling/outputs/feature_importance.png`.
- Capture du modele sauvegarde `modeling/models/best_model.joblib`.
- Capture de l'asset Dagster `model_training` apres materialisation si vous
  lancez le pipeline complet.
