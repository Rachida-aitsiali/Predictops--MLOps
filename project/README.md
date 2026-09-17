# Orchestration DataOps — Amina (Chapitre 5)

Ce projet orchestre avec **Dagster** le pipeline complet :

```
dlt (ingestion, Imane)  →  dbt (staging + marts, Sara)  →  ML (Maroua)
   raw_sensor_data          stg_machine_data                model_training
                             clean_machine_data
                             ml_features            →  ml_ready_data
```

Le graphe a été **testé de bout en bout** (ingestion + 21 vérifications dbt +
contrôle de la table finale) avant de te l'envoyer : tout fonctionne.

## 1. Installation (Windows)

Ouvre un terminal (PowerShell ou CMD) dans le dossier du projet, puis :

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configurer le profil dbt

Dagster a besoin de savoir où se trouve `profiles.yml`. Deux options :

**Option A — dossier par défaut** (`%USERPROFILE%\.dbt\profiles.yml`) :

Crée le fichier `C:\Users\<TonNom>\.dbt\profiles.yml` avec ce contenu :

```yaml
predictive_maintenance:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "CHEMIN_COMPLET_VERS\\ingestion\\maintenance_predictive_pipeline.duckdb"
      threads: 4
```

Remplace `CHEMIN_COMPLET_VERS` par le chemin absolu du dossier du projet sur
ton PC (ex : `C:\Users\Amina\Desktop\orchestration_project`).

**Option B — variable d'environnement** (si tu préfères garder profiles.yml
dans le projet) :

```powershell
set DBT_PROFILES_DIR=CHEMIN_VERS_UN_DOSSIER_CONTENANT_profiles.yml
```

## 3. Lancer l'interface Dagster

```powershell
dagster dev -f orchestration/definitions.py
```

Ouvre ensuite **http://localhost:3000**. Tu y verras :
- Le graphe complet des assets (onglet **Assets** → **View global asset lineage**)
- Un bouton **Materialize all** pour lancer tout le pipeline en un clic
- Les logs détaillés de chaque étape (onglet **Runs**)

C'est exactement ce qu'il te faut pour tes captures d'écran (interface,
graphe, exécution réussie, logs).

## 4. Structure du projet

```
orchestration_project/
├── orchestration/
│   ├── assets.py           # les 6 assets du pipeline (voir ci-dessous)
│   └── definitions.py       # point d'entrée Dagster
├── ingestion/                # script dlt d'Imane (inchangé)
│   ├── load_data.py
│   └── sources.py
├── predictive_maintenance/   # projet dbt de Sara (inchangé)
│   └── models/...
├── data/raw_data/             # dataset CSV source
├── requirements.txt
└── README.md
```

## 5. Les 6 assets orchestrés

| Asset | Étape | Rôle |
|---|---|---|
| `raw_sensor_data` | Ingestion dlt | Imane |
| `stg_machine_data` | dbt staging | Sara |
| `clean_machine_data` | dbt marts (nettoyage) | Sara |
| `ml_features` | dbt marts (table finale) | Sara |
| `ml_ready_data` | Vérification (lignes, distribution cible) | Amina |
| `model_training` | Entrainement ML et sauvegarde du modele | Maroua |

Les dépendances entre `raw_sensor_data` et les modèles dbt sont **automatiques** :
Dagster relie la source dbt `raw_sensor_data` (déclarée dans
`models/staging/sources.yml`) à l'asset du même nom.

## 6. Partie ML de Maroua

Le script `modeling/train_model.py` est maintenant branche dans l'asset
Dagster `model_training`. Il compare plusieurs modeles, selectionne le
meilleur, sauvegarde `modeling/models/best_model.joblib` et genere les
metriques/graphes dans `modeling/outputs/`.
## 7. Captures à faire pour ton rapport (Chapitre 5)

- Interface Dagster (page d'accueil avec le graphe)
- Graphe complet du pipeline (lineage view)
- Une exécution réussie (`Materialize all` → tous les carrés verts)
- Les logs d'une étape (ex. `raw_sensor_data` ou `predictive_maintenance_dbt_assets`)
- Si vous faites du planning Agile : captures des sprints (Trello/Notion) avec Rahil
