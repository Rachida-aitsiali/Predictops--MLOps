# Ingestion des données — Maintenance prédictive industrielle

Ce dossier contient le pipeline d'ingestion des données brutes, réalisé avec **dlt**
(data load tool), pour le projet de maintenance prédictive industrielle.

## Structure

```
ingestion/
 ├── load_data.py       # script principal : exécute le pipeline dlt
 ├── sources.py          # définit la source de données (lecture du CSV)
 ├── raw_data/
 │    └── smart_manufacturing_data.csv   # dataset brut (Kaggle)
 └── README.md
```

## Source de données

Dataset : **Smart Manufacturing IoT-Cloud Monitoring Dataset** (Kaggle)
https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset

100 000 relevés de capteurs pour 50 machines industrielles, avec les colonnes :

| Colonne | Description |
|---|---|
| timestamp | Horodatage du relevé |
| machine_id | Identifiant de la machine |
| temperature | Température (°C) |
| vibration | Niveau de vibration |
| humidity | Humidité (%) |
| pressure | Pression |
| energy_consumption | Consommation d'énergie |
| machine_status | État de la machine (0 = Idle, 1 = Running, 2 = Failure) |
| anomaly_flag | Indicateur d'anomalie détectée (0/1) |
| predicted_remaining_life | Durée de vie restante estimée |
| failure_type | Type de panne (Normal, Vibration Issue, Overheating, Pressure Drop, Electrical Fault) |
| downtime_risk | Score de risque d'arrêt machine |
| maintenance_required | Cible : maintenance nécessaire (0 = Non, 1 = Oui) |

## Installation

```bash
pip install "dlt[duckdb]"
pip install pandas
```

## Exécution du pipeline

1. Placer le fichier `smart_manufacturing_data.csv` dans `ingestion/raw_data/`.
2. Lancer le script :

```bash
cd ingestion
python load_data.py
```

## Résultat attendu

Le script crée un fichier `maintenance_predictive_pipeline.duckdb` contenant
une table `raw.raw_sensor_data` avec les 100 000 lignes du fichier CSV brut,
prête à être utilisée par les étapes suivantes du pipeline (transformation,
nettoyage, modélisation).

## Vérification

```bash
python -c "
import duckdb
con = duckdb.connect('maintenance_predictive_pipeline.duckdb')
print(con.sql('SELECT COUNT(*) FROM raw.raw_sensor_data').df())
"
```

Résultat obtenu lors du test : **100 000 lignes** chargées avec succès, 0 échec.

## Difficultés rencontrées

- Le dataset ne contient pas de colonnes "vitesse" et "temps d'utilisation"
  explicitement demandées dans le cahier des charges initial. Elles ont été
  remplacées par des colonnes équivalentes déjà présentes dans le dataset :
  `energy_consumption` et `predicted_remaining_life`.
- Le fichier CSV étant volumineux (100 000 lignes), la lecture a été faite
  par lots (`chunksize`) dans `sources.py` pour rester efficace en mémoire.