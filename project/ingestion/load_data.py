"""
load_data.py
-------------
Script principal d'ingestion. Crée un pipeline dlt qui charge les données
brutes du fichier CSV (via sources.py) dans une couche "raw" en base DuckDB.

Usage :
    python load_data.py
"""

import dlt
from sources import manufacturing_source


def main():
    # Création du pipeline dlt
    # - pipeline_name : nom du pipeline (sert aussi pour les logs/état dlt)
    # - destination    : DuckDB, un fichier local léger, parfait pour la couche raw
    # - dataset_name   : nom du schéma/dataset dans DuckDB -> "raw" (couche raw)
    pipeline = dlt.pipeline(
        pipeline_name="maintenance_predictive_pipeline",
        destination="duckdb",
        dataset_name="raw",
    )

    print("Démarrage de l'ingestion des données capteurs...")

    # Exécution du pipeline : lit la source et charge dans DuckDB
    load_info = pipeline.run(manufacturing_source())

    print("\nIngestion terminée avec succès.")
    print(load_info)

    return load_info


if __name__ == "__main__":
    main()