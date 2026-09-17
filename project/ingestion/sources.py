"""
sources.py
----------

Définit la source de données du pipeline d'ingestion pour le projet
"Maintenance Prédictive Industrielle".

Cette source lit le fichier CSV contenant les données des capteurs
des machines industrielles et expose ces données comme une ressource dlt.
"""

from pathlib import Path

import dlt
import pandas as pd

# ------------------------------------------------------------------
# Emplacement du dataset
# ------------------------------------------------------------------

RAW_DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "raw_data"
    / "smart_manufacturing_data.csv"
)


@dlt.resource(
    name="raw_sensor_data",
    write_disposition="replace",
)
def sensor_data_resource():
    """
    Lit le fichier CSV par morceaux (chunks) puis envoie les données
    à dlt sous forme de dictionnaires.

    Colonnes du dataset :

    - timestamp
    - machine_id
    - temperature
    - vibration
    - humidity
    - pressure
    - energy_consumption
    - machine_status
    - anomaly_flag
    - predicted_remaining_life
    - failure_type
    - downtime_risk
    - maintenance_required
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {RAW_DATA_PATH}"
        )

    # Lecture par lots de 5000 lignes
    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=5000):
        yield chunk.to_dict(orient="records")


@dlt.source
def manufacturing_source():
    """
    Source principale du projet.

    Elle regroupe toutes les ressources de données
    pouvant être utilisées par le pipeline.
    """
    return sensor_data_resource()