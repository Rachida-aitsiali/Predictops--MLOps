"""
definitions.py
---------------
Point d'entrée Dagster. Regroupe tous les assets et les ressources
(connexion CLI dbt) du pipeline "Maintenance Prédictive Industrielle".

Lancer l'interface avec :
    dagster dev -f orchestration/definitions.py
"""

import os
import shutil

from dagster import Definitions
from dagster_dbt import DbtCliResource

from .assets import (
    dbt_project,
    ml_ready_data,
    model_training,
    predictive_maintenance_dbt_assets,
    raw_sensor_data,
)

# Dossier contenant profiles.yml. Par défaut : ~/.dbt (peut être surchargé
# via la variable d'environnement DBT_PROFILES_DIR, utile sur Windows/CI).
DBT_PROFILES_DIR = os.environ.get(
    "DBT_PROFILES_DIR", os.path.expanduser("~/.dbt")
)

defs = Definitions(
    assets=[
        raw_sensor_data,
        predictive_maintenance_dbt_assets,
        ml_ready_data,
        model_training,
    ],
    resources={
        "dbt": DbtCliResource(
            project_dir=dbt_project,
            profiles_dir=DBT_PROFILES_DIR,
            dbt_executable=shutil.which("dbt") or "dbt",
        ),
    },
)
