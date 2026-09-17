"""
assets.py
---------
Définition des assets Dagster qui orchestrent le pipeline complet du projet
"Maintenance Prédictive Industrielle" :

    dlt (ingestion, Imane) -> dbt (transformations, Sara) -> ML (Maroua)

Rôle : Amina — DataOps Engineer / Orchestration
"""

import json
import subprocess
import sys
from pathlib import Path

import dlt
import duckdb
from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

# ------------------------------------------------------------------
# Chemins du projet
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
INGESTION_DIR = ROOT_DIR / "ingestion"
DBT_PROJECT_DIR = ROOT_DIR / "predictive_maintenance"
DUCKDB_PATH = INGESTION_DIR / "maintenance_predictive_pipeline.duckdb"
MODEL_SCRIPT_PATH = ROOT_DIR / "modeling" / "train_model.py"
ML_METRICS_PATH = ROOT_DIR / "modeling" / "outputs" / "metrics.json"

# Projet dbt géré par Dagster (génère/rafraîchit le manifest.json au démarrage
# en dev grâce à prepare_if_dev())
dbt_project = DbtProject(project_dir=DBT_PROJECT_DIR)
dbt_project.prepare_if_dev()


# ------------------------------------------------------------------
# 1. ASSET D'INGESTION — étape réalisée par Imane (dlt), orchestrée ici
# ------------------------------------------------------------------
@asset(
    group_name="ingestion",
    description=(
        "Ingestion des données brutes des capteurs (CSV) via dlt, "
        "chargées dans le schéma 'raw' de DuckDB. Script source : Imane."
    ),
)
def raw_sensor_data(context: AssetExecutionContext) -> MaterializeResult:
    # Permet d'importer sources.py sans dupliquer le code d'Imane
    sys.path.insert(0, str(INGESTION_DIR))
    from sources import manufacturing_source  # noqa: E402

    pipeline = dlt.pipeline(
        pipeline_name="maintenance_predictive_pipeline",
        destination=dlt.destinations.duckdb(credentials=str(DUCKDB_PATH)),
        dataset_name="raw",
    )

    context.log.info("Démarrage de l'ingestion des données capteurs (dlt)...")
    load_info = pipeline.run(manufacturing_source())
    context.log.info(str(load_info))

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    row_count = con.sql("SELECT COUNT(*) FROM raw.raw_sensor_data").fetchone()[0]
    con.close()

    return MaterializeResult(
        metadata={
            "row_count": row_count,
            "load_info": MetadataValue.text(str(load_info)),
        }
    )


# ------------------------------------------------------------------
# 2. ASSETS DBT — étape réalisée par Sara (staging -> marts), orchestrée ici
#    Un asset Dagster est généré automatiquement par modèle dbt :
#    stg_machine_data, clean_machine_data, ml_features.
#    Le lien avec raw_sensor_data se fait automatiquement car le nom
#    de la source dbt (raw_sensor_data) correspond à l'asset ci-dessus.
# ------------------------------------------------------------------
@dbt_assets(manifest=dbt_project.manifest_path)
def predictive_maintenance_dbt_assets(
    context: AssetExecutionContext, dbt: DbtCliResource
):
    # "build" = run + test en une seule commande (transformations + qualité)
    yield from dbt.cli(["build"], context=context).stream()


# ------------------------------------------------------------------
# 3. ASSET DE VÉRIFICATION — les données sont prêtes pour le ML
# ------------------------------------------------------------------
@asset(
    deps=[AssetKey("ml_features")],
    group_name="ml",
    description="Vérifie que la table finale ml_features est prête à être utilisée pour l'entraînement du modèle.",
)
def ml_ready_data(context: AssetExecutionContext) -> MaterializeResult:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    row_count = con.sql("SELECT COUNT(*) FROM main.ml_features").fetchone()[0]
    target_distribution = con.sql(
        """
        SELECT maintenance_required, COUNT(*) AS total
        FROM main.ml_features
        GROUP BY maintenance_required
        ORDER BY maintenance_required
        """
    ).df()
    con.close()

    context.log.info(f"ml_features prête : {row_count} lignes.")

    return MaterializeResult(
        metadata={
            "row_count": row_count,
            "target_distribution": MetadataValue.md(
                target_distribution.to_markdown(index=False)
            ),
        }
    )


# ------------------------------------------------------------------
# 4. ASSET D'ENTRAÎNEMENT ML — étape réalisée par Maroua
#    Ce sera à Maroua de fournir son script (ex: modeling/train_model.py).
#    Amina n'a qu'à orchestrer son appel une fois le script livré :
#    -> décommenter les lignes ci-dessous et adapter le chemin.
# ------------------------------------------------------------------
@asset(
    deps=[ml_ready_data],
    group_name="ml",
    description="Déclenche l'entraînement du modèle ML (script de Maroua). À brancher dès que son script est prêt.",
)
def model_training(context: AssetExecutionContext) -> MaterializeResult:
    if not MODEL_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Script ML introuvable : {MODEL_SCRIPT_PATH}")

    context.log.info("Demarrage de l'entrainement ML de Maroua...")
    result = subprocess.run(
        [sys.executable, str(MODEL_SCRIPT_PATH)],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    context.log.info(result.stdout)
    if result.stderr:
        context.log.warning(result.stderr)

    with ML_METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    return MaterializeResult(
        metadata={
            "best_model": MetadataValue.text(metrics["best_model"]),
            "accuracy": float(metrics["best_metrics"]["accuracy"]),
            "precision": float(metrics["best_metrics"]["precision"]),
            "recall": float(metrics["best_metrics"]["recall"]),
            "f1_score": float(metrics["best_metrics"]["f1_score"]),
            "roc_auc": float(metrics["best_metrics"]["roc_auc"]),
            "model_path": MetadataValue.path(metrics["artifacts"]["model"]),
            "comparison_path": MetadataValue.path(metrics["artifacts"]["comparison"]),
            "confusion_matrix": MetadataValue.path(metrics["artifacts"]["confusion_matrix"]),
            "feature_importance": MetadataValue.path(metrics["artifacts"]["feature_importance"]),
        }
    )