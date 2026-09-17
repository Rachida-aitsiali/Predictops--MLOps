"""
train_model.py
--------------
Partie Maroua - Machine Learning pour la prediction des pannes.
Partie Rachida - Ajout du tracking MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DUCKDB_PATH = ROOT_DIR / "ingestion" / "maintenance_predictive_pipeline.duckdb"
CSV_FALLBACK_PATH = ROOT_DIR / "data" / "raw_data" / "smart_manufacturing_data.csv"
OUTPUT_DIR = ROOT_DIR / "modeling" / "outputs"
MODEL_DIR = ROOT_DIR / "modeling" / "models"
TARGET = "maintenance_required"
RANDOM_STATE = 42
THRESHOLDS = np.arange(0.05, 0.951, 0.01)
MAX_SELECTION_PRECISION = 0.95
SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
]

MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"
MLFLOW_REGISTERED_MODEL_NAME = "predictive_maintenance_model"


def load_ml_dataset() -> tuple[pd.DataFrame, str]:
    """Load ML data from DuckDB, with CSV fallback for standalone execution."""
    if DUCKDB_PATH.exists():
        try:
            with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
                tables = con.sql("SHOW TABLES").df()["name"].tolist()
                if "ml_features" in tables:
                    return con.sql("SELECT * FROM main.ml_features").df(), "duckdb:main.ml_features"
        except Exception as exc:
            print(f"DuckDB non disponible pour le ML, fallback CSV: {exc}")

    if not CSV_FALLBACK_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable: {CSV_FALLBACK_PATH}")

    return pd.read_csv(CSV_FALLBACK_PATH), "csv:data/raw_data/smart_manufacturing_data.csv"


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create past-only temporal features per machine."""
    if "machine_id" not in df.columns or "recorded_at" not in df.columns:
        return df

    df = df.sort_values(["machine_id", "recorded_at"]).copy()
    grouped = df.groupby("machine_id", sort=False)

    for column in [col for col in SENSOR_COLUMNS if col in df.columns]:
        previous = grouped[column].shift(1)
        rolling_5 = previous.groupby(df["machine_id"]).rolling(5, min_periods=1)
        rolling_15 = previous.groupby(df["machine_id"]).rolling(15, min_periods=1)

        df[f"{column}_lag_1"] = previous
        df[f"{column}_diff_1"] = df[column] - previous
        df[f"{column}_roll5_mean"] = rolling_5.mean().reset_index(level=0, drop=True)
        df[f"{column}_roll5_std"] = rolling_5.std().reset_index(level=0, drop=True)
        df[f"{column}_roll15_mean"] = rolling_15.mean().reset_index(level=0, drop=True)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for supervised classification."""
    df = df.copy()
    df.columns = [column.lower() for column in df.columns]

    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible absente: {TARGET}")

    if "timestamp" in df.columns and "recorded_at" not in df.columns:
        df = df.rename(columns={"timestamp": "recorded_at"})

    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")
        df = add_temporal_features(df)
        recorded_at = df["recorded_at"]
        df["recorded_hour"] = recorded_at.dt.hour
        df["recorded_dayofweek"] = recorded_at.dt.dayofweek
        df["recorded_month"] = recorded_at.dt.month
        df = df.drop(columns=["recorded_at"])

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    leakage_columns = [
        "anomaly_flag",
        "machine_status",
        "failure_type",
        "downtime_risk",
        "predicted_remaining_life",
    ]
    X = X.drop(columns=[column for column in leakage_columns if column in X.columns])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def get_model_params(classifier: object) -> dict:
    """Recupere les hyperparametres importants d'un modele pour MLflow."""
    params = classifier.get_params()
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool)) or v is None}


def predict_with_threshold(model: Pipeline, X: pd.DataFrame, threshold: float) -> np.ndarray:
    y_proba = model.predict_proba(X)[:, 1]
    return (y_proba >= threshold).astype(int)


def find_best_threshold(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"f1_score": -1.0}
    fallback_threshold = 0.5
    fallback_metrics = {"f1_score": -1.0}

    for threshold in THRESHOLDS:
        y_pred = predict_with_threshold(model, X_val, float(threshold))
        metrics = {
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_pred, zero_division=0),
        }
        if metrics["f1_score"] > fallback_metrics["f1_score"]:
            fallback_threshold = float(threshold)
            fallback_metrics = metrics
        if (
            metrics["precision"] <= MAX_SELECTION_PRECISION
            and metrics["f1_score"] > best_metrics["f1_score"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics["f1_score"] < 0:
        return fallback_threshold, fallback_metrics

    return best_threshold, best_metrics


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float]:
    y_pred = predict_with_threshold(model, X_test, threshold)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "threshold": threshold,
    }

    return metrics


def save_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> None:
    y_pred = predict_with_threshold(model, X_test, threshold)
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No maintenance", "Maintenance"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Matrice de confusion - seuil {threshold:.2f}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def save_feature_importance(model: Pipeline, X_train: pd.DataFrame) -> None:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = X_train.columns

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
    else:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title("Importance des variables - meilleur modele")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df, source = load_ml_dataset()
    X, y = prepare_features(df)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    results = []
    fitted_models: dict[str, Pipeline] = {}
    run_ids: dict[str, str] = {}

    for name, classifier in candidate_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(X_train)),
                    ("classifier", classifier),
                ]
            )
            pipeline.fit(X_train, y_train)
            threshold, validation_metrics = find_best_threshold(pipeline, X_val, y_val)
            test_metrics = evaluate_model(pipeline, X_test, y_test, threshold)

            mlflow.log_param("model_type", name)
            for param_name, param_value in get_model_params(classifier).items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_metric("accuracy", test_metrics["accuracy"])
            mlflow.log_metric("precision", test_metrics["precision"])
            mlflow.log_metric("recall", test_metrics["recall"])
            mlflow.log_metric("f1_score", test_metrics["f1_score"])
            mlflow.log_metric("roc_auc", test_metrics["roc_auc"])
            mlflow.log_metric("threshold", test_metrics["threshold"])
            mlflow.log_metric("validation_f1_score", validation_metrics["f1_score"])

            mlflow.sklearn.log_model(pipeline, "model", serialization_format="pickle")

            run_ids[name] = mlflow.active_run().info.run_id

            results.append(
                {
                    "model": name,
                    **test_metrics,
                    "validation_f1_score": validation_metrics["f1_score"],
                }
            )
            fitted_models[name] = pipeline

    comparison = pd.DataFrame(results).sort_values(
        by=["f1_score", "roc_auc", "recall"],
        ascending=False,
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_model_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    best_threshold = float(comparison.iloc[0]["threshold"])
    best_run_id = run_ids[best_model_name]

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "features": X.columns.tolist(),
            "target": TARGET,
        },
        MODEL_DIR / "best_model.joblib",
    )

    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MLFLOW_REGISTERED_MODEL_NAME)
    print(f"Modele enregistre dans le Registry: {MLFLOW_REGISTERED_MODEL_NAME} version {registered_model.version}")

    save_confusion_matrix(best_model, X_test, y_test, best_threshold)
    save_feature_importance(best_model, X_train)

    y_pred = predict_with_threshold(best_model, X_test, best_threshold)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics_payload = {
        "data_source": source,
        "rows": int(len(df)),
        "features_used": X.columns.tolist(),
        "target": TARGET,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "best_model": str(best_model_name),
        "best_threshold": best_threshold,
        "best_metrics": comparison.iloc[0].drop(labels=["model"]).to_dict(),
        "classification_report": report,
        "mlflow_run_id": best_run_id,
        "mlflow_registered_version": registered_model.version,
        "artifacts": {
            "model": str(MODEL_DIR / "best_model.joblib"),
            "metrics": str(OUTPUT_DIR / "metrics.json"),
            "comparison": str(OUTPUT_DIR / "model_comparison.csv"),
            "confusion_matrix": str(OUTPUT_DIR / "confusion_matrix.png"),
            "feature_importance": str(OUTPUT_DIR / "feature_importance.png"),
        },
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    print(json.dumps(metrics_payload["best_metrics"], indent=2))
    print(f"Meilleur modele: {best_model_name}")
    print(f"Modele sauvegarde: {MODEL_DIR / 'best_model.joblib'}")

    return metrics_payload


if __name__ == "__main__":
    main()