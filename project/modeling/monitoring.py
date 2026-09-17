"""
monitoring.py
--------------
Partie Rachida - Monitoring simple du modele et detection de derive des donnees.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from train_model import (
    RANDOM_STATE,
    TARGET,
    load_ml_dataset,
    prepare_features,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "modeling" / "outputs"
DRIFT_ZSCORE_THRESHOLD = 0.3  # seuil d'alerte en ecarts-types


def check_performance_drift() -> None:
    """Compare l'accuracy du dernier run avec les runs precedents."""
    print("=" * 60)
    print("1. SUIVI DE PERFORMANCE DANS LE TEMPS")
    print("=" * 60)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("predictive_maintenance")

    if experiment is None:
        print("Aucune experience trouvee. Lance d'abord train_model.py")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
    )

    if not runs:
        print("Aucun run trouve.")
        return

    print(f"\n{'Run':<25} {'Accuracy':<12} {'F1-score':<12}")
    print("-" * 50)

    accuracies = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "unknown")
        accuracy = run.data.metrics.get("accuracy")
        f1 = run.data.metrics.get("f1_score")
        if accuracy is not None:
            accuracies.append((run_name, accuracy))
            print(f"{run_name:<25} {accuracy:<12.4f} {f1:<12.4f}")

    if len(accuracies) >= 2:
        first_acc = accuracies[0][1]
        last_acc = accuracies[-1][1]
        variation = ((last_acc - first_acc) / first_acc) * 100

        print(f"\nPremiere accuracy enregistree : {first_acc:.4f}")
        print(f"Derniere accuracy enregistree : {last_acc:.4f}")
        print(f"Variation : {variation:+.2f}%")

        if variation < -5:
            print("⚠️  ALERTE : baisse de performance detectee (> 5%)")
        else:
            print("✅ Performance stable, pas d'alerte.")


def compute_drift_table(
    X_reference: pd.DataFrame,
    X_new: pd.DataFrame,
    numeric_columns: pd.Index,
) -> tuple[dict, list[str]]:
    """Calcule le z-score de derive pour chaque colonne numerique.

    Le z-score mesure l'ecart entre la moyenne des nouvelles donnees et la
    moyenne de reference, exprime en nombre d'ecarts-types de reference.
    Cette mesure reste stable meme quand la moyenne est proche de zero.
    """
    drift_report = {}
    alerts = []

    print(f"{'Variable':<30} {'Moyenne ref.':<15} {'Moyenne nouv.':<15} {'Ecart (z-score)':<15}")
    print("-" * 78)

    for column in numeric_columns:
        mean_ref = X_reference[column].mean()
        std_ref = X_reference[column].std()
        mean_new = X_new[column].mean()

        if std_ref == 0 or pd.isna(std_ref):
            continue

        z_score = abs(mean_new - mean_ref) / std_ref

        drift_report[column] = {
            "mean_reference": float(mean_ref),
            "mean_new": float(mean_new),
            "std_reference": float(std_ref),
            "z_score": float(z_score),
        }

        flag = ""
        if z_score > DRIFT_ZSCORE_THRESHOLD:
            flag = "⚠️  DERIVE"
            alerts.append(column)

        print(f"{column:<30} {mean_ref:<15.3f} {mean_new:<15.3f} {z_score:<15.3f} {flag}")

    return drift_report, alerts


def check_data_drift_stable_case() -> dict:
    """Cas 1 : compare le train aux donnees de test reelles (non modifiees).

    Note: en l'absence d'un nouveau jeu de donnees reel, on utilise le jeu de
    test (donnees jamais vues par le modele, meme dataset source) pour
    demontrer le mecanisme dans une situation "normale", sans changement
    reel attendu.
    """
    print("\n" + "=" * 60)
    print("2. CAS STABLE - Test avec les donnees reelles (pas de derive attendue)")
    print("=" * 60)

    df, _ = load_ml_dataset()
    X, y = prepare_features(df)

    X_train_val, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    numeric_columns = X_train_val.select_dtypes(include=["number"]).columns

    drift_report, alerts = compute_drift_table(X_train_val, X_test, numeric_columns)

    print(f"\nSeuil d'alerte : {DRIFT_ZSCORE_THRESHOLD} ecart(s)-type(s)")
    if alerts:
        print(f"⚠️  {len(alerts)} variable(s) avec derive detectee : {', '.join(alerts)}")
    else:
        print("✅ Aucune derive significative detectee (resultat attendu).")

    drift_report["alerts"] = alerts
    return drift_report, X_train_val, numeric_columns


def check_data_drift_simulated_case(
    X_train_val: pd.DataFrame,
    numeric_columns: pd.Index,
) -> dict:
    """Cas 2 : simule volontairement une derive pour valider le mecanisme d'alerte.

    On modifie artificiellement une copie des donnees (temperature et
    vibration augmentees) pour reproduire un scenario realiste de machine en
    debut de surchauffe. Cela permet de prouver que le systeme de detection
    reagit correctement lorsqu'une vraie derive se produit, et pas seulement
    lorsqu'il n'y a rien a detecter.
    """
    print("\n" + "=" * 60)
    print("3. CAS SIMULE - Donnees volontairement modifiees (derive attendue)")
    print("=" * 60)
    print("Scenario : temperature +30%, vibration +40% (ex. debut de surchauffe)\n")

    X_simulated = X_train_val.sample(frac=0.2, random_state=RANDOM_STATE).copy()

    if "temperature" in X_simulated.columns:
        X_simulated["temperature"] = X_simulated["temperature"] * 1.30
    if "vibration" in X_simulated.columns:
        X_simulated["vibration"] = X_simulated["vibration"] * 1.40

    drift_report, alerts = compute_drift_table(X_train_val, X_simulated, numeric_columns)

    print(f"\nSeuil d'alerte : {DRIFT_ZSCORE_THRESHOLD} ecart(s)-type(s)")
    if alerts:
        print(f"⚠️  {len(alerts)} variable(s) avec derive detectee : {', '.join(alerts)}")
        print("✅ Le mecanisme de detection fonctionne correctement : la derive")
        print("   simulee a bien ete identifiee.")
    else:
        print("❌ Aucune derive detectee alors qu'une derive etait attendue.")
        print("   Le seuil ou la methode devraient etre revus.")

    drift_report["alerts"] = alerts
    return drift_report


def main() -> None:
    check_performance_drift()

    stable_report, X_train_val, numeric_columns = check_data_drift_stable_case()
    simulated_report = check_data_drift_simulated_case(X_train_val, numeric_columns)

    full_report = {
        "stable_case": stable_report,
        "simulated_drift_case": simulated_report,
        "threshold_zscore": DRIFT_ZSCORE_THRESHOLD,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "drift_report.json").open("w", encoding="utf-8") as file:
        json.dump(full_report, file, indent=2)

    print(f"\nRapport complet sauvegarde : {OUTPUT_DIR / 'drift_report.json'}")


if __name__ == "__main__":
    main()