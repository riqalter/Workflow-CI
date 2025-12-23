"""
modelling.py - ML Model Training with MLflow Logging for CI/CD

Script ini didesain untuk berjalan di GitHub Actions CI environment.
Menggunakan DagsHub untuk remote tracking dan menyimpan artifacts lokal.

Struktur Artefak yang Dihasilkan (lokal & DagsHub):
├── artifacts/
│   ├── model/
│   │   ├── MLmodel
│   │   ├── conda.yaml
│   │   ├── model.pkl
│   │   ├── python_env.yaml
│   │   └── requirements.txt
│   ├── estimator.html
│   ├── metric_info.json
│   ├── training_confusion_matrix.png
│   ├── classification_report.json
│   └── feature_importance.png

Cara menjalankan:
    mlflow run . --env-manager=local
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.utils import estimator_html_repr

import mlflow
import mlflow.sklearn

# =============================================================================
# KONFIGURASI MLFLOW - LOCAL TRACKING UNTUK CI
# =============================================================================

# Gunakan DagsHub atau folder lokal untuk tracking
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Nama experiment sesuai ketentuan
EXPERIMENT_NAME = "Latihan MSML"
mlflow.set_experiment(EXPERIMENT_NAME)

# Folder untuk menyimpan artifacts lokal (untuk GitHub Actions upload)
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =============================================================================
# LOAD DATASET
# =============================================================================

# Path ke dataset hasil preprocessing
DATA_PATH = "titanic_preprocessing/train_processed.csv"

print("=" * 60)
print("MODELLING.PY - ML TRAINING WITH MLFLOW (CI-READY)")
print("=" * 60)
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"Experiment: {EXPERIMENT_NAME}")
print("-" * 60)

# Load data
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Dataset shape: {df.shape}")

# =============================================================================
# PREPROCESSING - PISAHKAN FITUR DAN TARGET
# =============================================================================

print("\n[2] Memisahkan fitur (X) dan target (y)...")

# Target: Survived
y = df["Survived"]

# Fitur: semua kolom kecuali Survived
X = df.drop(columns=["Survived"])

print(f"    Fitur (X): {list(X.columns)}")
print("    Target (y): Survived")

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

print("\n[3] Split data train-test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")

# =============================================================================
# HYPERPARAMETER TUNING DENGAN GRIDSEARCHCV
# =============================================================================

print("\n[4] Hyperparameter tuning dengan GridSearchCV...")

# Definisi parameter grid untuk tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

print(f"    Parameter grid: {param_grid}")

# Base model
base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross validation
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

print("\n    Menjalankan GridSearchCV...")
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print("\n    [HASIL TUNING]")
print(f"    Best CV Score: {best_cv_score:.4f}")
print(f"    Best Parameters: {best_params}")

# =============================================================================
# MODEL EVALUATION DENGAN BEST MODEL
# =============================================================================

print("\n[5] Evaluasi model terbaik...")

# Model terbaik dari GridSearchCV
best_model = grid_search.best_estimator_

# Prediksi pada test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Hitung metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"    Accuracy:  {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1 Score:  {f1:.4f}")
print(f"    Log Loss:  {logloss:.4f}")
print(f"    ROC AUC:   {roc_auc:.4f}")

# =============================================================================
# MLFLOW LOGGING
# =============================================================================

print("\n[6] Logging ke MLflow...")

# Mulai MLflow run
with mlflow.start_run(run_name="RandomForest_Tuned") as run:
    # ---------------------------------
    # LOG PARAMETERS
    # ---------------------------------
    print("    Logging parameters...")
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("random_state", 42)

    # ---------------------------------
    # LOG METRICS
    # ---------------------------------
    print("    Logging metrics...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("best_cv_score", best_cv_score)

    # ---------------------------------
    # LOG MODEL - ke DagsHub dan simpan lokal
    # ---------------------------------
    print("    Logging model...")
    mlflow.sklearn.log_model(best_model, "model")
    
    # Simpan model lokal juga untuk upload ke GitHub Actions
    local_model_path = os.path.join(ARTIFACTS_DIR, "model")
    mlflow.sklearn.save_model(best_model, local_model_path)
    print(f"    Model saved locally to: {local_model_path}")

    # ---------------------------------
    # ARTEFAK 1: estimator.html
    # ---------------------------------
    print("    Creating estimator.html...")
    estimator_html = estimator_html_repr(best_model)
    html_path = os.path.join(ARTIFACTS_DIR, "estimator.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(estimator_html)
    mlflow.log_artifact(html_path)

    # ---------------------------------
    # ARTEFAK 2: training_confusion_matrix.png
    # ---------------------------------
    print("    Creating training_confusion_matrix.png...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )
    plt.title("Training Confusion Matrix - Random Forest (Tuned)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "training_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ---------------------------------
    # ARTEFAK 3: metric_info.json
    # ---------------------------------
    print("    Creating metric_info.json...")
    metric_info = {
        "model_name": "RandomForestClassifier",
        "tuning_method": "GridSearchCV",
        "cv_folds": 5,
        "best_params": best_params,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "log_loss": logloss,
            "roc_auc": roc_auc,
            "best_cv_score": best_cv_score,
        },
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(X.columns),
        },
    }
    mi_path = os.path.join(ARTIFACTS_DIR, "metric_info.json")
    with open(mi_path, "w") as f:
        json.dump(metric_info, f, indent=2)
    mlflow.log_artifact(mi_path)

    # ---------------------------------
    # ARTEFAK 4: classification_report.json
    # ---------------------------------
    print("    Creating classification_report.json...")
    class_report = classification_report(
        y_test, y_pred, target_names=["Not Survived", "Survived"], output_dict=True
    )
    cr_path = os.path.join(ARTIFACTS_DIR, "classification_report.json")
    with open(cr_path, "w") as f:
        json.dump(class_report, f, indent=2)
    mlflow.log_artifact(cr_path)

    # ---------------------------------
    # ARTEFAK 5: feature_importance.png
    # ---------------------------------
    print("    Creating feature_importance.png...")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color="steelblue",
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Random Forest (Tuned)")
    plt.tight_layout()
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    mlflow.log_artifact(fi_path)

    # Get run info
    run_id = run.info.run_id
    print(f"\n    Run ID: {run_id}")

print("\n" + "=" * 60)
print("TRAINING SELESAI!")
print("=" * 60)
print(f"\n>>> MLflow Tracking: {MLFLOW_TRACKING_URI}")
print(f">>> Experiment: {EXPERIMENT_NAME}")
print(f">>> Run ID: {run_id}")
print("\nArtefak yang disimpan:")
print("    - model/ (MLmodel, conda.yaml, model.pkl, dll)")
print("    - estimator.html")
print("    - metric_info.json")
print("    - training_confusion_matrix.png")
print("    - classification_report.json")
print("    - feature_importance.png")
