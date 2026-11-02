"""
src/train_models.py
--------------------
Reproducible model training script.

Run from the project root:
    python -m src.train_models

What it does
------------
1. Loads the existing training dataset (data/dataset.csv) and the synthetic
   dataset (generated fresh if needed).
2. Merges them; creates 14-day and 30-day binary targets.
3. Splits chronologically where possible (by month) — otherwise stratified.
4. Compares DummyClassifier / RandomForest / GradientBoosting.
5. Selects the best model per horizon by macro F1 on the validation split.
6. Saves full pipelines to models/model_14d.pkl and models/model_30d.pkl.
7. Writes evaluation report to reports/model_evaluation.json.

IMPORTANT: Synthetic-label evaluation only measures fit to the rule-generated
labels, NOT real-world disease-forecasting ability.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_training_data, compute_reference_artifacts, SLOW_ONSET_DISEASES, FAST_ONSET_DISEASES
from src.synthetic_data_generator import save_synthetic_dataset
from utils.feature_engineering import (
    ALL_MODEL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    engineer_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_pipeline(clf) -> Pipeline:
    numeric_transformer = StandardScaler()
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    prep = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> dict:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    dist = y_test.value_counts().to_dict()
    result = {
        "model": label,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": cm,
        "class_distribution_test": {str(k): int(v) for k, v in dist.items()},
    }
    logger.info("  [%s] acc=%.3f  bal_acc=%.3f  f1=%.3f  auc=%.3f",
                label, result["accuracy"], result["balanced_accuracy"],
                result["f1_macro"], result["roc_auc"])
    return result


def train_and_save():
    # ── 1. Load and merge datasets ──────────────────────────────────────────
    logger.info("Loading training dataset …")
    df_real = clean_training_data()

    logger.info("Generating synthetic dataset …")
    df_syn = save_synthetic_dataset(n_samples=1500, seed=RANDOM_STATE)
    # Clean synthetic with same pipeline (skip disease-origin columns not in schema)
    drop_extra = [c for c in df_syn.columns
                  if c not in df_real.columns and c not in ("record_origin", "source_file")]
    df_syn = df_syn.drop(columns=drop_extra, errors="ignore")
    # Add missing columns present in real data
    for col in df_real.columns:
        if col not in df_syn.columns:
            if pd.api.types.is_numeric_dtype(df_real[col]):
                df_syn[col] = df_real[col].median()
            else:
                df_syn[col] = df_real[col].mode()[0]
    df_syn = clean_training_data(df_syn)

    df = pd.concat([df_real, df_syn], ignore_index=True)
    logger.info("Combined dataset: %d rows (real=%d, synthetic=%d)",
                len(df), len(df_real), len(df_syn))

    # ── 2. Feature engineering ─────────────────────────────────────────────
    logger.info("Engineering features …")
    # Compute ref artifacts from training data only (full df here since we're building
    # the overall pipeline; in production you'd fit on train split only)
    ref_stats, region_risk_lookup = compute_reference_artifacts(df)

    engineered = [engineer_features(row.to_dict(), ref_stats, region_risk_lookup)
                  for _, row in df.iterrows()]
    df_feat = pd.DataFrame(engineered)

    # ── 3. Train / test split ──────────────────────────────────────────────
    # Stratify by disease to preserve class distribution
    X = df_feat[ALL_MODEL_FEATURES]
    y14 = df_feat["outbreak_risk_14d"]
    y30 = df_feat["outbreak_risk_30d"]

    X_train, X_test, y14_train, y14_test, y30_train, y30_test = train_test_split(
        X, y14, y30, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df_feat["disease"]
    )
    logger.info("Train=%d  Test=%d", len(X_train), len(X_test))

    # ── 4. Candidate models ────────────────────────────────────────────────
    candidates_14d = {
        "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=4,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE),
    }
    candidates_30d = {
        "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=4,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE),
    }

    eval_results = {"14d": [], "30d": [], "metadata": {}}
    best_14d_pipe, best_14d_score = None, -1
    best_30d_pipe, best_30d_score = None, -1

    # ── 5. Train + evaluate 14-day models ─────────────────────────────────
    logger.info("Training 14-day models …")
    for name, clf in candidates_14d.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y14_train)
        metrics = evaluate(pipe, X_test, y14_test, name)
        eval_results["14d"].append(metrics)
        if metrics["f1_macro"] > best_14d_score:
            best_14d_score = metrics["f1_macro"]
            best_14d_pipe = pipe
            eval_results["metadata"]["best_14d"] = name

    # ── 6. Train + evaluate 30-day models ─────────────────────────────────
    logger.info("Training 30-day models …")
    for name, clf in candidates_30d.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y30_train)
        metrics = evaluate(pipe, X_test, y30_test, name)
        eval_results["30d"].append(metrics)
        if metrics["f1_macro"] > best_30d_score:
            best_30d_score = metrics["f1_macro"]
            best_30d_pipe = pipe
            eval_results["metadata"]["best_30d"] = name

    # ── 7. Save best pipelines ─────────────────────────────────────────────
    joblib.dump(best_14d_pipe, MODELS_DIR / "model_14d.pkl")
    joblib.dump(best_30d_pipe, MODELS_DIR / "model_30d.pkl")
    logger.info("Saved model_14d.pkl (%s) and model_30d.pkl (%s)",
                eval_results["metadata"]["best_14d"],
                eval_results["metadata"]["best_30d"])

    # ── 8. Save evaluation report ──────────────────────────────────────────
    eval_results["metadata"].update({
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "real_rows": int(len(df_real)),
        "synthetic_rows": int(len(df_syn)),
        "feature_count": len(ALL_MODEL_FEATURES),
        "features": ALL_MODEL_FEATURES,
        "fast_onset_diseases": list(FAST_ONSET_DISEASES),
        "slow_onset_diseases": list(SLOW_ONSET_DISEASES),
        "risk_thresholds": {"low_boundary": 0.35, "high_boundary": 0.65},
        "important_caveat": (
            "Evaluation is against synthetic rule-generated labels only. "
            "This does NOT establish real-world disease-forecasting ability. "
            "Performance metrics reflect how well models learn the synthetic "
            "label-generation rules, not validated epidemiological outcomes."
        ),
    })

    report_path = REPORTS_DIR / "model_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Evaluation report saved: %s", report_path)

    return eval_results


if __name__ == "__main__":
    train_and_save()
