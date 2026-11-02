"""
tests/test_model_evaluation.py
--------------------------------
Tests for reports/model_evaluation.json — verifies that the evaluation
report exists, has the right structure, and performance metrics are
within reasonable bounds (sanity checks, not ground-truth validation).
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "model_evaluation.json"


@pytest.fixture(scope="module")
def report():
    if not REPORT_PATH.exists():
        pytest.skip("reports/model_evaluation.json not found; run `python -m src.train_models` first")
    with open(REPORT_PATH) as f:
        return json.load(f)


class TestReportStructure:
    def test_has_14d_key(self, report):
        assert "14d" in report

    def test_has_30d_key(self, report):
        assert "30d" in report

    def test_has_metadata_key(self, report):
        assert "metadata" in report

    def test_three_models_per_horizon(self, report):
        for horizon in ("14d", "30d"):
            assert len(report[horizon]) == 3, \
                f"Expected 3 models in {horizon}, got {len(report[horizon])}"

    def test_model_names_include_gradient_boosting(self, report):
        names_14d = {e["model"] for e in report["14d"]}
        assert "GradientBoosting" in names_14d

    def test_metadata_has_caveat(self, report):
        meta = report["metadata"]
        assert "important_caveat" in meta
        assert len(meta["important_caveat"]) > 20


class TestMetricRanges:
    @pytest.mark.parametrize("horizon", ["14d", "30d"])
    def test_accuracy_in_range(self, report, horizon):
        for entry in report[horizon]:
            acc = entry["accuracy"]
            assert 0.0 <= acc <= 1.0, f"{horizon}/{entry['model']} accuracy={acc}"

    @pytest.mark.parametrize("horizon", ["14d", "30d"])
    def test_roc_auc_above_dummy(self, report, horizon):
        entries = {e["model"]: e for e in report[horizon]}
        dummy_auc = entries["Dummy (stratified)"]["roc_auc"]
        gb_auc = entries["GradientBoosting"]["roc_auc"]
        assert gb_auc > dummy_auc, \
            f"{horizon}: GBT AUC ({gb_auc}) should beat Dummy ({dummy_auc})"

    @pytest.mark.parametrize("horizon", ["14d", "30d"])
    def test_f1_above_0_5(self, report, horizon):
        entries = {e["model"]: e for e in report[horizon]}
        for name in ("RandomForest", "GradientBoosting"):
            f1 = entries[name]["f1_macro"]
            assert f1 > 0.5, f"{horizon}/{name} f1_macro={f1} below 0.5"

    @pytest.mark.parametrize("horizon", ["14d", "30d"])
    def test_confusion_matrix_shape(self, report, horizon):
        for entry in report[horizon]:
            cm = entry["confusion_matrix"]
            assert len(cm) == 2, "Confusion matrix should be 2x2"
            assert all(len(row) == 2 for row in cm)


class TestMetadataFields:
    def test_train_rows_positive(self, report):
        assert report["metadata"]["train_rows"] > 0

    def test_test_rows_positive(self, report):
        assert report["metadata"]["test_rows"] > 0

    def test_real_rows_are_from_dataset(self, report):
        # Training dataset has 798 rows
        real = report["metadata"]["real_rows"]
        assert 700 <= real <= 900, f"Unexpected real_rows count: {real}"

    def test_feature_count(self, report):
        assert report["metadata"]["feature_count"] == 40

    def test_best_model_keys(self, report):
        assert "best_14d" in report["metadata"]
        assert "best_30d" in report["metadata"]
