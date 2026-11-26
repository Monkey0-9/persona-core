"""Tests for versioned ML artifact store, KS + PSI drift, and OOD scoring checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mall_seg.drift import compute_ks_test, compute_reference_stats, evaluate_single_record_drift
from ml.drift import evaluate_batch_drift
from ml.train import train_and_export_versioned_model


def test_versioned_ml_bundle_and_drift() -> None:
    # Ensure default artifacts/models/v1.0.0 is written cleanly and matches schema
    result = train_and_export_versioned_model(version="v1.0.0", log_mlflow=False)
    assert result["row_count"] == 200

    v_dir = Path("artifacts/models/v1.0.0")
    assert (v_dir / "model.joblib").exists()
    assert (v_dir / "metrics.json").exists()
    assert (v_dir / "feature_schema.json").exists()
    assert (v_dir / "reference_stats.json").exists()

    schema = json.loads((v_dir / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["model_version"] == "v1.0.0"
    assert len(schema["feature_schema_hash"]) == 16

    ref_stats = json.loads((v_dir / "reference_stats.json").read_text(encoding="utf-8"))
    assert "Age" in ref_stats and "q95" in ref_stats["Age"]

    # Single record OOD drift check
    in_range, flagged_ok = evaluate_single_record_drift(
        {"Age": 32.0, "Annual Income (k$)": 60.0, "Spending Score (1-100)": 55.0},
        reference_stats=ref_stats,
    )
    assert in_range is False
    assert flagged_ok == []

    out_of_range, flagged_bad = evaluate_single_record_drift(
        {"Age": 95.0, "Annual Income (k$)": 320.0, "Spending Score (1-100)": 55.0},
        reference_stats=ref_stats,
    )
    assert out_of_range is True
    assert "Annual Income (k$)" in flagged_bad

    # Batch PSI + KS statistics
    rng = np.random.default_rng(7)
    base = rng.normal(50.0, 10.0, size=300)
    shifted = rng.normal(85.0, 10.0, size=300)
    ks_stat, ks_pval = compute_ks_test(base, shifted)
    assert ks_stat > 0.7
    assert ks_pval < 0.01

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    df = pd.DataFrame({col: base for col in features})
    stats_out = compute_reference_stats(df, features)
    assert stats_out["Age"]["count"] == 300.0

    batch = evaluate_batch_drift(
        {col: base.tolist() for col in features},
        pd.DataFrame({col: shifted for col in features}),
        features,
    )
    assert batch.max_ks_statistic > 0.7
    assert batch.overall_status == "drift"
