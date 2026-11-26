"""Tests for data contracts, privacy stripping, EDA, and feature/PCA pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from mall_seg.contracts import (
    generate_deterministic_fixture,
    resolve_dataset_path,
    strip_customer_id,
    validate_raw_dataframe,
)
from mall_seg.eda import compute_eda_report
from mall_seg.features import fit_transform_features, transform_features


def _make_sample_df(tmp_path: Path) -> pd.DataFrame:
    fixture = generate_deterministic_fixture(tmp_path / "sample.csv", n_rows=200, seed=42)
    return pd.read_csv(fixture)


def test_resolve_dataset_prioritizes_primary(tmp_path: Path) -> None:
    primary = tmp_path / "Mall_Customers.csv"
    fallback = tmp_path / "fallback.csv"
    generate_deterministic_fixture(primary, n_rows=60, seed=7)

    resolved, kind = resolve_dataset_path(primary, fallback)
    assert resolved == primary
    assert kind == "primary"

    primary.unlink()
    resolved_fb, kind_fb = resolve_dataset_path(primary, fallback)
    assert resolved_fb == fallback
    assert kind_fb == "fixture"
    assert fallback.exists()


def test_pandera_contract_accepts_valid_and_rejects_corrupt(tmp_path: Path) -> None:
    df = _make_sample_df(tmp_path)
    validated = validate_raw_dataframe(df)
    assert len(validated) == 200

    # Duplicate CustomerID
    dup_df = df.copy()
    dup_df.loc[1, "CustomerID"] = dup_df.loc[0, "CustomerID"]
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_dataframe(dup_df)

    # Invalid Gender
    bad_gender = df.copy()
    bad_gender.loc[0, "Gender"] = "Unknown"
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_dataframe(bad_gender)

    # Degenerate variance
    zero_var = df.copy()
    zero_var["Age"] = 30
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_dataframe(zero_var)

    # Too few rows
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_dataframe(df.iloc[:20].copy(), min_rows=50)


def test_privacy_strip_and_gender_opt_in_features(tmp_path: Path) -> None:
    df = _make_sample_df(tmp_path)

    with pytest.raises(ValueError, match="CustomerID must be stripped"):
        fit_transform_features(df, include_gender=False)

    df_no_id = strip_customer_id(df)
    assert "CustomerID" not in df_no_id.columns

    art_no_gender = fit_transform_features(
        df_no_id, include_gender=False, pca_components=2, seed=42
    )
    assert art_no_gender.X_scaled.shape == (200, 3)
    assert art_no_gender.X_pca.shape == (200, 2)
    assert len(art_no_gender.feature_names) == 3
    assert np.allclose(np.mean(art_no_gender.X_scaled, axis=0), 0.0, atol=1e-7)

    art_with_gender = fit_transform_features(
        df_no_id, include_gender=True, pca_components=2, seed=42
    )
    assert art_with_gender.X_scaled.shape == (200, 4)
    assert len(art_with_gender.feature_names) == 4

    X_s, X_p = transform_features(
        df_no_id.iloc[:5], art_no_gender.preprocessor, art_no_gender.pca
    )
    assert X_s.shape == (5, 3)
    assert X_p.shape == (5, 2)


def test_eda_report_outputs(tmp_path: Path) -> None:
    df = _make_sample_df(tmp_path)
    report = compute_eda_report(df)
    assert report.row_count == 200
    assert len(report.numeric_summaries) == 3
    total_share = sum(report.gender_breakdown.shares.values())
    assert pytest.approx(total_share, abs=1e-3) == 1.0
    assert len(report.correlation_highlights) == 3
