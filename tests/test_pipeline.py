"""Pytest suite verifying invariants & determinism across the modular mall_seg engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mall_seg.clustering import (
    compute_k_distance_curve,
    fit_agglomerative_ward,
    fit_dbscan_tuned,
    fit_gmm,
    fit_kmeans,
)
from mall_seg.config import load_config
from mall_seg.evaluation import compare_partitions, evaluate_kmeans_stability, evaluate_partition
from mall_seg.preprocessing import prepare_and_project_features, project_new_records
from mall_seg.profiling import generate_segment_kpis
from mall_seg.schema import (
    NUMERIC_FEATURE_COLUMNS,
    MallCustomerRecord,
    generate_deterministic_fixture,
    rows_to_dataframe,
    validate_domain_invariants,
)
from mall_seg.visualization import (
    build_2d_pca_figure,
    build_3d_segment_figure,
    build_k_distance_figure,
)


def test_config_loads_pipeline_yaml() -> None:
    cfg = load_config("configs/pipeline_config.yaml")
    assert cfg.clustering.seed == 42
    assert cfg.clustering.target_k == 5
    assert cfg.privacy.include_gender is False
    assert cfg.privacy.strip_identifier is True


def test_schema_and_domain_invariants(tmp_path: Path) -> None:
    sample_path = generate_deterministic_fixture(tmp_path / "mall.csv", n_rows=80, seed=42)
    df = pd.read_csv(sample_path)
    validated = validate_domain_invariants(df, min_rows=50)
    assert len(validated) == 80
    assert list(validated.columns[:2]) == ["CustomerID", "Gender"]

    typed_rows = [
        MallCustomerRecord(
            customer_id=i + 1,
            gender="Female" if i % 2 == 0 else "Male",
            age=20 + i,
            annual_income_k=25.0 + i * 2.0,
            spending_score=float(10 + (i * 5) % 85),
        )
        for i in range(15)
    ]
    typed_df = rows_to_dataframe(typed_rows)
    assert len(validate_domain_invariants(typed_df, min_rows=15)) == 15


def test_preprocessing_and_clustering_determinism(tmp_path: Path) -> None:
    sample_path = generate_deterministic_fixture(tmp_path / "mall.csv", n_rows=200, seed=42)
    raw_df = pd.read_csv(sample_path)

    feats_a = prepare_and_project_features(raw_df, include_gender=False, pca_components=2, seed=42)
    feats_b = prepare_and_project_features(raw_df, include_gender=False, pca_components=2, seed=42)
    assert np.allclose(feats_a.X_scaled, feats_b.X_scaled)
    assert np.allclose(feats_a.X_pca, feats_b.X_pca)
    assert feats_a.feature_names == NUMERIC_FEATURE_COLUMNS

    X_s_new, X_p_new = project_new_records(raw_df.iloc[:4], feats_a.preprocessor, feats_a.pca)
    assert np.allclose(X_s_new, feats_a.X_scaled[:4])
    assert np.allclose(X_p_new, feats_a.X_pca[:4])

    _, km_a = fit_kmeans(feats_a.X_scaled, n_clusters=5, n_init=20, seed=42)
    _, km_b = fit_kmeans(feats_b.X_scaled, n_clusters=5, n_init=20, seed=42)
    assert np.array_equal(km_a, km_b)

    _, ward_labels = fit_agglomerative_ward(feats_a.X_scaled, n_clusters=5)
    assert len(np.unique(ward_labels)) == 5

    kdist = compute_k_distance_curve(feats_a.X_scaled, min_samples=5)
    assert kdist.recommended_eps >= 0.25
    assert len(kdist.k_distances) == 200
    assert kdist.k_distances == sorted(kdist.k_distances)

    _, db_labels, used_eps = fit_dbscan_tuned(feats_a.X_scaled, min_samples=5)
    assert used_eps == pytest.approx(kdist.recommended_eps)
    assert len(db_labels) == 200

    _, gmm_labels, posteriors, entropy = fit_gmm(feats_a.X_scaled, n_components=5, seed=42)
    assert posteriors.shape == (200, 5)
    assert np.all((entropy >= 0.0) & (entropy <= 1.0))

    comparison = compare_partitions(
        feats_a.X_scaled,
        {"K-Means": km_a, "Ward": ward_labels, "GMM": gmm_labels, "DBSCAN": db_labels},
    )
    assert len(comparison) == 4
    single_eval = evaluate_partition(feats_a.X_scaled, km_a, "K-Means")
    assert single_eval.silhouette is not None and single_eval.silhouette > 0.25


def test_profiling_kpis_stability_and_plotly_builders(tmp_path: Path) -> None:
    sample_path = generate_deterministic_fixture(tmp_path / "mall.csv", n_rows=200, seed=42)
    raw_df = pd.read_csv(sample_path)
    feats = prepare_and_project_features(raw_df, include_gender=False, pca_components=2, seed=42)
    _, km_labels = fit_kmeans(feats.X_scaled, n_clusters=5, n_init=20, seed=42)
    _, _, _, entropy = fit_gmm(feats.X_scaled, n_components=5, seed=42)
    boundary = entropy > 0.45

    stability = evaluate_kmeans_stability(
        feats.X_scaled,
        km_labels,
        n_clusters=5,
        n_bootstrap=20,
        bootstrap_n_init=10,
        min_jaccard_stability=0.65,
        seed=42,
    )
    assert stability.passed_gate is True

    kpis = generate_segment_kpis(raw_df, km_labels, is_boundary=boundary)
    assert len(kpis) == 5
    assert sum(item.customer_count for item in kpis) == 200
    assert kpis[0].estimated_revenue_index >= kpis[-1].estimated_revenue_index

    scored_df = pd.DataFrame(
        {
            "customer_ref": [f"C-{i}" for i in range(200)],
            "gender": raw_df["Gender"].astype(str),
            "age": raw_df["Age"].astype(int),
            "annual_income_k": raw_df["Annual Income (k$)"].astype(float),
            "spending_score": raw_df["Spending Score (1-100)"].astype(float),
            "persona_name": [f"Cluster-{int(c)}" for c in km_labels],
            "recommended_action": ["Action"] * 200,
            "gmm_confidence": [0.9] * 200,
            "normalized_entropy": entropy,
            "is_boundary_customer": boundary,
            "pca_1": feats.X_pca[:, 0],
            "pca_2": feats.X_pca[:, 1],
        }
    )
    fig_2d = build_2d_pca_figure(scored_df, feats.explained_variance_ratio)
    fig_3d = build_3d_segment_figure(scored_df)
    fig_kdist = build_k_distance_figure(feats.X_scaled, min_samples=5)
    assert len(fig_2d.data) >= 1
    assert len(fig_3d.data) >= 1
    assert len(fig_kdist.data) == 1
