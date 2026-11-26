"""Tests for clustering suite, bootstrap stability, explainability, personas, and PSI drift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mall_seg.algorithms import (
    compute_normalized_entropy,
    run_clustering_suite,
    safe_cluster_metrics,
)
from mall_seg.contracts import (
    NUMERIC_FEATURE_COLUMNS,
    generate_deterministic_fixture,
    strip_customer_id,
)
from mall_seg.drift import compute_psi, evaluate_batch_drift
from mall_seg.explainability import fit_surrogate_explainer
from mall_seg.features import fit_transform_features
from mall_seg.personas import map_clusters_to_personas
from mall_seg.stability import evaluate_kmeans_stability


def test_clustering_suite_stability_and_personas(tmp_path: Path) -> None:
    fixture = generate_deterministic_fixture(tmp_path / "sample.csv", n_rows=200, seed=42)
    raw_df = pd.read_csv(fixture)
    df_no_id = strip_customer_id(raw_df)
    feats = fit_transform_features(df_no_id, include_gender=False, pca_components=2, seed=42)

    suite = run_clustering_suite(
        feats.X_scaled,
        target_k=5,
        k_min=2,
        k_max=8,
        kmeans_n_init=25,
        seed=42,
    )
    assert len(suite.benchmarks) == 5
    assert "ari_matrix" in suite.consensus
    assert len(suite.kmeans_sweep) == 7
    assert len(suite.gmm_sweep) == 7
    assert suite.ward_linkage_matrix.shape == (199, 4)
    assert np.all((suite.gmm_entropy >= 0.0) & (suite.gmm_entropy <= 1.0))
    assert np.allclose(np.sum(suite.gmm_posteriors, axis=1), 1.0, atol=1e-6)

    # Bootstrap stability gate
    stability = evaluate_kmeans_stability(
        feats.X_scaled,
        suite.kmeans_labels,
        n_clusters=5,
        n_bootstrap=40,
        bootstrap_n_init=10,
        min_jaccard_stability=0.70,
        seed=42,
    )
    assert stability.passed_gate is True
    assert stability.mean_jaccard >= 0.70

    strict_fail = evaluate_kmeans_stability(
        feats.X_scaled,
        suite.kmeans_labels,
        n_clusters=5,
        n_bootstrap=10,
        bootstrap_n_init=10,
        min_jaccard_stability=0.9999,
        seed=42,
    )
    assert strict_fail.passed_gate is False

    # Permutation invariance of deterministic persona mapping
    personas_a = map_clusters_to_personas(raw_df, suite.kmeans_labels)
    permuted_labels = (suite.kmeans_labels + 2) % 5
    personas_b = map_clusters_to_personas(raw_df, permuted_labels)
    assert {p.persona_key for p in personas_a.values()} == {
        p.persona_key for p in personas_b.values()
    }
    assert personas_a[0].persona_key == personas_b[2].persona_key

    # Surrogate explainer
    persona_names = {cid: p.persona_name for cid, p in personas_a.items()}
    expl = fit_surrogate_explainer(df_no_id, suite.kmeans_labels, persona_names, max_depth=3)
    assert expl.fidelity_score >= 0.75
    assert len(expl.rules) >= 3


def test_safe_metrics_with_dbscan_noise_and_entropy_bounds() -> None:
    X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1], [99.0, 99.0]])
    labels_with_noise = np.array([0, 0, 1, 1, -1])
    sil, ch, db, noise = safe_cluster_metrics(X, labels_with_noise)
    assert noise == 1
    assert sil is not None and sil > 0.8
    assert ch is not None and db is not None

    all_noise = np.array([-1, -1, -1, -1, -1])
    sil_n, ch_n, db_n, noise_n = safe_cluster_metrics(X, all_noise)
    assert sil_n is None and ch_n is None and db_n is None
    assert noise_n == 5

    uniform = np.full((2, 5), 0.2)
    certain = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    assert np.allclose(compute_normalized_entropy(uniform), 1.0, atol=1e-4)
    assert np.allclose(compute_normalized_entropy(certain), 0.0, atol=1e-4)


def test_psi_drift_tiers() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(50.0, 10.0, size=500)
    same = rng.normal(50.0, 10.0, size=500)
    shifted = rng.normal(78.0, 10.0, size=500)

    psi_stable, sev_stable = compute_psi(base, same, bins=10)
    assert psi_stable < 0.10
    assert sev_stable == "stable"

    psi_drift, sev_drift = compute_psi(base, shifted, bins=10)
    assert psi_drift >= 0.25
    assert sev_drift == "drift"

    base_dict = {col: rng.normal(40.0, 5.0, size=100).tolist() for col in NUMERIC_FEATURE_COLUMNS}
    curr_df = pd.DataFrame({col: rng.normal(40.0, 5.0, size=80) for col in NUMERIC_FEATURE_COLUMNS})
    batch_report = evaluate_batch_drift(base_dict, curr_df, NUMERIC_FEATURE_COLUMNS)
    assert batch_report.sample_size == 80
    assert batch_report.overall_status in {"stable", "monitor", "drift"}
