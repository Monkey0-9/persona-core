"""Cluster validity evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz & stability)."""

from __future__ import annotations

import numpy as np

from mall_seg.algorithms import AlgorithmMetrics, compute_normalized_entropy, safe_cluster_metrics
from mall_seg.stability import StabilityReport, evaluate_kmeans_stability


def evaluate_partition(
    X: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str,
    extra: dict[str, float | int | str] | None = None,
) -> AlgorithmMetrics:
    """Compute Silhouette, Calinski-Harabasz, Davies-Bouldin, and noise count for any partition."""
    sil, ch, db, noise_count = safe_cluster_metrics(X, labels)
    n_clusters = len([lbl for lbl in np.unique(labels) if int(lbl) != -1])
    return AlgorithmMetrics(
        algorithm=algorithm_name,
        n_clusters=n_clusters,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        noise_count=noise_count,
        extra=dict(extra or {}),
    )


def compare_partitions(
    X: np.ndarray,
    partitions: dict[str, np.ndarray],
) -> list[AlgorithmMetrics]:
    """Benchmark multiple named clustering assignments side by side."""
    results = [evaluate_partition(X, labels, name) for name, labels in partitions.items()]
    return sorted(
        results,
        key=lambda m: (
            m.silhouette is not None,
            m.silhouette if m.silhouette is not None else -1.0,
        ),
        reverse=True,
    )


__all__ = [
    "AlgorithmMetrics",
    "StabilityReport",
    "compare_partitions",
    "compute_normalized_entropy",
    "evaluate_kmeans_stability",
    "evaluate_partition",
    "safe_cluster_metrics",
]
