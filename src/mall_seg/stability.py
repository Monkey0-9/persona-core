"""Bootstrap cluster stability evaluation using Hungarian linear sum assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class StabilityReport:
    n_bootstrap: int
    n_clusters: int
    per_cluster_jaccard: dict[int, float]
    mean_jaccard: float
    min_jaccard: float
    min_jaccard_threshold: float
    passed_gate: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["per_cluster_jaccard"] = {str(k): v for k, v in self.per_cluster_jaccard.items()}
        return data


def evaluate_kmeans_stability(
    X_scaled: np.ndarray,
    reference_labels: np.ndarray,
    n_clusters: int = 5,
    n_bootstrap: int = 100,
    bootstrap_n_init: int = 20,
    min_jaccard_stability: float = 0.70,
    seed: int = 42,
) -> StabilityReport:
    """Measure bootstrap Jaccard cluster stability aligned via Hungarian assignment."""
    rng = np.random.default_rng(seed)
    n_samples = len(X_scaled)
    cluster_scores: dict[int, list[float]] = {c: [] for c in range(n_clusters)}

    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        km_boot = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=bootstrap_n_init,
            random_state=seed + b + 1,
        )
        km_boot.fit(X_scaled[boot_idx])

        # Evaluate set-level Jaccard on unique sampled indices to avoid duplicate-weight bias
        unique_idx = np.unique(boot_idx)
        ref_sub = reference_labels[unique_idx]
        boot_sub = km_boot.predict(X_scaled[unique_idx])

        jaccard_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
        for i in range(n_clusters):
            ref_mask = ref_sub == i
            for j in range(n_clusters):
                boot_mask = boot_sub == j
                union = int(np.sum(ref_mask | boot_mask))
                if union > 0:
                    inter = int(np.sum(ref_mask & boot_mask))
                    jaccard_matrix[i, j] = inter / float(union)

        row_ind, col_ind = linear_sum_assignment(-jaccard_matrix)
        for r, c in zip(row_ind, col_ind, strict=True):
            if np.any(ref_sub == r):
                cluster_scores[int(r)].append(float(jaccard_matrix[r, c]))

    per_cluster = {
        c: round(float(np.mean(scores)) if scores else 0.0, 4)
        for c, scores in cluster_scores.items()
    }
    mean_jaccard = round(float(np.mean(list(per_cluster.values()))), 4)
    min_jaccard = round(float(np.min(list(per_cluster.values()))), 4)
    passed = mean_jaccard >= min_jaccard_stability

    return StabilityReport(
        n_bootstrap=n_bootstrap,
        n_clusters=n_clusters,
        per_cluster_jaccard=per_cluster,
        mean_jaccard=mean_jaccard,
        min_jaccard=min_jaccard,
        min_jaccard_threshold=min_jaccard_stability,
        passed_gate=passed,
    )
