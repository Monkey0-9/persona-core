"""Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DriftSeverity = Literal["stable", "monitor", "drift"]
AlertStatus = Literal["NORMAL", "WARNING", "CRITICAL"]


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    severity: DriftSeverity
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float


@dataclass(frozen=True)
class BatchDriftReport:
    overall_status: DriftSeverity
    alert_status: AlertStatus
    max_psi: float
    max_ks_statistic: float
    sample_size: int
    evaluated_at: str
    features: list[FeatureDriftResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "alert_status": self.alert_status,
            "max_psi": self.max_psi,
            "max_ks_statistic": self.max_ks_statistic,
            "sample_size": self.sample_size,
            "evaluated_at": self.evaluated_at,
            "features": [asdict(f) for f in self.features],
        }


def classify_psi_severity(
    psi: float,
    stable_threshold: float = 0.10,
    monitor_threshold: float = 0.25,
) -> DriftSeverity:
    """Classify PSI score into stable (< 0.10), monitor (< 0.25), or drift (>= 0.25)."""
    if psi < stable_threshold:
        return "stable"
    if psi < monitor_threshold:
        return "monitor"
    return "drift"


def classify_alert_status(
    *,
    max_psi: float,
    max_ks_statistic: float,
    min_ks_pvalue: float,
    stable_threshold: float = 0.10,
    monitor_threshold: float = 0.25,
) -> AlertStatus:
    """Classify combined PSI + KS drift into NORMAL, WARNING, or CRITICAL."""
    if max_psi >= monitor_threshold or (max_ks_statistic >= 0.35 and min_ks_pvalue < 0.01):
        return "CRITICAL"
    if max_psi >= stable_threshold or (max_ks_statistic >= 0.20 and min_ks_pvalue < 0.05):
        return "WARNING"
    return "NORMAL"


def compute_reference_stats(
    df: pd.DataFrame,
    features: list[str],
) -> dict[str, dict[str, float]]:
    """Compute baseline reference distribution summary statistics per numeric feature."""
    stats: dict[str, dict[str, float]] = {}
    for feat in features:
        arr = df[feat].astype(float).dropna().to_numpy(dtype=np.float64)
        stats[feat] = {
            "count": float(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            "min": round(float(np.min(arr)), 4),
            "q05": round(float(np.quantile(arr, 0.05)), 4),
            "q25": round(float(np.quantile(arr, 0.25)), 4),
            "median": round(float(np.median(arr)), 4),
            "q75": round(float(np.quantile(arr, 0.75)), 4),
            "q95": round(float(np.quantile(arr, 0.95)), 4),
            "max": round(float(np.max(arr)), 4),
        }
    return stats


def compute_ks_test(
    baseline: np.ndarray | list[float],
    current: np.ndarray | list[float],
) -> tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov test statistic and p-value."""
    base_arr = np.asarray(baseline, dtype=np.float64)
    curr_arr = np.asarray(current, dtype=np.float64)
    base_arr = base_arr[~np.isnan(base_arr)]
    curr_arr = curr_arr[~np.isnan(curr_arr)]
    if len(base_arr) == 0 or len(curr_arr) == 0:
        raise ValueError("Baseline and current arrays must be non-empty for KS test.")
    res = ks_2samp(base_arr, curr_arr)
    return round(float(res.statistic), 4), round(float(res.pvalue), 4)


def compute_psi(
    baseline: np.ndarray | list[float],
    current: np.ndarray | list[float],
    bins: int = 10,
    stable_threshold: float = 0.10,
    monitor_threshold: float = 0.25,
) -> tuple[float, DriftSeverity]:
    """Compute Population Stability Index (PSI) using baseline quantile bins."""
    base_arr = np.asarray(baseline, dtype=np.float64)
    curr_arr = np.asarray(current, dtype=np.float64)
    base_arr = base_arr[~np.isnan(base_arr)]
    curr_arr = curr_arr[~np.isnan(curr_arr)]

    if len(base_arr) == 0 or len(curr_arr) == 0:
        raise ValueError("Baseline and current arrays must be non-empty for PSI calculation.")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(base_arr, quantiles))
    if len(edges) < 3:
        min_val = float(min(np.min(base_arr), np.min(curr_arr)))
        max_val = float(max(np.max(base_arr), np.max(curr_arr)))
        if abs(max_val - min_val) < 1e-9:
            return 0.0, "stable"
        edges = np.linspace(min_val, max_val, bins + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(base_arr, bins=edges)
    curr_counts, _ = np.histogram(curr_arr, bins=edges)

    eps = 1e-6
    base_pct = (base_counts / max(float(np.sum(base_counts)), 1.0)) + eps
    curr_pct = (curr_counts / max(float(np.sum(curr_counts)), 1.0)) + eps
    base_pct = base_pct / np.sum(base_pct)
    curr_pct = curr_pct / np.sum(curr_pct)

    psi_value = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    psi_rounded = round(max(psi_value, 0.0), 4)
    severity = classify_psi_severity(
        psi_rounded,
        stable_threshold=stable_threshold,
        monitor_threshold=monitor_threshold,
    )
    return psi_rounded, severity


def evaluate_single_record_drift(
    record: dict[str, float],
    reference_stats: dict[str, dict[str, float]],
    z_threshold: float = 2.75,
) -> tuple[bool, list[str]]:
    """Flag whether an incoming single customer is out-of-distribution relative to baseline."""
    flagged_features: list[str] = []
    for feat, stats in reference_stats.items():
        if feat not in record:
            continue
        val = float(record[feat])
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), 1e-6)
        min_v = float(stats.get("min", -np.inf))
        max_v = float(stats.get("max", np.inf))
        z_score = abs(val - mean) / std
        if z_score > z_threshold or val < min_v * 0.85 or val > max_v * 1.15:
            flagged_features.append(feat)
    return len(flagged_features) > 0, flagged_features


def evaluate_batch_drift(
    baseline_distributions: dict[str, list[float]] | pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    bins: int = 10,
    stable_threshold: float = 0.10,
    monitor_threshold: float = 0.25,
) -> BatchDriftReport:
    """Evaluate PSI and KS drift across all tracked numeric features for an incoming batch."""
    results: list[FeatureDriftResult] = []
    for feat in features:
        base_vals = np.asarray(baseline_distributions[feat], dtype=np.float64)
        curr_vals = np.asarray(current_df[feat], dtype=np.float64)
        psi, severity = compute_psi(
            base_vals,
            curr_vals,
            bins=bins,
            stable_threshold=stable_threshold,
            monitor_threshold=monitor_threshold,
        )
        ks_stat, ks_pval = compute_ks_test(base_vals, curr_vals)
        curr_std = float(np.std(curr_vals, ddof=1)) if len(curr_vals) > 1 else 0.0
        base_std = float(np.std(base_vals, ddof=1)) if len(base_vals) > 1 else 0.0
        results.append(
            FeatureDriftResult(
                feature=feat,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                severity=severity,
                baseline_mean=round(float(np.mean(base_vals)), 4),
                current_mean=round(float(np.mean(curr_vals)), 4),
                baseline_std=round(base_std, 4),
                current_std=round(curr_std, 4),
            )
        )

    max_psi = round(max((r.psi for r in results), default=0.0), 4)
    max_ks = round(max((r.ks_statistic for r in results), default=0.0), 4)
    min_pval = min((r.ks_pvalue for r in results), default=1.0)
    overall = classify_psi_severity(
        max_psi,
        stable_threshold=stable_threshold,
        monitor_threshold=monitor_threshold,
    )
    alert_status = classify_alert_status(
        max_psi=max_psi,
        max_ks_statistic=max_ks,
        min_ks_pvalue=min_pval,
        stable_threshold=stable_threshold,
        monitor_threshold=monitor_threshold,
    )
    return BatchDriftReport(
        overall_status=overall,
        alert_status=alert_status,
        max_psi=max_psi,
        max_ks_statistic=max_ks,
        sample_size=int(len(current_df)),
        evaluated_at=datetime.now(UTC).isoformat(),
        features=results,
    )
