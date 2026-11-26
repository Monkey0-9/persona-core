"""Exploratory Data Analysis module for Mall Customer Segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from mall_seg.contracts import NUMERIC_FEATURE_COLUMNS


@dataclass(frozen=True)
class NumericSummary:
    feature: str
    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    iqr_outlier_count: int


@dataclass(frozen=True)
class GenderBreakdown:
    counts: dict[str, int]
    shares: dict[str, float]


@dataclass(frozen=True)
class CorrelationPair:
    feature_a: str
    feature_b: str
    pearson: float
    spearman: float


@dataclass(frozen=True)
class EDAReport:
    row_count: int
    numeric_summaries: list[NumericSummary]
    gender_breakdown: GenderBreakdown
    pearson_matrix: dict[str, dict[str, float]]
    spearman_matrix: dict[str, dict[str, float]]
    correlation_highlights: list[CorrelationPair]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_eda_report(df: pd.DataFrame) -> EDAReport:
    """Compute distribution statistics, gender composition, correlations, and IQR outliers."""
    summaries: list[NumericSummary] = []
    for col in NUMERIC_FEATURE_COLUMNS:
        series = df[col].astype(float)
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((series < lower) | (series > upper)).sum())
        summaries.append(
            NumericSummary(
                feature=col,
                count=int(series.shape[0]),
                mean=round(float(series.mean()), 4),
                std=round(float(series.std(ddof=1)), 4),
                min=round(float(series.min()), 4),
                q1=round(q1, 4),
                median=round(float(series.median()), 4),
                q3=round(q3, 4),
                max=round(float(series.max()), 4),
                iqr_outlier_count=outliers,
            )
        )

    gender_counts_series = df["Gender"].value_counts()
    total = max(int(len(df)), 1)
    gender_counts = {str(k): int(v) for k, v in gender_counts_series.items()}
    gender_shares = {k: round(v / total, 4) for k, v in gender_counts.items()}

    num_df = df[NUMERIC_FEATURE_COLUMNS].astype(float)
    pearson_df = num_df.corr(method="pearson").round(4)
    spearman_df = num_df.corr(method="spearman").round(4)

    pearson_matrix = {
        str(row): {str(col): float(pearson_df.loc[row, col]) for col in pearson_df.columns}
        for row in pearson_df.index
    }
    spearman_matrix = {
        str(row): {str(col): float(spearman_df.loc[row, col]) for col in spearman_df.columns}
        for row in spearman_df.index
    }

    highlights: list[CorrelationPair] = []
    for i, feat_a in enumerate(NUMERIC_FEATURE_COLUMNS):
        for feat_b in NUMERIC_FEATURE_COLUMNS[i + 1 :]:
            p_val = float(np.nan_to_num(pearson_df.loc[feat_a, feat_b], nan=0.0))
            s_val = float(np.nan_to_num(spearman_df.loc[feat_a, feat_b], nan=0.0))
            highlights.append(
                CorrelationPair(
                    feature_a=feat_a,
                    feature_b=feat_b,
                    pearson=round(p_val, 4),
                    spearman=round(s_val, 4),
                )
            )

    return EDAReport(
        row_count=int(len(df)),
        numeric_summaries=summaries,
        gender_breakdown=GenderBreakdown(counts=gender_counts, shares=gender_shares),
        pearson_matrix=pearson_matrix,
        spearman_matrix=spearman_matrix,
        correlation_highlights=highlights,
    )
