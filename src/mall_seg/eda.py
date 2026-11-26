"""Exploratory Data Analysis module for Mall Customer Segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

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
class GenderFeatureStat:
    gender: str
    feature: str
    count: int
    mean: float
    std: float
    q1: float
    median: float
    q3: float


@dataclass(frozen=True)
class GenderHypothesisTest:
    feature: str
    female_mean: float
    male_mean: float
    mean_diff: float
    welch_t_stat: float
    welch_p_value: float
    mannwhitney_p_value: float
    cohens_d: float
    significant_at_05: bool


@dataclass(frozen=True)
class AgeCohortStat:
    cohort: str
    count: int
    share: float
    mean_income: float
    mean_spending: float
    female_share: float


@dataclass(frozen=True)
class EDAReport:
    row_count: int
    numeric_summaries: list[NumericSummary]
    gender_breakdown: GenderBreakdown
    pearson_matrix: dict[str, dict[str, float]]
    spearman_matrix: dict[str, dict[str, float]]
    correlation_highlights: list[CorrelationPair]
    gender_feature_stats: list[GenderFeatureStat]
    gender_hypothesis_tests: list[GenderHypothesisTest]
    age_cohorts: list[AgeCohortStat]

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

    gender_feature_stats: list[GenderFeatureStat] = []
    gender_hypothesis_tests: list[GenderHypothesisTest] = []
    female_df = df[df["Gender"] == "Female"]
    male_df = df[df["Gender"] == "Male"]

    for col in NUMERIC_FEATURE_COLUMNS:
        for g_label, g_sub in (("Female", female_df), ("Male", male_df)):
            if g_sub.empty:
                continue
            s_g = g_sub[col].astype(float)
            gender_feature_stats.append(
                GenderFeatureStat(
                    gender=g_label,
                    feature=col,
                    count=int(len(s_g)),
                    mean=round(float(s_g.mean()), 2),
                    std=round(float(s_g.std(ddof=1)) if len(s_g) > 1 else 0.0, 2),
                    q1=round(float(s_g.quantile(0.25)), 2),
                    median=round(float(s_g.median()), 2),
                    q3=round(float(s_g.quantile(0.75)), 2),
                )
            )

        if len(female_df) >= 2 and len(male_df) >= 2:
            f_arr = female_df[col].astype(float).to_numpy()
            m_arr = male_df[col].astype(float).to_numpy()
            t_res = ttest_ind(f_arr, m_arr, equal_var=False)
            mw_res = mannwhitneyu(f_arr, m_arr, alternative="two-sided")
            f_var = float(np.var(f_arr, ddof=1))
            m_var = float(np.var(m_arr, ddof=1))
            pooled_sd = float(
                np.sqrt(
                    ((len(f_arr) - 1) * f_var + (len(m_arr) - 1) * m_var)
                    / max(len(f_arr) + len(m_arr) - 2, 1)
                )
            )
            cohens_d = (float(np.mean(f_arr)) - float(np.mean(m_arr))) / max(pooled_sd, 1e-9)
            t_stat = float(np.nan_to_num(t_res.statistic, nan=0.0))
            p_welch = float(np.nan_to_num(t_res.pvalue, nan=1.0))
            p_mw = float(np.nan_to_num(mw_res.pvalue, nan=1.0))
            gender_hypothesis_tests.append(
                GenderHypothesisTest(
                    feature=col,
                    female_mean=round(float(np.mean(f_arr)), 2),
                    male_mean=round(float(np.mean(m_arr)), 2),
                    mean_diff=round(float(np.mean(f_arr) - np.mean(m_arr)), 2),
                    welch_t_stat=round(t_stat, 4),
                    welch_p_value=round(p_welch, 4),
                    mannwhitney_p_value=round(p_mw, 4),
                    cohens_d=round(cohens_d, 4),
                    significant_at_05=p_welch < 0.05,
                )
            )

    age_bins = [
        ("18-25", 16, 25),
        ("26-35", 26, 35),
        ("36-45", 36, 45),
        ("46-55", 46, 55),
        ("56+", 56, 120),
    ]
    age_cohorts: list[AgeCohortStat] = []
    ages_col = df["Age"].astype(float)
    for label, lo, hi in age_bins:
        sub = df[(ages_col >= lo) & (ages_col <= hi)]
        cnt = int(len(sub))
        if cnt == 0:
            continue
        f_cnt = int((sub["Gender"] == "Female").sum())
        age_cohorts.append(
            AgeCohortStat(
                cohort=label,
                count=cnt,
                share=round(cnt / total, 4),
                mean_income=round(float(sub["Annual Income (k$)"].astype(float).mean()), 2),
                mean_spending=round(float(sub["Spending Score (1-100)"].astype(float).mean()), 2),
                female_share=round(f_cnt / cnt, 4),
            )
        )

    return EDAReport(
        row_count=int(len(df)),
        numeric_summaries=summaries,
        gender_breakdown=GenderBreakdown(counts=gender_counts, shares=gender_shares),
        pearson_matrix=pearson_matrix,
        spearman_matrix=spearman_matrix,
        correlation_highlights=highlights,
        gender_feature_stats=gender_feature_stats,
        gender_hypothesis_tests=gender_hypothesis_tests,
        age_cohorts=age_cohorts,
    )
