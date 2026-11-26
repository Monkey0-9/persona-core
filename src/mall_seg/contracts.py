"""Data contracts, config loading, privacy stripping, and deterministic fixture generator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml
from pydantic import BaseModel, Field

REQUIRED_COLUMNS: list[str] = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

NUMERIC_FEATURE_COLUMNS: list[str] = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

VALID_GENDERS: tuple[str, ...] = ("Male", "Female")


class DataConfig(BaseModel):
    primary_path: str = "data/raw/Mall_Customers.csv"
    fallback_path: str = "tests/fixtures/mall_customers_sample.csv"
    min_rows: int = Field(default=50, ge=10)


class PrivacyConfig(BaseModel):
    include_gender: bool = False
    strip_identifier: bool = True
    log_raw_payloads: bool = False


class FeaturesConfig(BaseModel):
    numeric_columns: list[str] = Field(default_factory=lambda: list(NUMERIC_FEATURE_COLUMNS))
    categorical_columns: list[str] = Field(default_factory=lambda: ["Gender"])
    pca_components: int = Field(default=2, ge=2, le=3)


class ClusteringConfig(BaseModel):
    seed: int = 42
    target_k: int = Field(default=5, ge=2, le=10)
    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=10, ge=3, le=15)
    kmeans_n_init: int = Field(default=50, ge=5)
    dbscan_min_samples: int = Field(default=5, ge=2)
    gmm_covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    entropy_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class StabilityConfig(BaseModel):
    bootstrap_iterations: int = Field(default=100, ge=10)
    bootstrap_n_init: int = Field(default=20, ge=5)
    min_jaccard_stability: float = Field(default=0.70, ge=0.0, le=1.0)


class DriftConfig(BaseModel):
    bins: int = Field(default=10, ge=4, le=50)
    stable_threshold: float = Field(default=0.10, ge=0.0)
    monitor_threshold: float = Field(default=0.25, ge=0.0)


class ExplainabilityConfig(BaseModel):
    max_depth: int = Field(default=3, ge=1, le=6)


class ArtifactsConfig(BaseModel):
    model_bundle_path: str = "artifacts/model_bundle.joblib"
    visuals_dir: str = "artifacts/visuals"
    frontend_snapshot_path: str = "public/artifacts/dashboard_snapshot.json"
    mlflow_tracking_uri: str = "sqlite:///artifacts/mlflow.db"
    mlflow_experiment: str = "mall_customer_segmentation"


class ApiConfig(BaseModel):
    max_batch_size: int = Field(default=1000, ge=1, le=1000)
    default_limit: int = Field(default=50, ge=1, le=500)
    max_limit: int = Field(default=500, ge=1, le=1000)


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load YAML configuration into a strictly validated AppConfig."""
    if config_path is not None:
        resolved = Path(config_path)
    elif env_path := os.getenv("MALL_SEG_CONFIG"):
        resolved = Path(env_path)
    elif Path("configs/pipeline_config.yaml").exists():
        resolved = Path("configs/pipeline_config.yaml")
    else:
        resolved = Path("configs/prod.yaml")
    if not resolved.exists():
        return AppConfig()
    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


def build_raw_schema(min_rows: int = 50) -> pa.DataFrameSchema:
    """Build Pandera lazy validation schema for raw Mall Customers data."""

    def _has_min_rows(df: pd.DataFrame) -> bool:
        return len(df) >= min_rows

    def _has_non_degenerate_variance(df: pd.DataFrame) -> bool:
        for col in NUMERIC_FEATURE_COLUMNS:
            if col not in df.columns:
                return False
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 2 or float(series.std(ddof=1)) <= 0.0:
                return False
        return True

    return pa.DataFrameSchema(
        columns={
            "CustomerID": pa.Column(
                int,
                checks=[pa.Check.gt(0)],
                unique=True,
                nullable=False,
                coerce=True,
            ),
            "Gender": pa.Column(
                str,
                checks=[pa.Check.isin(list(VALID_GENDERS))],
                nullable=False,
            ),
            "Age": pa.Column(
                int,
                checks=[pa.Check.in_range(16, 100)],
                nullable=False,
                coerce=True,
            ),
            "Annual Income (k$)": pa.Column(
                float,
                checks=[pa.Check.in_range(5.0, 500.0)],
                nullable=False,
                coerce=True,
            ),
            "Spending Score (1-100)": pa.Column(
                float,
                checks=[pa.Check.in_range(1.0, 100.0)],
                nullable=False,
                coerce=True,
            ),
        },
        checks=[
            pa.Check(_has_min_rows, error=f"Dataset must contain at least {min_rows} rows."),
            pa.Check(
                _has_non_degenerate_variance,
                error="All numeric feature columns must have non-degenerate variance (std > 0).",
            ),
        ],
        strict=False,
        ordered=False,
    )


def validate_raw_dataframe(df: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
    """Validate raw customer dataframe with Pandera lazy validation."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    schema = build_raw_schema(min_rows=min_rows)
    validated = schema.validate(df.copy(), lazy=True)
    return validated[REQUIRED_COLUMNS].copy()


def strip_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Remove CustomerID before scaling and distance geometry."""
    return df.drop(columns=["CustomerID"], errors="ignore").copy()


def generate_deterministic_fixture(
    output_path: str | Path = "tests/fixtures/mall_customers_sample.csv",
    n_rows: int = 200,
    seed: int = 42,
) -> Path:
    """Generate a deterministic, realistic 200-row Mall Customers fixture."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Realistic 5-segment mixture matching Kaggle Mall Customer Segmentation geometry:
    # (count_share, age_mean, age_sd, inc_mean, inc_sd, spend_mean, spend_sd, female_prob)
    archetypes = [
        (0.40, 42.5, 11.5, 55.0, 9.0, 49.5, 7.5, 0.59),  # Core Mainstream
        (0.20, 32.7, 3.8, 86.5, 13.5, 82.0, 8.5, 0.54),  # Affluent VIPs
        (0.17, 41.0, 10.5, 88.0, 14.0, 17.5, 9.0, 0.46),  # Cautious Affluent
        (0.11, 25.2, 4.8, 26.0, 7.0, 78.5, 10.0, 0.58),  # Young Trendsetters
        (0.12, 45.2, 12.0, 26.5, 7.5, 21.0, 10.5, 0.60),  # Budget Pragmatists
    ]

    counts = [int(round(share * n_rows)) for share, *_ in archetypes]
    counts[0] += n_rows - sum(counts)

    rows: list[dict[str, object]] = []
    for count, (_, age_m, age_s, inc_m, inc_s, sp_m, sp_s, f_prob) in zip(
        counts, archetypes, strict=True
    ):
        ages = np.clip(np.rint(rng.normal(age_m, age_s, size=count)), 18, 70).astype(int)
        incomes = np.clip(np.rint(rng.normal(inc_m, inc_s, size=count)), 15, 137).astype(int)
        scores = np.clip(np.rint(rng.normal(sp_m, sp_s, size=count)), 1, 99).astype(int)
        genders = np.where(rng.random(size=count) < f_prob, "Female", "Male")
        for age, inc, sc, gen in zip(ages, incomes, scores, genders, strict=True):
            rows.append(
                {
                    "Gender": str(gen),
                    "Age": int(age),
                    "Annual Income (k$)": int(inc),
                    "Spending Score (1-100)": int(sc),
                }
            )

    # Shuffle deterministically and add a few realistic boundary customers
    perm = rng.permutation(len(rows))
    shuffled = [rows[int(i)] for i in perm]
    # Inject 6 deterministic boundary observations near cluster mid-planes for entropy sensitivity
    boundary_overrides = [
        {"Gender": "Female", "Age": 29, "Annual Income (k$)": 44, "Spending Score (1-100)": 68},
        {"Gender": "Male", "Age": 34, "Annual Income (k$)": 44, "Spending Score (1-100)": 68},
        {"Gender": "Female", "Age": 30, "Annual Income (k$)": 42, "Spending Score (1-100)": 66},
        {"Gender": "Male", "Age": 32, "Annual Income (k$)": 45, "Spending Score (1-100)": 69},
        {"Gender": "Female", "Age": 28, "Annual Income (k$)": 43, "Spending Score (1-100)": 65},
        {"Gender": "Male", "Age": 35, "Annual Income (k$)": 46, "Spending Score (1-100)": 67},
    ]
    for idx, override in enumerate(boundary_overrides):
        if idx < len(shuffled):
            shuffled[idx * 30 % len(shuffled)] = override

    df = pd.DataFrame(shuffled)
    df.insert(0, "CustomerID", np.arange(1, len(df) + 1, dtype=int))
    df.to_csv(target, index=False)
    return target


def resolve_dataset_path(
    primary_path: str | Path = "data/raw/Mall_Customers.csv",
    fallback_path: str | Path = "tests/fixtures/mall_customers_sample.csv",
) -> tuple[Path, Literal["primary", "fixture"]]:
    """Resolve primary Kaggle dataset when available, otherwise deterministic fixture."""
    primary = Path(primary_path)
    if primary.exists() and primary.is_file():
        return primary, "primary"
    fallback = Path(fallback_path)
    if not fallback.exists():
        generate_deterministic_fixture(fallback)
    return fallback, "fixture"


def load_raw_dataset(config: AppConfig) -> tuple[pd.DataFrame, Literal["primary", "fixture"], Path]:
    """Load and contract-validate the raw Mall Customers dataset."""
    dataset_path, source_kind = resolve_dataset_path(
        config.data.primary_path, config.data.fallback_path
    )
    raw_df = pd.read_csv(dataset_path)
    validated_df = validate_raw_dataframe(raw_df, min_rows=config.data.min_rows)
    return validated_df, source_kind, dataset_path
