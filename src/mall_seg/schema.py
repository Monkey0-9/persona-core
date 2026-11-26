"""Strict data contract and domain validation for Mall Customers data."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from mall_seg.contracts import (
    NUMERIC_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_GENDERS,
    build_raw_schema,
    generate_deterministic_fixture,
    load_raw_dataset,
    resolve_dataset_path,
    strip_customer_id,
    validate_raw_dataframe,
)


class MallCustomerRecord(BaseModel):
    """Typed Pydantic schema representing one validated raw customer row."""

    customer_id: int = Field(gt=0)
    gender: Literal["Male", "Female"]
    age: int = Field(ge=16, le=100)
    annual_income_k: float = Field(ge=5.0, le=500.0)
    spending_score: float = Field(ge=1.0, le=100.0)


def validate_domain_invariants(df: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
    """Enforce Pandera schema, non-null invariants, and non-degenerate feature variance."""
    validated = validate_raw_dataframe(df, min_rows=min_rows)
    if validated["CustomerID"].duplicated().any():
        raise ValueError("CustomerID values must be strictly unique.")
    if validated.isna().any().any():
        raise ValueError("Validated customer dataframe must not contain missing values.")
    return validated


def rows_to_dataframe(rows: list[MallCustomerRecord]) -> pd.DataFrame:
    """Convert typed MallCustomerRecord instances into canonical dataset column layout."""
    return pd.DataFrame(
        [
            {
                "CustomerID": r.customer_id,
                "Gender": r.gender,
                "Age": r.age,
                "Annual Income (k$)": r.annual_income_k,
                "Spending Score (1-100)": r.spending_score,
            }
            for r in rows
        ]
    )


__all__ = [
    "MallCustomerRecord",
    "NUMERIC_FEATURE_COLUMNS",
    "REQUIRED_COLUMNS",
    "VALID_GENDERS",
    "build_raw_schema",
    "generate_deterministic_fixture",
    "load_raw_dataset",
    "resolve_dataset_path",
    "rows_to_dataframe",
    "strip_customer_id",
    "validate_domain_invariants",
    "validate_raw_dataframe",
]
