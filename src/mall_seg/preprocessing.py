"""Sklearn ColumnTransformer preprocessing & PCA dimensionality reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from mall_seg.contracts import strip_customer_id
from mall_seg.features import (
    FeatureArtifacts,
    build_feature_pipeline,
    fit_transform_features,
    transform_features,
)


def prepare_and_project_features(
    raw_df: pd.DataFrame,
    include_gender: bool = False,
    pca_components: int = 2,
    seed: int = 42,
) -> FeatureArtifacts:
    """Strip identifiers, fit StandardScaler/opt-in OHE ColumnTransformer, and fit PCA."""
    df_no_id = strip_customer_id(raw_df)
    return fit_transform_features(
        df_no_id=df_no_id,
        include_gender=include_gender,
        pca_components=pca_components,
        seed=seed,
    )


def project_new_records(
    records_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    pca: PCA,
) -> tuple[np.ndarray, np.ndarray]:
    """Strip identifiers if present and transform new customer rows into scaled and PCA spaces."""
    df_no_id = strip_customer_id(records_df)
    return transform_features(df_no_id=df_no_id, preprocessor=preprocessor, pca=pca)


__all__ = [
    "FeatureArtifacts",
    "build_feature_pipeline",
    "fit_transform_features",
    "prepare_and_project_features",
    "project_new_records",
    "transform_features",
]
