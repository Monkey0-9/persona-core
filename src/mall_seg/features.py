"""Feature engineering pipeline with privacy-aware Gender opt-in and 2D PCA projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mall_seg.contracts import NUMERIC_FEATURE_COLUMNS


@dataclass
class FeatureArtifacts:
    X_scaled: np.ndarray
    X_pca: np.ndarray
    feature_names: list[str]
    explained_variance_ratio: list[float]
    pca_loadings: dict[str, list[float]]
    preprocessor: ColumnTransformer
    pca: PCA
    include_gender: bool

    def summary_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "include_gender": self.include_gender,
            "explained_variance_ratio": [round(float(v), 4) for v in self.explained_variance_ratio],
            "cumulative_variance_ratio": round(float(sum(self.explained_variance_ratio)), 4),
            "pca_loadings": self.pca_loadings,
        }


def build_feature_pipeline(
    include_gender: bool = False,
    pca_components: int = 2,
    seed: int = 42,
) -> tuple[ColumnTransformer, PCA]:
    """Construct ColumnTransformer (StandardScaler + opt-in Gender OHE) and PCA."""
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", StandardScaler(), list(NUMERIC_FEATURE_COLUMNS)),
    ]
    if include_gender:
        transformers.append(
            (
                "gender",
                OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"),
                ["Gender"],
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pca = PCA(n_components=pca_components, random_state=seed)
    return preprocessor, pca


def fit_transform_features(
    df_no_id: pd.DataFrame,
    include_gender: bool = False,
    pca_components: int = 2,
    seed: int = 42,
) -> FeatureArtifacts:
    """Fit preprocessor & PCA after enforcing that CustomerID has been stripped."""
    if "CustomerID" in df_no_id.columns:
        raise ValueError("CustomerID must be stripped before feature scaling and clustering.")

    preprocessor, pca = build_feature_pipeline(
        include_gender=include_gender,
        pca_components=pca_components,
        seed=seed,
    )
    X_scaled = np.asarray(preprocessor.fit_transform(df_no_id), dtype=np.float64)
    X_pca = np.asarray(pca.fit_transform(X_scaled), dtype=np.float64)

    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    explained = [float(v) for v in pca.explained_variance_ratio_]
    loadings: dict[str, list[float]] = {
        f"PC{idx + 1}": [round(float(val), 4) for val in comp]
        for idx, comp in enumerate(pca.components_)
    }

    return FeatureArtifacts(
        X_scaled=X_scaled,
        X_pca=X_pca,
        feature_names=feature_names,
        explained_variance_ratio=explained,
        pca_loadings=loadings,
        preprocessor=preprocessor,
        pca=pca,
        include_gender=include_gender,
    )


def transform_features(
    df_no_id: pd.DataFrame,
    preprocessor: ColumnTransformer,
    pca: PCA,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform new customer records using fitted preprocessor and PCA."""
    if "CustomerID" in df_no_id.columns:
        raise ValueError("CustomerID must be stripped before inference transformation.")
    X_scaled = np.asarray(preprocessor.transform(df_no_id), dtype=np.float64)
    X_pca = np.asarray(pca.transform(X_scaled), dtype=np.float64)
    return X_scaled, X_pca
