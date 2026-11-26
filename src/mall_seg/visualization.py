"""Publication-grade 2D/3D Plotly & Matplotlib exports for customer segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram

from mall_seg.clustering import compute_k_distance_curve
from mall_seg.config import load_config
from mall_seg.schema import NUMERIC_FEATURE_COLUMNS

DARK_BG = "#0B1120"
PANEL_BG = "#111827"
GRID_COLOR = "#1F2937"
TEXT_COLOR = "#E2E8F0"
MUTED_TEXT = "#94A3B8"
PALETTE = ["#38BDF8", "#34D399", "#FBBF24", "#F472B6", "#A78BFA", "#FB7185"]


def _apply_dark_axes(ax: Any) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)


def build_2d_pca_figure(
    scored_df: pd.DataFrame,
    explained_variance: list[float] | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """Build interactive 2D Plotly PCA projection sized by GMM normalized entropy."""
    var_pct = (
        f"PC1 ({explained_variance[0] * 100:.1f}%)"
        if explained_variance and len(explained_variance) >= 2
        else "PC1"
    )
    var_pct_2 = (
        f"PC2 ({explained_variance[1] * 100:.1f}%)"
        if explained_variance and len(explained_variance) >= 2
        else "PC2"
    )
    fig = px.scatter(
        scored_df,
        x="pca_1",
        y="pca_2",
        color="persona_name",
        size="normalized_entropy",
        symbol="is_boundary_customer",
        color_discrete_sequence=PALETTE,
        hover_data=["customer_ref", "age", "annual_income_k", "spending_score", "gmm_confidence"],
        labels={
            "pca_1": var_pct,
            "pca_2": var_pct_2,
            "persona_name": "Persona",
            "is_boundary_customer": "Boundary Flag",
        },
        title="2D PCA Segment Projection (Point Size Scaled by GMM Entropy)",
        template=template,
    )
    return fig


def build_3d_segment_figure(
    scored_df: pd.DataFrame,
    template: str = "plotly_dark",
) -> go.Figure:
    """Build interactive 3D Plotly Age x Income x Spending customer scatter."""
    fig_3d = px.scatter_3d(
        scored_df,
        x="age",
        y="annual_income_k",
        z="spending_score",
        color="persona_name",
        symbol="is_boundary_customer",
        size="gmm_confidence",
        size_max=14,
        color_discrete_sequence=PALETTE,
        hover_data={
            "customer_ref": True,
            "gender": True,
            "gmm_confidence": ":.3f",
            "normalized_entropy": ":.3f",
            "recommended_action": True,
        },
        labels={
            "age": "Age",
            "annual_income_k": "Annual Income (k$)",
            "spending_score": "Spending Score (1-100)",
            "persona_name": "Segment Persona",
            "is_boundary_customer": "Boundary Flag",
        },
        title="3D Mall Customer Behavioral Segments & Boundary Uncertainty",
        template=template,
    )
    return fig_3d


def build_k_distance_figure(
    X_scaled: np.ndarray,
    min_samples: int = 5,
    template: str = "plotly_white",
) -> go.Figure:
    """Build interactive DBSCAN sorted k-NN distance knee diagnostic figure."""
    curve = compute_k_distance_curve(X_scaled, min_samples=min_samples)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve.sorted_indices,
            y=curve.k_distances,
            mode="lines",
            name=f"{curve.min_samples}-NN Distance",
            line=dict(color="#1D4ED8", width=2.4),
        )
    )
    fig.add_hline(
        y=curve.recommended_eps,
        line_dash="dash",
        line_color="#BE123C",
        annotation_text=f"Knee eps = {curve.recommended_eps:.4f}",
    )
    fig.update_layout(
        title=f"DBSCAN k-Distance Knee Tuner (k={curve.min_samples})",
        xaxis_title="Customers Sorted by k-NN Distance",
        yaxis_title=f"{curve.min_samples}-NN Distance (Standardized Space)",
        template=template,
    )
    return fig


def render_all_visuals(
    config_path: str | None = None,
    bundle: dict[str, Any] | None = None,
) -> list[Path]:
    """Render static PNG and interactive HTML visual artifacts from a model bundle."""
    cfg = load_config(config_path)
    if bundle is None:
        bundle_path = Path(cfg.artifacts.model_bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. Run training pipeline first."
            )
        loaded = joblib.load(bundle_path)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected dict bundle at {bundle_path}")
        bundle = loaded

    out_dir = Path(cfg.artifacts.visuals_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(bundle["scored_rows"])
    df_features = df.rename(
        columns={
            "age": "Age",
            "annual_income_k": "Annual Income (k$)",
            "spending_score": "Spending Score (1-100)",
        }
    )

    generated: list[Path] = []

    # 1. EDA Distributions & Correlation Matrix
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=DARK_BG)
    fig.suptitle(
        "Mall Customer Feature Distributions & Pearson Correlations",
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    hist_specs = [
        ("Age", "#38BDF8", axes[0, 0]),
        ("Annual Income (k$)", "#34D399", axes[0, 1]),
        ("Spending Score (1-100)", "#FBBF24", axes[1, 0]),
    ]
    for col, color, ax in hist_specs:
        _apply_dark_axes(ax)
        vals = df_features[col].astype(float)
        ax.hist(vals, bins=18, color=color, alpha=0.85, edgecolor=DARK_BG)
        ax.axvline(vals.mean(), color="#F8FAFC", linestyle="--", linewidth=1.2, label="Mean")
        ax.set_title(f"{col} (mean={vals.mean():.1f}, sd={vals.std(ddof=1):.1f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Customer Count")
        ax.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=8)

    ax_corr = axes[1, 1]
    _apply_dark_axes(ax_corr)
    ax_corr.grid(False)
    corr_mat = df_features[NUMERIC_FEATURE_COLUMNS].corr(method="pearson").to_numpy()
    im = ax_corr.imshow(corr_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    short_labels = ["Age", "Income (k$)", "Spending"]
    ax_corr.set_xticks(range(3), labels=short_labels, color=TEXT_COLOR)
    ax_corr.set_yticks(range(3), labels=short_labels, color=TEXT_COLOR)
    ax_corr.set_title("Pearson Correlation Matrix")
    for i in range(3):
        for j in range(3):
            val = float(corr_mat[i, j])
            txt_color = "#0F172A" if (abs(val) < 0.35 or abs(val) > 0.55) else TEXT_COLOR
            ax_corr.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=10,
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED_TEXT)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    p1 = out_dir / "01_eda_distributions.png"
    fig.savefig(p1, dpi=160, facecolor=DARK_BG)
    plt.close(fig)
    generated.append(p1)

    # 2. K-Means Elbow / Silhouette + GMM BIC Sweep
    km_sweep = bundle["kmeans_sweep"]
    gmm_sweep = bundle["gmm_sweep"]
    ks = [pt["k"] for pt in km_sweep]
    inertias = [pt["inertia"] for pt in km_sweep]
    sils = [pt["silhouette"] for pt in km_sweep]
    gmm_ks = [pt["k"] for pt in gmm_sweep]
    bics = [pt["bic"] for pt in gmm_sweep]
    target_k = int(bundle["config"]["clustering"]["target_k"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=DARK_BG)
    fig.suptitle(
        "Model Selection Sweep (K-Means Elbow & Silhouette, GMM BIC)",
        color=TEXT_COLOR,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    panels = [
        (ks, inertias, "K-Means Inertia (Elbow)", "Inertia", "#38BDF8", axes[0]),
        (ks, sils, "K-Means Silhouette Score", "Silhouette", "#34D399", axes[1]),
        (gmm_ks, bics, "GMM BIC Curve", "BIC (Lower Better)", "#FBBF24", axes[2]),
    ]
    for x_vals, y_vals, title, ylabel, color, ax in panels:
        _apply_dark_axes(ax)
        ax.plot(x_vals, y_vals, marker="o", linewidth=2.2, color=color)
        ax.axvline(target_k, color="#F472B6", linestyle="--", linewidth=1.3, label=f"k={target_k}")
        ax.set_title(title)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_vals)
        ax.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=8)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    p2 = out_dir / "02_kmeans_sweep.png"
    fig.savefig(p2, dpi=160, facecolor=DARK_BG)
    plt.close(fig)
    generated.append(p2)

    # 3. Ward Hierarchical Dendrogram
    linkage_mat = np.asarray(bundle["ward_linkage_matrix"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=DARK_BG)
    _apply_dark_axes(ax)
    ax.grid(axis="x", visible=False)
    cut_height = float((linkage_mat[-(target_k - 1), 2] + linkage_mat[-target_k, 2]) / 2.0)
    dendrogram(
        linkage_mat,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=45.0,
        leaf_font_size=8.5,
        show_contracted=True,
        color_threshold=cut_height,
    )
    ax.axhline(
        cut_height,
        color="#F472B6",
        linestyle="--",
        linewidth=1.5,
        label=f"Cut for k={target_k} (Ward dist={cut_height:.2f})",
    )
    ax.set_title("Ward Hierarchical Clustering Dendrogram (Top 30 Merges)")
    ax.set_xlabel("Cluster Leaf / Sample Count")
    ax.set_ylabel("Ward Linkage Distance")
    ax.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    p3 = out_dir / "03_ward_dendrogram.png"
    fig.savefig(p3, dpi=160, facecolor=DARK_BG)
    plt.close(fig)
    generated.append(p3)

    # 4. 2D PCA Clusters Sized by GMM Normalized Entropy
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
    _apply_dark_axes(ax)
    var_ratio = bundle["feature_summary"]["explained_variance_ratio"]
    unique_personas = sorted(df["persona_name"].unique())
    for idx, pname in enumerate(unique_personas):
        sub = df[df["persona_name"] == pname]
        color = PALETTE[idx % len(PALETTE)]
        sizes = 45.0 + 260.0 * sub["normalized_entropy"].to_numpy()
        ax.scatter(
            sub["pca_1"],
            sub["pca_2"],
            s=sizes,
            color=color,
            alpha=0.82,
            edgecolors="#0F172A",
            linewidth=0.8,
            label=f"{pname} (n={len(sub)})",
        )
        ax.scatter(
            [sub["pca_1"].mean()],
            [sub["pca_2"].mean()],
            s=220,
            marker="X",
            color=color,
            edgecolors="#FFFFFF",
            linewidth=1.3,
            zorder=5,
        )

    boundary_df = df[df["is_boundary_customer"]]
    if not boundary_df.empty:
        ax.scatter(
            boundary_df["pca_1"],
            boundary_df["pca_2"],
            s=220,
            facecolors="none",
            edgecolors="#F8FAFC",
            linewidth=1.4,
            linestyle="--",
            label=f"Boundary Holdout (n={len(boundary_df)})",
            zorder=6,
        )

    ax.set_title("2D PCA Segment Projection (Point Size Scaled by GMM Normalized Entropy)")
    ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% variance)")
    ax.legend(
        facecolor=PANEL_BG,
        edgecolor="#334155",
        labelcolor=TEXT_COLOR,
        fontsize=8.5,
        loc="best",
    )

    plt.tight_layout()
    p4 = out_dir / "04_pca_clusters.png"
    fig.savefig(p4, dpi=160, facecolor=DARK_BG)
    plt.close(fig)
    generated.append(p4)

    # 5. Interactive 3D Segment Scatter HTML (Plotly dark theme)
    fig_3d = build_3d_segment_figure(df, template="plotly_dark")
    fig_3d.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        legend=dict(bgcolor="rgba(17,24,39,0.85)", bordercolor="#334155", borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=48),
    )
    p5 = out_dir / "05_segments_3d.html"
    fig_3d.write_html(str(p5), include_plotlyjs="cdn")
    generated.append(p5)

    return generated


__all__ = [
    "build_2d_pca_figure",
    "build_3d_segment_figure",
    "build_k_distance_figure",
    "render_all_visuals",
]
