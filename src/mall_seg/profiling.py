"""Statistical cluster persona and retail KPI generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from mall_seg.eda import EDAReport, compute_eda_report
from mall_seg.personas import (
    CANONICAL_ARCHETYPES,
    PersonaArchetype,
    PersonaProfile,
    ScoredCustomerPrediction,
    execute_segmentation_workflow,
    map_clusters_to_personas,
    score_customer_records,
)


@dataclass(frozen=True)
class SegmentKPI:
    """Executive KPI breakdown per segmented customer persona."""

    cluster_id: int
    persona_name: str
    customer_count: int
    population_share_pct: float
    mean_age: float
    mean_income_k: float
    mean_spending_score: float
    estimated_revenue_index: float
    female_share_pct: float
    male_share_pct: float
    boundary_count: int
    target_channel: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_segment_kpis(
    df_with_gender: pd.DataFrame,
    labels: np.ndarray,
    is_boundary: np.ndarray | None = None,
) -> list[SegmentKPI]:
    """Generate statistical cluster personas and commercial KPI table from labels."""
    personas = map_clusters_to_personas(df_with_gender=df_with_gender, labels=labels)
    kpis: list[SegmentKPI] = []
    for cid in sorted(personas.keys()):
        p = personas[cid]
        mask = labels == cid
        b_cnt = int(np.sum(is_boundary[mask])) if is_boundary is not None else 0
        # Commercial revenue index: Income (k$) * Spending Score / 100 per customer * segment share
        rev_idx = round((p.mean_income * (p.mean_spending / 100.0)) * p.customer_count, 2)
        kpis.append(
            SegmentKPI(
                cluster_id=cid,
                persona_name=p.persona_name,
                customer_count=p.customer_count,
                population_share_pct=round(p.share * 100.0, 2),
                mean_age=p.mean_age,
                mean_income_k=p.mean_income,
                mean_spending_score=p.mean_spending,
                estimated_revenue_index=rev_idx,
                female_share_pct=round(p.female_share * 100.0, 2),
                male_share_pct=round(p.male_share * 100.0, 2),
                boundary_count=b_cnt,
                target_channel=p.target_channel,
                recommended_action=p.recommended_action,
            )
        )
    return sorted(kpis, key=lambda item: item.estimated_revenue_index, reverse=True)


__all__ = [
    "CANONICAL_ARCHETYPES",
    "EDAReport",
    "PersonaArchetype",
    "PersonaProfile",
    "ScoredCustomerPrediction",
    "SegmentKPI",
    "compute_eda_report",
    "execute_segmentation_workflow",
    "generate_segment_kpis",
    "map_clusters_to_personas",
    "score_customer_records",
]
