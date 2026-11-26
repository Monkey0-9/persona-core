"""Surrogate Decision Tree explainability for human-readable cluster rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree, export_text

from mall_seg.contracts import NUMERIC_FEATURE_COLUMNS


@dataclass(frozen=True)
class SurrogateRule:
    rule_id: int
    cluster_id: int
    persona_name: str
    support_count: int
    confidence: float
    condition_text: str


@dataclass(frozen=True)
class ExplainabilityReport:
    fidelity_score: float
    max_depth: int
    feature_importances: dict[str, float]
    rules: list[SurrogateRule]
    tree_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity_score": self.fidelity_score,
            "max_depth": self.max_depth,
            "feature_importances": self.feature_importances,
            "rules": [asdict(r) for r in self.rules],
            "tree_text": self.tree_text,
        }


def _extract_tree_rules(
    tree_model: DecisionTreeClassifier,
    feature_names: list[str],
    persona_names: dict[int, str],
) -> list[SurrogateRule]:
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[int(i)] if int(i) != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]
    classes = tree_model.classes_
    rules: list[SurrogateRule] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = float(tree_.threshold[node])
            recurse(int(tree_.children_left[node]), [*conditions, f"{name} <= {threshold:.2f}"])
            recurse(int(tree_.children_right[node]), [*conditions, f"{name} > {threshold:.2f}"])
        else:
            counts = tree_.value[node][0]
            total = int(np.sum(counts))
            best_idx = int(np.argmax(counts))
            cluster_id = int(classes[best_idx])
            confidence = round(float(counts[best_idx] / max(total, 1)), 4)
            cond_str = " AND ".join(conditions) if conditions else "ALL RECORDS"
            p_name = persona_names.get(cluster_id, f"Segment {cluster_id}")
            rules.append(
                SurrogateRule(
                    rule_id=len(rules) + 1,
                    cluster_id=cluster_id,
                    persona_name=p_name,
                    support_count=total,
                    confidence=confidence,
                    condition_text=f"IF {cond_str} THEN {p_name}",
                )
            )

    recurse(0, [])
    return rules


def fit_surrogate_explainer(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    persona_names: dict[int, str],
    max_depth: int = 3,
    seed: int = 42,
) -> ExplainabilityReport:
    """Fit a shallow DecisionTreeClassifier on numeric features to explain segments."""
    feature_cols = list(NUMERIC_FEATURE_COLUMNS)
    X_raw = df_features[feature_cols].astype(float)

    surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    surrogate.fit(X_raw, labels)

    preds = surrogate.predict(X_raw)
    fidelity = round(float(accuracy_score(labels, preds)), 4)

    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_cols, surrogate.feature_importances_, strict=True)
    }
    rules = _extract_tree_rules(surrogate, feature_cols, persona_names)
    tree_text = str(export_text(surrogate, feature_names=feature_cols))

    return ExplainabilityReport(
        fidelity_score=fidelity,
        max_depth=max_depth,
        feature_importances=importances,
        rules=rules,
        tree_text=tree_text,
    )
