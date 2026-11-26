"""Add governance fields to model_versions, batch_id tracking, and rolling drift metadata."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002_registry_batch_drift"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    mv_cols = {col["name"] for col in inspector.get_columns("model_versions")}
    if "status" not in mv_cols:
        op.add_column(
            "model_versions",
            sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        )
    if "parent_version" not in mv_cols:
        op.add_column(
            "model_versions",
            sa.Column("parent_version", sa.String(length=64), nullable=True),
        )
    if "notes" not in mv_cols:
        op.add_column("model_versions", sa.Column("notes", sa.Text(), nullable=True))
    if "promoted_at" not in mv_cols:
        op.add_column(
            "model_versions",
            sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        )

    sc_cols = {col["name"] for col in inspector.get_columns("scored_customers")}
    if "batch_id" not in sc_cols:
        op.add_column(
            "scored_customers",
            sa.Column("batch_id", sa.String(length=64), nullable=True),
        )

    se_cols = {col["name"] for col in inspector.get_columns("scoring_events")}
    if "batch_id" not in se_cols:
        op.add_column(
            "scoring_events",
            sa.Column("batch_id", sa.String(length=64), nullable=True),
        )

    dr_cols = {col["name"] for col in inspector.get_columns("drift_reports")}
    if "max_ks_statistic" not in dr_cols:
        op.add_column(
            "drift_reports",
            sa.Column("max_ks_statistic", sa.Float(), nullable=False, server_default="0.0"),
        )
    if "window_label" not in dr_cols:
        op.add_column(
            "drift_reports",
            sa.Column("window_label", sa.String(length=64), nullable=False, server_default="batch"),
        )
    if "alert_status" not in dr_cols:
        op.add_column(
            "drift_reports",
            sa.Column("alert_status", sa.String(length=24), nullable=False, server_default="NORMAL"),
        )
    if "alert_dispatched" not in dr_cols:
        op.add_column(
            "drift_reports",
            sa.Column("alert_dispatched", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    for col in ("alert_dispatched", "alert_status", "window_label", "max_ks_statistic"):
        op.drop_column("drift_reports", col)
    op.drop_column("scoring_events", "batch_id")
    op.drop_column("scored_customers", "batch_id")
    for col in ("promoted_at", "notes", "parent_version", "status"):
        op.drop_column("model_versions", col)
