"""Initial schema covering users, refresh_tokens, model_versions, pipeline_runs, and scoring."""

from __future__ import annotations

from alembic import op
from mall_seg.db import Base

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
