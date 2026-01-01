"""add trajectories table for continual learning

Revision ID: 56e2a174be96
Revises: 39577145c45d
Create Date: 2026-01-01 16:25:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op
from letta.constants import MAX_EMBEDDING_DIM
from letta.settings import settings

# revision identifiers, used by Alembic.
revision: str = "56e2a174be96"
down_revision: Union[str, None] = "39577145c45d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create trajectories table for capturing agent execution traces.

    Trajectories enable continual learning by storing what agents DID:
    - decisions, reasoning, actions, results
    - LLM-generated summaries and scores
    - pgvector embeddings for similarity search
    """
    # Only create pgvector column for Postgres
    if settings.letta_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        op.create_table(
            "trajectories",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("agent_id", sa.String(), nullable=False),
            sa.Column("organization_id", sa.String(), nullable=True),
            sa.Column("data", sa.JSON(), nullable=False),
            sa.Column("searchable_summary", sa.Text(), nullable=True),
            sa.Column("outcome_score", sa.Float(), nullable=True),
            sa.Column("score_reasoning", sa.Text(), nullable=True),
            sa.Column("embedding", Vector(MAX_EMBEDDING_DIM), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
            sa.Column("_created_by_id", sa.String(), nullable=True),
            sa.Column("_last_updated_by_id", sa.String(), nullable=True),
            sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], name="fk_trajectories_agent_id"),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], name="fk_trajectories_organization_id"),
            sa.PrimaryKeyConstraint("id", name="pk_trajectories"),
        )

        # Create indexes
        op.create_index("ix_trajectories_agent_id", "trajectories", ["agent_id"])
        op.create_index("ix_trajectories_organization_id", "trajectories", ["organization_id"])
        op.create_index("ix_trajectories_created_at", "trajectories", ["created_at", "id"])
        op.create_index("ix_trajectories_outcome_score", "trajectories", ["outcome_score"])
    else:
        # SQLite version (no pgvector)
        op.create_table(
            "trajectories",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("agent_id", sa.String(), nullable=False),
            sa.Column("organization_id", sa.String(), nullable=True),
            sa.Column("data", sa.JSON(), nullable=False),
            sa.Column("searchable_summary", sa.Text(), nullable=True),
            sa.Column("outcome_score", sa.Float(), nullable=True),
            sa.Column("score_reasoning", sa.Text(), nullable=True),
            sa.Column("embedding", sa.LargeBinary(), nullable=True),  # Fallback for SQLite
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("0"), nullable=False),
            sa.Column("_created_by_id", sa.String(), nullable=True),
            sa.Column("_last_updated_by_id", sa.String(), nullable=True),
            sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], name="fk_trajectories_agent_id"),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], name="fk_trajectories_organization_id"),
            sa.PrimaryKeyConstraint("id", name="pk_trajectories"),
        )

        # Create indexes
        op.create_index("ix_trajectories_agent_id", "trajectories", ["agent_id"])
        op.create_index("ix_trajectories_organization_id", "trajectories", ["organization_id"])
        op.create_index("ix_trajectories_created_at", "trajectories", ["created_at", "id"])
        op.create_index("ix_trajectories_outcome_score", "trajectories", ["outcome_score"])


def downgrade() -> None:
    """Drop trajectories table and all related indexes."""
    op.drop_index("ix_trajectories_outcome_score", table_name="trajectories")
    op.drop_index("ix_trajectories_created_at", table_name="trajectories")
    op.drop_index("ix_trajectories_organization_id", table_name="trajectories")
    op.drop_index("ix_trajectories_agent_id", table_name="trajectories")
    op.drop_table("trajectories")
