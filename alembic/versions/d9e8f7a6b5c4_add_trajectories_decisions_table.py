"""add trajectories_decisions table for decision-level embeddings

Revision ID: d9e8f7a6b5c4
Revises: c8d9e0f1a2b3
Create Date: 2026-01-10 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d9e8f7a6b5c4"
down_revision: Union[str, None] = "c8d9e0f1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add trajectories_decisions table for decision-level semantic search.

    This table stores individual decisions extracted from trajectories,
    each with its own embedding for fine-grained similarity search.

    Use cases:
    - Find similar tool call patterns across trajectories
    - Search for decisions with similar context/state
    - Learn from specific decision outcomes (success/failure)
    """
    # Check if PostgreSQL (for vector type)
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # Create the table
    op.create_table(
        "trajectories_decisions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "trajectory_id",
            sa.String(),
            sa.ForeignKey("trajectories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(), nullable=False),
        # Position
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("decision_index", sa.Integer(), nullable=False),
        # Decision content
        sa.Column("action", sa.String(256), nullable=False),
        sa.Column("decision_type", sa.String(64), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        # Outcome
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_type", sa.String(256), nullable=True),
        # Search
        sa.Column("searchable_text", sa.Text(), nullable=False),
        # Timestamps
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Add embedding column based on database type
    if is_postgres:
        # Use pgvector for PostgreSQL
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.add_column(
            "trajectories_decisions",
            sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=True),
        )
    else:
        # Use JSON for SQLite
        op.add_column(
            "trajectories_decisions",
            sa.Column("embedding", sa.JSON(), nullable=True),
        )

    # Create indexes for common queries
    op.create_index("ix_td_trajectory_id", "trajectories_decisions", ["trajectory_id"])
    op.create_index("ix_td_organization_id", "trajectories_decisions", ["organization_id"])
    op.create_index("ix_td_action", "trajectories_decisions", ["action"])
    op.create_index("ix_td_success", "trajectories_decisions", ["success"])
    op.create_index("ix_td_org_action", "trajectories_decisions", ["organization_id", "action"])
    op.create_index("ix_td_org_success", "trajectories_decisions", ["organization_id", "success"])
    op.create_index(
        "ix_td_traj_turn_decision",
        "trajectories_decisions",
        ["trajectory_id", "turn_index", "decision_index"],
        unique=True,
    )

    # Add vector index for PostgreSQL (for similarity search)
    if is_postgres:
        # Create IVFFlat index for approximate nearest neighbor search
        # Note: This requires data to be populated first for optimal performance
        op.execute(
            """
            CREATE INDEX ix_td_embedding
            ON trajectories_decisions
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )


def downgrade() -> None:
    """Remove trajectories_decisions table."""
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # Drop indexes
    if is_postgres:
        op.execute("DROP INDEX IF EXISTS ix_td_embedding")

    op.drop_index("ix_td_traj_turn_decision", table_name="trajectories_decisions")
    op.drop_index("ix_td_org_success", table_name="trajectories_decisions")
    op.drop_index("ix_td_org_action", table_name="trajectories_decisions")
    op.drop_index("ix_td_success", table_name="trajectories_decisions")
    op.drop_index("ix_td_action", table_name="trajectories_decisions")
    op.drop_index("ix_td_organization_id", table_name="trajectories_decisions")
    op.drop_index("ix_td_trajectory_id", table_name="trajectories_decisions")

    # Drop table
    op.drop_table("trajectories_decisions")
