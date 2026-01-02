"""add LLM-extracted labels and metadata to trajectories

Revision ID: b4c7e8f9d2a1
Revises: ae47f3b8d1c9
Create Date: 2026-01-01 19:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4c7e8f9d2a1"
down_revision: Union[str, None] = "ae47f3b8d1c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add LLM-extracted labels and metadata fields to trajectories table.

    Enables structured metadata extraction during LLM processing:
    - tags: Semantic labels for filtering (ARRAY for PostgreSQL, JSON for SQLite)
    - task_category: Primary task classification
    - complexity_level: Task difficulty assessment
    - trajectory_metadata: Flexible JSONB for additional extracted information

    This supports pattern detection, dashboard analytics, and better retrieval.
    """
    # Detect database dialect
    conn = op.get_bind()
    is_postgres = conn.dialect.name == "postgresql"

    # Add tags column (ARRAY for PostgreSQL, JSON for SQLite)
    if is_postgres:
        op.add_column(
            "trajectories",
            sa.Column(
                "tags",
                postgresql.ARRAY(sa.String()),
                nullable=True,
                comment="Semantic tags for filtering (LLM-generated)"
            )
        )
    else:
        op.add_column(
            "trajectories",
            sa.Column(
                "tags",
                sa.JSON(),
                nullable=True,
                comment="Semantic tags for filtering (LLM-generated)"
            )
        )

    # Add task_category column
    op.add_column(
        "trajectories",
        sa.Column(
            "task_category",
            sa.String(),
            nullable=True,
            comment="Primary task classification (LLM-generated)"
        )
    )

    # Add complexity_level column
    op.add_column(
        "trajectories",
        sa.Column(
            "complexity_level",
            sa.String(),
            nullable=True,
            comment="Task complexity (LLM-generated)"
        )
    )

    # Add trajectory_metadata column
    op.add_column(
        "trajectories",
        sa.Column(
            "trajectory_metadata",
            sa.JSON(),
            nullable=True,
            comment="Flexible metadata extracted by LLM"
        )
    )

    # Create indexes for efficient querying
    op.create_index(
        "ix_trajectories_task_category",
        "trajectories",
        ["task_category"],
        unique=False
    )

    op.create_index(
        "ix_trajectories_complexity_level",
        "trajectories",
        ["complexity_level"],
        unique=False
    )

    # Create GIN index for tags array (PostgreSQL only)
    if is_postgres:
        op.create_index(
            "ix_trajectories_tags",
            "trajectories",
            ["tags"],
            unique=False,
            postgresql_using="gin"
        )


def downgrade() -> None:
    """Remove LLM-extracted labels and metadata from trajectories table."""
    # Detect database dialect
    conn = op.get_bind()
    is_postgres = conn.dialect.name == "postgresql"

    # Drop indexes first
    if is_postgres:
        op.drop_index("ix_trajectories_tags", table_name="trajectories")

    op.drop_index("ix_trajectories_complexity_level", table_name="trajectories")
    op.drop_index("ix_trajectories_task_category", table_name="trajectories")

    # Drop columns
    op.drop_column("trajectories", "trajectory_metadata")
    op.drop_column("trajectories", "complexity_level")
    op.drop_column("trajectories", "task_category")
    op.drop_column("trajectories", "tags")
