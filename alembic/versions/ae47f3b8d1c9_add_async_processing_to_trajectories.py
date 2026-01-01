"""add async processing fields to trajectories

Revision ID: ae47f3b8d1c9
Revises: 56e2a174be96
Create Date: 2026-01-01 18:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ae47f3b8d1c9"
down_revision: Union[str, None] = "56e2a174be96"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add async processing fields to trajectories table.

    Enables background processing of trajectories:
    - processing_status: Track processing state (pending, processing, completed, failed)
    - processing_started_at: When LLM processing began
    - processing_completed_at: When LLM processing finished
    - processing_error: Error message if processing failed
    """
    # Add processing status columns
    op.add_column(
        "trajectories",
        sa.Column(
            "processing_status",
            sa.String(),
            nullable=False,
            server_default="pending",
            comment="Processing status: pending, processing, completed, failed"
        )
    )
    op.add_column(
        "trajectories",
        sa.Column(
            "processing_started_at",
            sa.DateTime(),
            nullable=True,
            comment="When LLM processing started"
        )
    )
    op.add_column(
        "trajectories",
        sa.Column(
            "processing_completed_at",
            sa.DateTime(),
            nullable=True,
            comment="When LLM processing completed"
        )
    )
    op.add_column(
        "trajectories",
        sa.Column(
            "processing_error",
            sa.Text(),
            nullable=True,
            comment="Error message if processing failed"
        )
    )

    # Add index for querying pending trajectories
    op.create_index(
        "ix_trajectories_processing_status",
        "trajectories",
        ["processing_status", "created_at"],
        unique=False
    )


def downgrade() -> None:
    """Remove async processing fields from trajectories table."""
    # Drop index first
    op.drop_index("ix_trajectories_processing_status", table_name="trajectories")

    # Drop columns
    op.drop_column("trajectories", "processing_error")
    op.drop_column("trajectories", "processing_completed_at")
    op.drop_column("trajectories", "processing_started_at")
    op.drop_column("trajectories", "processing_status")
