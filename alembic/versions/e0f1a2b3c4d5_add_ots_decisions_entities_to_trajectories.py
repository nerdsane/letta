"""add OTS decisions and entities columns to trajectories

Revision ID: e0f1a2b3c4d5
Revises: d9e8f7a6b5c4
Create Date: 2026-01-10 15:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e0f1a2b3c4d5"
down_revision: Union[str, None] = "d9e8f7a6b5c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add OTS LLM-extracted decisions and entities columns to trajectories table.

    These columns store rich data extracted by GPT-5 mini during trajectory processing:
    - ots_decisions: Decisions with rationale, alternatives considered, confidence
    - ots_entities: Entities (services, files, concepts, etc.) mentioned in trajectory

    This enables:
    - Detailed decision analysis with why/what-else information
    - Entity extraction for understanding context and dependencies
    - Better learning from past agent behavior
    """
    # Add ots_decisions column (JSON for both PostgreSQL and SQLite)
    op.add_column(
        "trajectories",
        sa.Column(
            "ots_decisions",
            sa.JSON(),
            nullable=True,
            comment="OTS-style decisions with rationale/alternatives/confidence (LLM-extracted with GPT-5 mini)"
        )
    )

    # Add ots_entities column (JSON for both PostgreSQL and SQLite)
    op.add_column(
        "trajectories",
        sa.Column(
            "ots_entities",
            sa.JSON(),
            nullable=True,
            comment="Entities extracted from trajectory: services, files, users, concepts (LLM + programmatic)"
        )
    )


def downgrade() -> None:
    """Remove OTS decisions and entities columns from trajectories table."""
    op.drop_column("trajectories", "ots_entities")
    op.drop_column("trajectories", "ots_decisions")
