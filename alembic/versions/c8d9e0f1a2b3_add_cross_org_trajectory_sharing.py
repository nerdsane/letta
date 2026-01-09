"""add cross-org trajectory sharing with domain_type

Revision ID: c8d9e0f1a2b3
Revises: b4c7e8f9d2a1
Create Date: 2026-01-09 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d9e0f1a2b3"
down_revision: Union[str, None] = "b4c7e8f9d2a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add cross-organization trajectory sharing support.

    Changes:
    1. Add domain_type to agents table (canonical source)
    2. Add domain_type to trajectories table (denormalized for queries)
    3. Add share_cross_org flag to trajectories table (opt-in sharing)
    4. Create indexes for efficient cross-org queries

    This enables trajectories to be shared across organizations by domain_type,
    with privacy preserved through anonymization of cross-org results.
    """
    # Detect database dialect
    conn = op.get_bind()
    is_postgres = conn.dialect.name == "postgresql"

    # 1. Add domain_type to agents table
    op.add_column(
        "agents",
        sa.Column(
            "domain_type",
            sa.String(64),
            nullable=True,
            comment="Custom domain classification for cross-org trajectory sharing"
        )
    )
    op.create_index("ix_agents_domain_type", "agents", ["domain_type"], unique=False)

    # 2. Add domain_type to trajectories table (denormalized from agent)
    op.add_column(
        "trajectories",
        sa.Column(
            "domain_type",
            sa.String(64),
            nullable=True,
            comment="Custom domain type inherited from agent for cross-org sharing"
        )
    )
    op.create_index("ix_trajectories_domain_type", "trajectories", ["domain_type"], unique=False)

    # 3. Add share_cross_org flag to trajectories
    if is_postgres:
        op.add_column(
            "trajectories",
            sa.Column(
                "share_cross_org",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
                comment="Whether this trajectory is visible (anonymized) to other organizations"
            )
        )
    else:
        # SQLite uses 0/1 for booleans
        op.add_column(
            "trajectories",
            sa.Column(
                "share_cross_org",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
                comment="Whether this trajectory is visible (anonymized) to other organizations"
            )
        )

    op.create_index("ix_trajectories_share_cross_org", "trajectories", ["share_cross_org"], unique=False)

    # 4. Create composite index for efficient cross-org queries
    op.create_index(
        "ix_trajectories_domain_share",
        "trajectories",
        ["domain_type", "share_cross_org"],
        unique=False
    )


def downgrade() -> None:
    """Remove cross-org trajectory sharing support."""
    # Drop indexes first
    op.drop_index("ix_trajectories_domain_share", table_name="trajectories")
    op.drop_index("ix_trajectories_share_cross_org", table_name="trajectories")
    op.drop_index("ix_trajectories_domain_type", table_name="trajectories")
    op.drop_index("ix_agents_domain_type", table_name="agents")

    # Drop columns
    op.drop_column("trajectories", "share_cross_org")
    op.drop_column("trajectories", "domain_type")
    op.drop_column("agents", "domain_type")
