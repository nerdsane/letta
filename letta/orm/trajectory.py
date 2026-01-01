"""
Trajectory ORM model for capturing agent execution traces.

Trajectories capture what agents DID - the decisions, reasoning, actions, and outcomes.
This enables context learning (retrieval), reinforcement learning (training), and continual improvement.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.constants import MAX_EMBEDDING_DIM
from letta.orm.custom_columns import CommonVector
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.trajectory import Trajectory as PydanticTrajectory
from letta.settings import DatabaseChoice, settings

if TYPE_CHECKING:
    from letta.orm.agent import Agent
    from letta.orm.organization import Organization


class Trajectory(SqlalchemyBase, OrganizationMixin):
    """
    Trajectories are complete records of agent execution: observations, reasoning, actions, and outcomes.

    They enable:
    - Context Learning: Retrieve similar past decisions as examples
    - Reinforcement Learning: Train on successful trajectories
    - Continual Improvement: Learn patterns to improve system prompts

    Design Philosophy:
    - Generic: Works for any agent type (coding, research, storytelling, support)
    - Flexible: JSONB for extensibility
    - Searchable: pgvector for similarity search
    - Outcome-aware: Track success/failure for learning
    """

    __tablename__ = "trajectories"
    __pydantic_model__ = PydanticTrajectory
    __table_args__ = (
        Index("ix_trajectories_agent_id", "agent_id"),
        Index("ix_trajectories_organization_id", "organization_id"),
        Index("ix_trajectories_created_at", "created_at", "id"),
        Index("ix_trajectories_outcome_score", "outcome_score"),
    )

    # Primary key with trajectory- prefix
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"trajectory-{uuid.uuid4()}")

    # Agent relationship
    agent_id: Mapped[str] = mapped_column(
        String, ForeignKey("agents.id"), nullable=False, doc="The agent that generated this trajectory"
    )

    # Core trajectory data (flexible JSONB)
    data: Mapped[dict] = mapped_column(JSON, nullable=False, doc="Complete trajectory data including turns, metadata, etc.")

    # LLM-generated searchable content
    searchable_summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Natural language summary for search and display (LLM-generated)"
    )

    # Outcome scoring
    outcome_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="Quality score 0-1 (higher is better, LLM-generated)"
    )
    score_reasoning: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Explanation of the outcome score (LLM-generated)"
    )

    # Vector embedding for similarity search (from searchable_summary)
    if settings.database_engine is DatabaseChoice.POSTGRES:
        from pgvector.sqlalchemy import Vector

        embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True, doc="Vector embedding for similarity search")
    else:
        embedding = Column(CommonVector, nullable=True, doc="Vector embedding for similarity search")

    # Async processing status
    processing_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
        doc="Processing status: pending, processing, completed, failed"
    )
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, doc="When LLM processing started"
    )
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, doc="When LLM processing completed"
    )
    processing_error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Error message if processing failed"
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="trajectories")
    organization: Mapped[Optional["Organization"]] = relationship("Organization", back_populates="trajectories")
