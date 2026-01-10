"""
TrajectoryDecision Pydantic schemas for API requests/responses.

Decision-level schemas for fine-grained semantic search across agent decisions.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import ConfigDict, Field

from letta.helpers.datetime_helpers import get_utc_time
from letta.schemas.letta_base import LettaBase


class TrajectoryDecisionBase(LettaBase):
    """Base class for trajectory decision schemas"""

    __id_prefix__ = "decision"


class TrajectoryDecision(TrajectoryDecisionBase):
    """
    An individual decision extracted from a trajectory.

    Decisions represent choice points: the state, action taken,
    reasoning, and outcome. Embeddings enable semantic search
    across decisions to find relevant past examples.
    """

    id: str = TrajectoryDecisionBase.generate_id_field()

    # Parent trajectory
    trajectory_id: str = Field(..., description="Parent trajectory containing this decision")
    organization_id: Optional[str] = Field(None, description="Organization that owns this decision")

    # Position within trajectory
    turn_index: int = Field(..., description="Zero-indexed turn number within trajectory")
    decision_index: int = Field(..., description="Zero-indexed decision number within turn")

    # Decision content
    action: str = Field(..., description="Tool or action name (e.g., 'world_manager', 'story_manager')")
    decision_type: Optional[str] = Field(
        None,
        description="Type: tool_selection, parameter_choice, reasoning_step, response_formulation"
    )
    rationale: Optional[str] = Field(None, description="Agent's reasoning for this decision")

    # Outcome
    success: Optional[bool] = Field(None, description="Whether the decision succeeded")
    result_summary: Optional[str] = Field(None, description="Summary of the decision outcome")
    error_type: Optional[str] = Field(None, description="Error type if decision failed")

    # Search
    searchable_text: str = Field(..., description="Combined text representation for embedding")

    # Timestamps
    created_at: datetime = Field(default_factory=get_utc_time, description="When this decision record was created")

    model_config = ConfigDict(populate_by_name=True)


class TrajectoryDecisionCreate(TrajectoryDecisionBase):
    """
    Request model for creating a new decision record.

    Typically created automatically when trajectories are processed,
    not manually via API.
    """

    trajectory_id: str = Field(..., description="Parent trajectory ID")
    turn_index: int = Field(..., description="Turn index within trajectory")
    decision_index: int = Field(..., description="Decision index within turn")

    action: str = Field(..., description="Action/tool name")
    decision_type: Optional[str] = Field(None)
    rationale: Optional[str] = Field(None)

    success: Optional[bool] = Field(None)
    result_summary: Optional[str] = Field(None)
    error_type: Optional[str] = Field(None)

    searchable_text: str = Field(..., description="Text for embedding generation")

    model_config = ConfigDict(extra="forbid")


class TrajectoryDecisionSearchRequest(LettaBase):
    """Request model for searching decisions by semantic similarity."""

    query: str = Field(..., description="Natural language description to search for")
    action: Optional[str] = Field(None, description="Filter by specific action/tool name")
    success: Optional[bool] = Field(None, description="Filter by success/failure status")
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100)
    min_similarity: float = Field(0.5, description="Minimum similarity threshold", ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class TrajectoryDecisionSearchResult(LettaBase):
    """A single result from decision search."""

    decision: TrajectoryDecision
    similarity: float = Field(..., description="Similarity score (0-1)")
    trajectory_summary: Optional[str] = Field(None, description="Summary of parent trajectory")


class TrajectoryDecisionSearchResponse(LettaBase):
    """Response model for decision search."""

    results: List[TrajectoryDecisionSearchResult]
    query: str
    total_candidates: int = Field(0, description="Total decisions searched")
