"""
Trajectory Pydantic schemas for API requests/responses.

Trajectories capture agent execution traces for continual learning.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from letta.helpers.datetime_helpers import get_utc_time
from letta.schemas.enums import PrimitiveType
from letta.schemas.letta_base import LettaBase


class TrajectoryBase(LettaBase):
    """Base class for trajectory schemas"""

    __id_prefix__ = PrimitiveType.TRAJECTORY.value


class Trajectory(TrajectoryBase):
    """
    A complete record of an agent's execution.

    Flexible JSONB structure allows any agent type to store relevant data:
    - Story creation: phases, questions, worlds, validation
    - Code generation: spec, design, implementation, tests
    - Research: queries, sources, analysis, findings
    - Customer support: issue, investigation, resolution

    The data field is intentionally unstructured to support diverse agent types.
    """

    id: str = TrajectoryBase.generate_id_field()

    # Agent relationship
    agent_id: str = Field(..., description="The unique identifier of the agent that generated this trajectory")

    # Core trajectory data (flexible)
    data: Dict[str, Any] = Field(..., description="Complete trajectory data with flexible structure")

    # LLM-generated content
    searchable_summary: Optional[str] = Field(None, description="Natural language summary for search (LLM-generated)")
    outcome_score: Optional[float] = Field(None, description="Quality score 0-1 (LLM-generated)", ge=0.0, le=1.0)
    score_reasoning: Optional[str] = Field(None, description="Explanation of the outcome score (LLM-generated)")

    # LLM-extracted labels and metadata
    tags: Optional[List[str]] = Field(None, description="Semantic tags for filtering (LLM-generated)")
    task_category: Optional[str] = Field(None, description="Primary task classification (LLM-generated)")
    complexity_level: Optional[str] = Field(None, description="Task complexity (LLM-generated)")
    trajectory_metadata: Optional[Dict[str, Any]] = Field(None, description="Flexible metadata extracted by LLM")

    # Vector embedding (not included in API responses - internal only)
    # embedding: Optional[List[float]] = Field(None, exclude=True)

    # Async processing status
    processing_status: str = Field("pending", description="Processing status: pending, processing, completed, failed")
    processing_started_at: Optional[datetime] = Field(None, description="When LLM processing started")
    processing_completed_at: Optional[datetime] = Field(None, description="When LLM processing completed")
    processing_error: Optional[str] = Field(None, description="Error message if processing failed")

    # Timestamps
    created_at: datetime = Field(default_factory=get_utc_time, description="When the trajectory was created")
    updated_at: Optional[datetime] = Field(None, description="When the trajectory was last updated")

    # Organization context
    organization_id: Optional[str] = Field(None, description="Organization that owns this trajectory")

    model_config = ConfigDict(populate_by_name=True)


class TrajectoryCreate(TrajectoryBase):
    """
    Request model for creating a new trajectory.

    Start with minimal data, add more as execution progresses.
    Processing (summary, scoring, embedding) happens asynchronously.
    """

    agent_id: str = Field(..., description="The agent generating this trajectory")
    data: Dict[str, Any] = Field(..., description="Initial trajectory data")

    model_config = ConfigDict(extra="forbid")


class TrajectoryUpdate(TrajectoryBase):
    """
    Request model for updating an existing trajectory.

    Typically used to:
    - Add more turns as execution progresses
    - Update outcome data when complete
    - Trigger reprocessing of summary/score
    """

    data: Optional[Dict[str, Any]] = Field(None, description="Updated trajectory data")
    searchable_summary: Optional[str] = Field(None, description="Manually override the LLM-generated summary")
    outcome_score: Optional[float] = Field(None, description="Manually override the outcome score", ge=0.0, le=1.0)
    score_reasoning: Optional[str] = Field(None, description="Manually override the score reasoning")
    tags: Optional[List[str]] = Field(None, description="Manually override LLM-generated tags")
    task_category: Optional[str] = Field(None, description="Manually override task classification")
    complexity_level: Optional[str] = Field(None, description="Manually override complexity level")
    trajectory_metadata: Optional[Dict[str, Any]] = Field(None, description="Manually override or add metadata")

    model_config = ConfigDict(extra="ignore")


class TrajectorySearchRequest(LettaBase):
    """
    Request model for searching trajectories by similarity.

    Find trajectories where the agent handled similar situations.
    """

    query: str = Field(..., description="Natural language query describing what you're looking for")
    agent_id: Optional[str] = Field(None, description="Filter to specific agent")
    min_score: Optional[float] = Field(
        None, description="Minimum outcome score (0-1) - use to filter for successful trajectories", ge=0.0, le=1.0
    )
    max_score: Optional[float] = Field(
        None, description="Maximum outcome score (0-1) - use to find failures/anti-patterns", ge=0.0, le=1.0
    )
    tags: Optional[List[str]] = Field(None, description="Filter by tags (trajectories must have all specified tags)")
    task_category: Optional[str] = Field(None, description="Filter by task category")
    complexity_level: Optional[str] = Field(None, description="Filter by complexity level")
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=100)

    model_config = ConfigDict(extra="forbid")


class TrajectorySearchResult(LettaBase):
    """
    A single search result with similarity score.
    """

    trajectory: Trajectory = Field(..., description="The matching trajectory")
    similarity: float = Field(..., description="Cosine similarity to the query (0-1, higher is more similar)", ge=0.0, le=1.0)


class TrajectorySearchResponse(LettaBase):
    """
    Response model for trajectory search.
    """

    results: List[TrajectorySearchResult] = Field(..., description="Matching trajectories ordered by similarity")
    query: str = Field(..., description="The original search query")
