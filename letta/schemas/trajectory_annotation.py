"""
Trajectory Annotation Pydantic schemas for API requests/responses.

Annotations are linked evaluations for trajectories, turns, or decisions.
They follow the Open Trajectory Specification (OTS) pattern of separating
evaluations from trajectory data.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from letta.helpers.datetime_helpers import get_utc_time
from letta.schemas.enums import PrimitiveType
from letta.schemas.letta_base import LettaBase


class EvaluatorType(str, Enum):
    """Types of evaluators that can annotate trajectories."""
    HUMAN = "human"           # Human annotator
    MODEL = "model"           # LLM-based evaluation
    HEURISTIC = "heuristic"   # Programmatic rules


class AnnotationLabel(str, Enum):
    """Common annotation labels."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    AMBIGUOUS = "ambiguous"
    PIVOTAL = "pivotal"       # Critical decision point
    SUBOPTIMAL = "suboptimal"


class TrajectoryAnnotationBase(LettaBase):
    """Base class for annotation schemas."""
    __id_prefix__ = PrimitiveType.ANNOTATION.value


class TrajectoryAnnotation(TrajectoryAnnotationBase):
    """
    A linked evaluation for a trajectory, turn, or decision.

    Annotations are separate from trajectories for:
    - Multiple evaluators can annotate the same trajectory
    - Annotations can be added retroactively
    - Different retention policies for trajectories vs annotations

    Granularity:
    - trajectory_id only: Trajectory-level (overall quality)
    - trajectory_id + turn_id: Turn-level (step quality)
    - trajectory_id + turn_id + decision_id: Decision-level (choice quality)
    """

    id: str = TrajectoryAnnotationBase.generate_id_field()

    # Target
    trajectory_id: str = Field(..., description="ID of the trajectory being annotated")
    turn_id: Optional[int] = Field(None, description="Turn index (null = trajectory-level)")
    decision_id: Optional[str] = Field(None, description="Decision ID (null = turn-level)")

    # Evaluator
    evaluator_id: str = Field(..., description="Evaluator identifier (e.g., 'human:user123', 'model:gpt-4')")
    evaluator_type: EvaluatorType = Field(..., description="Type of evaluator")
    evaluator_version: Optional[str] = Field(None, description="Version of the evaluator")

    # Evaluation
    score: float = Field(..., description="Quality score 0-1 (higher is better)", ge=0.0, le=1.0)
    label: Optional[str] = Field(None, description="Categorical label")
    feedback: Optional[str] = Field(None, description="Detailed feedback or explanation")
    criteria_scores: Optional[Dict[str, float]] = Field(
        None, description="Per-criterion scores (e.g., {'accuracy': 0.9, 'completeness': 0.8})"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=get_utc_time)

    # Organization
    organization_id: Optional[str] = Field(None, description="Organization that owns this annotation")

    model_config = ConfigDict(populate_by_name=True)


class TrajectoryAnnotationCreate(TrajectoryAnnotationBase):
    """
    Request model for creating a new annotation.
    """

    trajectory_id: str = Field(..., description="ID of the trajectory to annotate")
    turn_id: Optional[int] = Field(None, description="Turn index for turn-level annotation")
    decision_id: Optional[str] = Field(None, description="Decision ID for decision-level annotation")

    evaluator_id: str = Field(..., description="Evaluator identifier")
    evaluator_type: EvaluatorType = Field(..., description="Type of evaluator")
    evaluator_version: Optional[str] = Field(None, description="Version of the evaluator")

    score: float = Field(..., description="Quality score 0-1", ge=0.0, le=1.0)
    label: Optional[str] = Field(None, description="Categorical label")
    feedback: Optional[str] = Field(None, description="Detailed feedback")
    criteria_scores: Optional[Dict[str, float]] = Field(None, description="Per-criterion scores")

    model_config = ConfigDict(extra="forbid")


class TrajectoryAnnotationUpdate(TrajectoryAnnotationBase):
    """
    Request model for updating an existing annotation.
    """

    score: Optional[float] = Field(None, description="Updated score", ge=0.0, le=1.0)
    label: Optional[str] = Field(None, description="Updated label")
    feedback: Optional[str] = Field(None, description="Updated feedback")
    criteria_scores: Optional[Dict[str, float]] = Field(None, description="Updated criteria scores")

    model_config = ConfigDict(extra="ignore")


class AnnotationSearchRequest(LettaBase):
    """
    Request model for searching annotations.
    """

    trajectory_id: Optional[str] = Field(None, description="Filter by trajectory")
    turn_id: Optional[int] = Field(None, description="Filter by turn")
    decision_id: Optional[str] = Field(None, description="Filter by decision")
    evaluator_id: Optional[str] = Field(None, description="Filter by evaluator")
    evaluator_type: Optional[EvaluatorType] = Field(None, description="Filter by evaluator type")
    min_score: Optional[float] = Field(None, description="Minimum score", ge=0.0, le=1.0)
    max_score: Optional[float] = Field(None, description="Maximum score", ge=0.0, le=1.0)
    label: Optional[str] = Field(None, description="Filter by label")
    limit: int = Field(50, description="Maximum results", ge=1, le=500)
    offset: int = Field(0, description="Offset for pagination", ge=0)

    model_config = ConfigDict(extra="forbid")


class AnnotationAggregation(LettaBase):
    """
    Aggregated annotation statistics.
    """

    trajectory_id: str
    annotation_count: int
    avg_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    evaluator_count: int
    labels: Dict[str, int] = Field(default_factory=dict, description="Label counts")


class BatchAnnotationCreate(LettaBase):
    """
    Request model for creating multiple annotations at once.
    """

    annotations: List[TrajectoryAnnotationCreate] = Field(
        ..., description="List of annotations to create", min_length=1, max_length=100
    )

    model_config = ConfigDict(extra="forbid")
