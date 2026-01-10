"""
Pydantic models for Open Trajectory Specification (OTS).

These models define the structure for decision traces that enable
continual learning through display, context learning, simulation, and RL training.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class DecisionType(str, Enum):
    """Types of decisions an agent can make."""
    TOOL_SELECTION = "tool_selection"
    PARAMETER_CHOICE = "parameter_choice"
    REASONING_STEP = "reasoning_step"
    RESPONSE_FORMULATION = "response_formulation"


class EvaluatorType(str, Enum):
    """Types of evaluators for annotations."""
    HUMAN = "human"
    MODEL = "model"
    HEURISTIC = "heuristic"


class OutcomeType(str, Enum):
    """Trajectory outcome types."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


class MessageRole(str, Enum):
    """Message roles in a turn."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, Enum):
    """Content types for messages."""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    WIDGET = "widget"
    DASHBOARD = "dashboard"
    STRUCTURED = "structured"


# === Entity and Resource ===


class OTSEntity(BaseModel):
    """Entity referenced in trajectory context."""
    type: str = Field(..., description="Entity type (e.g., 'tool', 'resource', custom types)")
    id: str = Field(..., description="Entity identifier")
    name: Optional[str] = Field(None, description="Human-readable name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Type-specific attributes")


class OTSResource(BaseModel):
    """Resource accessed during trajectory."""
    type: str = Field(..., description="Resource type (e.g., 'file', 'api', 'database')")
    uri: str = Field(..., description="Resource URI")
    accessed_at: Optional[datetime] = Field(None, description="When resource was accessed")


# === Context ===


class OTSUser(BaseModel):
    """User context."""
    id: str
    handle: Optional[str] = None
    org_id: Optional[str] = None
    teams: Optional[List[str]] = None
    timezone: Optional[str] = None


class OTSContext(BaseModel):
    """Initial context for trajectory."""
    referrer: Optional[str] = Field(None, description="URL or path where agent invoked")
    user: Optional[OTSUser] = None
    entities: List[OTSEntity] = Field(default_factory=list)
    resources: List[OTSResource] = Field(default_factory=list)
    custom_context: Optional[str] = Field(None, description="Framework-specific context")


# === Message ===


class OTSMessageContent(BaseModel):
    """Content of a message."""
    type: ContentType = ContentType.TEXT
    data: Optional[Dict[str, Any]] = None
    text: Optional[str] = None


class OTSVisibility(BaseModel):
    """Visibility controls for a message."""
    send_to_user: bool = True
    persist: bool = True


class OTSContextSnapshot(BaseModel):
    """Context at a specific message."""
    entities: List[str] = Field(default_factory=list, description="Entity IDs active")
    available_tools: List[str] = Field(default_factory=list, description="Tools available")


class OTSMessage(BaseModel):
    """A single message in a turn."""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    timestamp: datetime
    content: OTSMessageContent
    reasoning: Optional[str] = Field(None, description="Chain-of-thought (assistant only)")
    visibility: Optional[OTSVisibility] = None
    context_snapshot: Optional[OTSContextSnapshot] = None


# === Decision ===


class OTSAlternative(BaseModel):
    """An alternative action that was considered."""
    action: str
    rationale: Optional[str] = None
    rejected_reason: Optional[str] = None


class OTSDecisionState(BaseModel):
    """State at the moment of decision."""
    context_summary: Optional[str] = None
    available_actions: List[str] = Field(default_factory=list)


class OTSChoice(BaseModel):
    """The chosen action."""
    action: str
    arguments: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class OTSConsequence(BaseModel):
    """Consequence of a decision."""
    success: bool
    result_summary: Optional[str] = None
    error_type: Optional[str] = None


class OTSCounterfactual(BaseModel):
    """Counterfactual analysis."""
    better_alternative: Optional[str] = None
    estimated_improvement: Optional[float] = None


class OTSDecisionEvaluation(BaseModel):
    """Evaluation of a decision."""
    evaluator_id: str
    score: float = Field(..., ge=0, le=1)
    criteria_scores: Optional[Dict[str, float]] = None
    feedback: Optional[str] = None
    counterfactual: Optional[OTSCounterfactual] = None


class OTSCreditAssignment(BaseModel):
    """Credit assignment for a decision."""
    contribution_to_outcome: float = Field(..., ge=-1, le=1, alias="impact")
    pivotal: bool = False
    explanation: Optional[str] = None

    class Config:
        populate_by_name = True


class OTSDecision(BaseModel):
    """
    An atomic decision point within a turn.

    Captures: state -> alternatives -> choice -> consequence
    """
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_type: DecisionType
    state: Optional[OTSDecisionState] = None
    alternatives: Optional[Dict[str, List[OTSAlternative]]] = None
    choice: OTSChoice
    consequence: OTSConsequence
    evaluation: Optional[OTSDecisionEvaluation] = None
    credit_assignment: Optional[OTSCreditAssignment] = None


# === Turn ===


class OTSTurn(BaseModel):
    """
    One LLM interaction cycle.

    Contains messages and extracted decisions.
    """
    turn_id: int
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: Optional[str] = None
    timestamp: datetime
    duration_ms: Optional[float] = None
    error: bool = False
    turn_reward: Optional[float] = None
    messages: List[OTSMessage] = Field(default_factory=list)
    decisions: List[OTSDecision] = Field(default_factory=list)


# === Metadata ===


class OTSMetadata(BaseModel):
    """Trajectory metadata."""
    task_description: str
    domain: Optional[str] = None
    timestamp_start: datetime
    timestamp_end: Optional[datetime] = None
    duration_ms: Optional[float] = None
    agent_id: str
    framework: Optional[str] = Field(None, description="Agent framework (e.g., 'letta', 'langchain')")
    environment: Optional[str] = None
    outcome: OutcomeType
    feedback_score: Optional[float] = Field(None, ge=0, le=1)
    human_reviewed: bool = False
    tags: List[str] = Field(default_factory=list)
    parent_trajectory_id: Optional[str] = None


# === Trajectory ===


class OTSSystemMessage(BaseModel):
    """System message at trajectory start."""
    content: str
    timestamp: datetime


class OTSTrajectory(BaseModel):
    """
    Open Trajectory Specification (OTS) format.

    A complete record of an agent's execution as a decision trace.
    Enables: display, context learning, simulation, RL training.
    """
    trajectory_id: str = Field(default_factory=lambda: str(uuid4()))
    version: str = "0.1.0"
    metadata: OTSMetadata
    context: OTSContext = Field(default_factory=OTSContext)
    system_message: Optional[OTSSystemMessage] = None
    turns: List[OTSTurn] = Field(default_factory=list)
    final_reward: Optional[float] = Field(None, ge=0, le=1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OTSTrajectory":
        """Create trajectory from dictionary."""
        return cls.model_validate(data)


# === Annotation (Linked Entity) ===


class OTSEvaluator(BaseModel):
    """Evaluator information."""
    id: str
    type: EvaluatorType
    version: Optional[str] = None


class OTSAnnotation(BaseModel):
    """
    Linked annotation for trajectory, turn, or decision.

    Annotations are separate from trajectories for:
    - Multiple evaluators per trajectory
    - Retroactive annotations
    - Different retention policies
    """
    annotation_id: str = Field(default_factory=lambda: str(uuid4()))
    trajectory_id: str
    turn_id: Optional[int] = Field(None, description="null = trajectory-level")
    decision_id: Optional[str] = Field(None, description="null = turn-level")
    evaluator: OTSEvaluator
    score: float = Field(..., ge=0, le=1)
    label: Optional[str] = None
    feedback: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json", exclude_none=True)
