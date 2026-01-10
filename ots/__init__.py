"""
OTS - Open Trajectory Specification

A decision trace format for agent continual learning.

Basic usage:
    from ots import TrajectoryStore, ContextLearning

    store = TrajectoryStore()
    await store.store(trajectory)

    learner = ContextLearning(store)
    context = await learner.get_context("user query")
"""

from ots.models import (
    # Core models
    OTSTrajectory,
    OTSTurn,
    OTSMessage,
    OTSDecision,
    OTSAnnotation,
    OTSEntity,
    # Supporting models
    OTSMetadata,
    OTSContext,
    OTSMessageContent,
    OTSDecisionState,
    OTSChoice,
    OTSConsequence,
    OTSAlternative,
    OTSDecisionEvaluation,
    OTSCreditAssignment,
    OTSEvaluator,
    # Enums
    DecisionType,
    MessageRole,
    ContentType,
    OutcomeType,
    EvaluatorType,
)
from ots.protocols import (
    TrajectoryAdapter,
    StorageBackend,
    EntityExtractor,
    EmbeddingProvider,
    LLMClient,
    SearchResult,
)
from ots.store import (
    TrajectoryStore,
    SQLiteBackend,
    FileBackend,
    MemoryBackend,
)
from ots.extraction import (
    DecisionExtractor,
    ToolEntityExtractor,
)
from ots.learning import (
    ContextLearning,
    RetrievedExample,
    ContextLearningResult,
)

# Privacy module
from ots.privacy import (
    anonymize_trajectory,
    hash_identifier,
    AnonymizationPolicy,
    DefaultAnonymizationPolicy,
    LearningPreservingPolicy,
)

# Management module
from ots.management import (
    DecisionManager,
    DecisionRecord,
    DecisionSearchResult,
)

# Analytics module
from ots.analytics import (
    get_pure_ots_analytics,
    compute_decision_success_rate,
    compute_action_frequency,
    compute_decision_type_breakdown,
    compute_turn_distribution,
    compute_error_type_frequency,
    compute_trajectory_outcomes,
    OTSAnalytics,
)

__version__ = "0.1.0"

__all__ = [
    # Core models
    "OTSTrajectory",
    "OTSTurn",
    "OTSMessage",
    "OTSDecision",
    "OTSAnnotation",
    "OTSEntity",
    # Supporting models
    "OTSMetadata",
    "OTSContext",
    "OTSMessageContent",
    "OTSDecisionState",
    "OTSChoice",
    "OTSConsequence",
    "OTSAlternative",
    "OTSDecisionEvaluation",
    "OTSCreditAssignment",
    "OTSEvaluator",
    # Enums
    "DecisionType",
    "MessageRole",
    "ContentType",
    "OutcomeType",
    "EvaluatorType",
    # Protocols
    "TrajectoryAdapter",
    "StorageBackend",
    "EntityExtractor",
    "EmbeddingProvider",
    "LLMClient",
    "SearchResult",
    # Store
    "TrajectoryStore",
    "SQLiteBackend",
    "FileBackend",
    "MemoryBackend",
    # Extraction
    "DecisionExtractor",
    "ToolEntityExtractor",
    # Learning
    "ContextLearning",
    "RetrievedExample",
    "ContextLearningResult",
    # Privacy
    "anonymize_trajectory",
    "hash_identifier",
    "AnonymizationPolicy",
    "DefaultAnonymizationPolicy",
    "LearningPreservingPolicy",
    # Management
    "DecisionManager",
    "DecisionRecord",
    "DecisionSearchResult",
    # Analytics
    "get_pure_ots_analytics",
    "compute_decision_success_rate",
    "compute_action_frequency",
    "compute_decision_type_breakdown",
    "compute_turn_distribution",
    "compute_error_type_frequency",
    "compute_trajectory_outcomes",
    "OTSAnalytics",
    # Version
    "__version__",
]
