"""
Open Trajectory Specification (OTS) implementation.

OTS is a decision trace format for agent continual learning.
This module provides:
- Pydantic models for OTS format
- Adapter to convert Letta trajectories to OTS
- Decision extractor for extracting decisions from turns
- Storage layer for OTS trajectories and annotations
- DSF-specific entity extraction
- Context learning retrieval for agent improvement
"""

from letta.trajectories.ots.models import (
    OTSTrajectory,
    OTSTurn,
    OTSMessage,
    OTSDecision,
    OTSAnnotation,
    OTSEntity,
    OTSMetadata,
    OTSContext,
)
from letta.trajectories.ots.adapter import OTSAdapter
from letta.trajectories.ots.decision_extractor import DecisionExtractor
from letta.trajectories.ots.store import OTSStore, store_ots_trajectory, get_ots_trajectory
from letta.trajectories.ots.decision_embeddings import (
    DecisionEmbedder,
    DecisionSearcher,
    embed_decision,
    find_similar_decisions,
)
from letta.trajectories.ots.dsf_entity_extractor import (
    DSFEntityExtractor,
    DSFEvaluationIntegrator,
    extract_dsf_entities,
    enrich_trajectory_context,
)
from letta.trajectories.ots.context_learning import (
    DSFContextLearning,
    RetrievedExample,
    ContextLearningResult,
    get_dsf_context,
    get_anti_patterns,
)

__all__ = [
    # Models
    "OTSTrajectory",
    "OTSTurn",
    "OTSMessage",
    "OTSDecision",
    "OTSAnnotation",
    "OTSEntity",
    "OTSMetadata",
    "OTSContext",
    # Adapter
    "OTSAdapter",
    # Decision extraction
    "DecisionExtractor",
    # Storage
    "OTSStore",
    "store_ots_trajectory",
    "get_ots_trajectory",
    # Decision embeddings
    "DecisionEmbedder",
    "DecisionSearcher",
    "embed_decision",
    "find_similar_decisions",
    # DSF Integration
    "DSFEntityExtractor",
    "DSFEvaluationIntegrator",
    "extract_dsf_entities",
    "enrich_trajectory_context",
    # Context Learning
    "DSFContextLearning",
    "RetrievedExample",
    "ContextLearningResult",
    "get_dsf_context",
    "get_anti_patterns",
]
