"""
Letta Trajectories module.

Provides:
- OTS (Open Trajectory Specification) format support
- Decision extraction from agent execution
- Annotation management for evaluations
"""

from letta.trajectories.ots import (
    OTSAdapter,
    OTSAnnotation,
    OTSContext,
    OTSDecision,
    OTSEntity,
    OTSMessage,
    OTSMetadata,
    OTSTrajectory,
    OTSTurn,
    DecisionExtractor,
)

__all__ = [
    "OTSAdapter",
    "OTSAnnotation",
    "OTSContext",
    "OTSDecision",
    "OTSEntity",
    "OTSMessage",
    "OTSMetadata",
    "OTSTrajectory",
    "OTSTurn",
    "DecisionExtractor",
]
