"""
Observability exporters for OTS trajectories.

This module re-exports observability functionality from the OTS library.
Letta uses these exporters to visualize trajectories in external platforms:
- Langfuse: LLM observability platform
- OpenTelemetry: Standard observability framework

See `ots.observability` for implementation details.
"""

# Re-export from canonical OTS library
# This avoids code duplication between letta and ots packages
from ots.observability import (
    LangfuseExporter,
    export_to_langfuse,
    OTelTrajectoryExporter,
    export_to_otel,
    link_trajectory_to_current_span,
)

__all__ = [
    "LangfuseExporter",
    "export_to_langfuse",
    "OTelTrajectoryExporter",
    "export_to_otel",
    "link_trajectory_to_current_span",
]
