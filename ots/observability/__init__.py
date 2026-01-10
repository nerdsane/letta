"""
Observability exporters for OTS trajectories.

IMPORTANT: These are EXPORT-ONLY integrations.

OTS provides exporters to visualize trajectories in existing observability platforms:
- Langfuse: LLM observability platform
- OpenTelemetry: Standard observability framework

Why export-only (no import from OTel/Langfuse)?
-----------------------------------------------
OTel and Langfuse traces capture WHAT an agent did (tool calls, timings),
but not WHY (reasoning, alternatives considered). OTS's core value is capturing
decisions = tool call + rationale + alternatives. Importing from OTel/Langfuse
would produce degraded trajectories without the reasoning context that enables
context learning.

For OTS's full value:
- Capture trajectories at the framework level (using TrajectoryAdapter)
- Export to Langfuse/OTel for visualization
- Store in OTS format for context learning

Do NOT:
- Import traces from OTel/Langfuse into OTS
- Use these exporters as bidirectional sync
"""

from ots.observability.langfuse import LangfuseExporter, export_to_langfuse
from ots.observability.otel import OTelTrajectoryExporter, export_to_otel, link_trajectory_to_current_span

__all__ = [
    "LangfuseExporter",
    "export_to_langfuse",
    "OTelTrajectoryExporter",
    "export_to_otel",
    "link_trajectory_to_current_span",
]
