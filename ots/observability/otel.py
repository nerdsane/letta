"""
OpenTelemetry exporter for OTS trajectories (EXPORT-ONLY).

Exports OTS trajectories as OpenTelemetry spans for visualization
in existing observability infrastructure (Jaeger, Datadog, Honeycomb, etc.).

This is a one-way export. It does NOT import OTel traces into OTS, because
OTel traces (even with GenAI semantic conventions) capture WHAT happened
(tool calls, timings) but not WHY (reasoning, alternatives considered).

OTS's core value is decisions = tool call + rationale + alternatives.
Importing from OTel would produce degraded trajectories that can't support
context learning.

Use this when:
- You want to visualize OTS trajectories in your existing OTel backend
- You need trace continuity across agent operations
- Your org mandates OTel for observability

Do NOT use this to:
- Import existing OTel traces into OTS (loses reasoning context)
- Replace OTS storage (use TrajectoryStore for context learning)
"""

import logging
from typing import Any, Optional

from ots.models import (
    OTSDecision,
    OTSTrajectory,
    OTSTurn,
)

logger = logging.getLogger(__name__)


class OTelTrajectoryExporter:
    """
    Exports OTS trajectories as OpenTelemetry spans.

    Integrates with existing OTel infrastructure to provide:
    - Trace continuity across agent operations
    - Decision-level span granularity
    - Standard OTLP export to any backend

    OTS concepts map to OTel:
    - OTSTrajectory -> Root span
    - OTSTurn -> Child span
    - OTSDecision -> Nested span with tool.* attributes
    - Annotations -> Span events

    Requires: pip install ots[otel]

    Example:
        exporter = OTelTrajectoryExporter()
        trace_id = exporter.export_trajectory(ots_trajectory)
    """

    def __init__(self, tracer_name: str = "ots.trajectories") -> None:
        """
        Initialize OTel exporter.

        Args:
            tracer_name: Name for the tracer
        """
        self.tracer_name = tracer_name
        self._tracer = None

    def _get_tracer(self):
        """Get or create tracer."""
        if self._tracer is None:
            try:
                from opentelemetry import trace
                self._tracer = trace.get_tracer(self.tracer_name)
            except ImportError:
                raise ImportError(
                    "opentelemetry packages not installed. Install with: pip install ots[otel]"
                )
        return self._tracer

    def export_trajectory(
        self,
        trajectory: OTSTrajectory,
        parent_context: Optional[Any] = None,
    ) -> str:
        """
        Export trajectory as OTel spans.

        Args:
            trajectory: OTS trajectory to export
            parent_context: Optional parent span context for linking

        Returns:
            Trace ID
        """
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        tracer = self._get_tracer()

        # Start root span for trajectory
        with tracer.start_as_current_span(
            name=f"trajectory:{trajectory.trajectory_id}",
            context=parent_context,
            kind=trace.SpanKind.INTERNAL,
        ) as root_span:
            # Set trajectory attributes
            root_span.set_attribute("trajectory.id", trajectory.trajectory_id)
            root_span.set_attribute("trajectory.version", trajectory.version)

            if trajectory.metadata:
                if trajectory.metadata.domain:
                    root_span.set_attribute("trajectory.domain", trajectory.metadata.domain)
                if trajectory.metadata.agent_id:
                    root_span.set_attribute("agent.id", trajectory.metadata.agent_id)
                if trajectory.metadata.outcome:
                    root_span.set_attribute("trajectory.outcome", trajectory.metadata.outcome.value)

            # Export turns
            for turn in trajectory.turns:
                self._export_turn_span(turn)

            # Set final status
            if trajectory.final_reward is not None:
                root_span.set_attribute("trajectory.final_reward", trajectory.final_reward)
                if trajectory.final_reward >= 0.7:
                    root_span.set_status(Status(StatusCode.OK))
                else:
                    root_span.set_status(Status(StatusCode.ERROR, "Low trajectory score"))

            # Get trace ID
            span_context = root_span.get_span_context()
            return format(span_context.trace_id, '032x')

    def _export_turn_span(self, turn: OTSTurn) -> None:
        """Export turn as child span."""
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        tracer = self._get_tracer()

        with tracer.start_as_current_span(
            name=f"turn:{turn.turn_id}",
            kind=trace.SpanKind.INTERNAL,
        ) as turn_span:
            turn_span.set_attribute("turn.id", turn.turn_id)
            turn_span.set_attribute("turn.span_id", turn.span_id)
            turn_span.set_attribute("turn.message_count", len(turn.messages))
            turn_span.set_attribute("turn.decision_count", len(turn.decisions))

            if turn.duration_ms:
                turn_span.set_attribute("turn.duration_ms", turn.duration_ms)

            if turn.error:
                turn_span.set_status(Status(StatusCode.ERROR, "Turn error"))
            else:
                turn_span.set_status(Status(StatusCode.OK))

            # Export decisions as nested spans
            for decision in turn.decisions:
                self._export_decision_span(decision)

            # Add turn reward as event
            if turn.turn_reward is not None:
                turn_span.add_event(
                    "turn_reward",
                    attributes={"reward": turn.turn_reward},
                )

    def _export_decision_span(self, decision: OTSDecision) -> None:
        """Export decision as nested span."""
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        tracer = self._get_tracer()

        with tracer.start_as_current_span(
            name=f"decision:{decision.choice.action}",
            kind=trace.SpanKind.INTERNAL,
        ) as decision_span:
            decision_span.set_attribute("decision.id", decision.decision_id)
            decision_span.set_attribute("decision.type", decision.decision_type.value)
            decision_span.set_attribute("tool.name", decision.choice.action)

            if decision.choice.confidence:
                decision_span.set_attribute("decision.confidence", decision.choice.confidence)

            if decision.choice.rationale:
                decision_span.set_attribute("decision.rationale", decision.choice.rationale[:500])

            # Set consequence status
            if decision.consequence.success:
                decision_span.set_status(Status(StatusCode.OK))
            else:
                decision_span.set_status(
                    Status(StatusCode.ERROR, decision.consequence.error_type or "Decision failed")
                )
                if decision.consequence.error_type:
                    decision_span.set_attribute("error.type", decision.consequence.error_type)

            if decision.consequence.result_summary:
                decision_span.set_attribute("decision.result", decision.consequence.result_summary[:500])

            # Add evaluation as event
            if decision.evaluation:
                decision_span.add_event(
                    "evaluation",
                    attributes={
                        "score": decision.evaluation.score,
                        "evaluator_id": decision.evaluation.evaluator_id,
                        "feedback": decision.evaluation.feedback[:200] if decision.evaluation.feedback else None,
                    },
                )

            # Add credit assignment as event
            if decision.credit_assignment:
                decision_span.add_event(
                    "credit_assignment",
                    attributes={
                        "impact": decision.credit_assignment.contribution_to_outcome,
                        "pivotal": decision.credit_assignment.pivotal,
                    },
                )


def link_trajectory_to_current_span(trajectory: OTSTrajectory) -> None:
    """
    Link an OTS trajectory to the current OTel span.

    Call this when capturing a trajectory within an existing traced operation
    to maintain span context continuity.

    Args:
        trajectory: OTS trajectory to link
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span:
            return

        span.set_attribute("trajectory.id", trajectory.trajectory_id)
        span.set_attribute("trajectory.turn_count", len(trajectory.turns))

        total_decisions = sum(len(t.decisions) for t in trajectory.turns)
        span.set_attribute("trajectory.decision_count", total_decisions)

        if trajectory.final_reward is not None:
            span.set_attribute("trajectory.final_reward", trajectory.final_reward)

    except ImportError:
        logger.debug("OpenTelemetry not installed, skipping span link")


def export_to_otel(trajectory: OTSTrajectory) -> str:
    """
    Export a trajectory as OTel spans.

    Convenience function for quick export.

    Args:
        trajectory: OTS trajectory to export

    Returns:
        OTel trace ID
    """
    exporter = OTelTrajectoryExporter()
    return exporter.export_trajectory(trajectory)
