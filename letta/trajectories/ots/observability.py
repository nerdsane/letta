"""
Observability exporters for OTS trajectories.

Provides integration with:
- Langfuse: LLM observability platform
- OpenTelemetry: Standard observability framework

These exporters convert OTS trajectories to formats consumable by
observability tools, enabling visualization and analysis of agent
decision traces.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from letta.log import get_logger
from letta.trajectories.ots.models import (
    OTSDecision,
    OTSMessage,
    OTSTrajectory,
    OTSTurn,
)

logger = get_logger(__name__)


class LangfuseExporter:
    """
    Exports OTS trajectories to Langfuse format.

    Langfuse is an open-source LLM observability platform that provides:
    - Trace visualization
    - Cost tracking
    - Quality scoring
    - Prompt management

    OTS trajectories map to Langfuse concepts:
    - OTSTrajectory → Langfuse Trace
    - OTSTurn → Langfuse Generation (for LLM calls)
    - OTSDecision → Langfuse Span (for tool calls)
    - Annotations → Langfuse Scores

    Usage:
        exporter = LangfuseExporter(public_key="...", secret_key="...")
        trace_id = await exporter.export_trajectory(ots_trajectory)
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
    ):
        """
        Initialize Langfuse exporter.

        Args:
            public_key: Langfuse public key (or set LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or set LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL
        """
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self._client = None

    def _get_client(self):
        """Get or create Langfuse client."""
        if self._client is None:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
            except ImportError:
                raise ImportError(
                    "langfuse package not installed. Install with: pip install langfuse"
                )
        return self._client

    async def export_trajectory(
        self,
        trajectory: OTSTrajectory,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Export an OTS trajectory to Langfuse.

        Args:
            trajectory: OTS trajectory to export
            user_id: Optional user ID for filtering
            session_id: Optional session ID for grouping
            tags: Optional tags for filtering

        Returns:
            Langfuse trace ID
        """
        client = self._get_client()

        # Create trace
        trace = client.trace(
            id=trajectory.trajectory_id,
            name=trajectory.metadata.task_description[:100] if trajectory.metadata.task_description else "Agent Trajectory",
            user_id=user_id or trajectory.metadata.agent_id,
            session_id=session_id,
            metadata={
                "domain": trajectory.metadata.domain,
                "framework": trajectory.metadata.framework,
                "outcome": trajectory.metadata.outcome.value if trajectory.metadata.outcome else None,
                "duration_ms": trajectory.metadata.duration_ms,
            },
            tags=tags or trajectory.metadata.tags,
            input=trajectory.metadata.task_description,
            output=self._get_trajectory_output(trajectory),
        )

        # Export turns as generations/spans
        for turn in trajectory.turns:
            self._export_turn(trace, turn)

        # Export annotations as scores
        if trajectory.final_reward is not None:
            trace.score(
                name="final_reward",
                value=trajectory.final_reward,
                comment="Overall trajectory quality score",
            )

        # Flush to ensure data is sent
        client.flush()

        return trajectory.trajectory_id

    def _export_turn(self, trace, turn: OTSTurn) -> None:
        """Export a turn as Langfuse generation and spans."""
        # Create generation for LLM interaction
        generation_id = f"{trace.id}-turn-{turn.turn_id}"

        # Collect messages for this turn
        messages = []
        for msg in turn.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content.text or str(msg.content.data),
            })

        # Find assistant response
        assistant_response = None
        for msg in turn.messages:
            if msg.role.value == "assistant":
                assistant_response = msg.content.text or str(msg.content.data)
                break

        # Create generation
        generation = trace.generation(
            id=generation_id,
            name=f"Turn {turn.turn_id}",
            start_time=turn.timestamp,
            end_time=self._calculate_end_time(turn),
            input=messages,
            output=assistant_response,
            metadata={
                "span_id": turn.span_id,
                "parent_span_id": turn.parent_span_id,
                "error": turn.error,
            },
        )

        # Export decisions as spans
        for decision in turn.decisions:
            self._export_decision(generation, decision)

        # Add turn reward if available
        if turn.turn_reward is not None:
            generation.score(
                name="turn_reward",
                value=turn.turn_reward,
                comment=f"Turn {turn.turn_id} quality score",
            )

    def _export_decision(self, parent, decision: OTSDecision) -> None:
        """Export a decision as a Langfuse span."""
        span = parent.span(
            id=decision.decision_id,
            name=f"Decision: {decision.choice.action}",
            input={
                "state": decision.state.context_summary if decision.state else None,
                "alternatives": decision.alternatives,
            },
            output={
                "action": decision.choice.action,
                "arguments": decision.choice.arguments,
                "success": decision.consequence.success,
                "result": decision.consequence.result_summary,
            },
            metadata={
                "decision_type": decision.decision_type.value,
                "rationale": decision.choice.rationale,
                "confidence": decision.choice.confidence,
                "error_type": decision.consequence.error_type,
            },
        )

        # Add decision evaluation as score
        if decision.evaluation:
            span.score(
                name="decision_quality",
                value=decision.evaluation.score,
                comment=decision.evaluation.feedback,
            )

    def _get_trajectory_output(self, trajectory: OTSTrajectory) -> str:
        """Get output summary from trajectory."""
        if not trajectory.turns:
            return ""

        # Get last assistant message
        for turn in reversed(trajectory.turns):
            for msg in reversed(turn.messages):
                if msg.role.value == "assistant" and msg.content.text:
                    return msg.content.text[:1000]

        return ""

    def _calculate_end_time(self, turn: OTSTurn) -> Optional[datetime]:
        """Calculate end time from duration."""
        if turn.duration_ms and turn.timestamp:
            from datetime import timedelta
            return turn.timestamp + timedelta(milliseconds=turn.duration_ms)
        return None

    def to_langfuse_format(self, trajectory: OTSTrajectory) -> Dict[str, Any]:
        """
        Convert trajectory to Langfuse-compatible dict format.

        Useful for batch export or custom integrations.

        Args:
            trajectory: OTS trajectory to convert

        Returns:
            Dict in Langfuse trace format
        """
        return {
            "id": trajectory.trajectory_id,
            "name": trajectory.metadata.task_description[:100] if trajectory.metadata.task_description else "Agent Trajectory",
            "timestamp": trajectory.metadata.timestamp_start.isoformat() if trajectory.metadata.timestamp_start else None,
            "metadata": {
                "domain": trajectory.metadata.domain,
                "framework": trajectory.metadata.framework,
                "outcome": trajectory.metadata.outcome.value if trajectory.metadata.outcome else None,
                "agent_id": trajectory.metadata.agent_id,
            },
            "tags": trajectory.metadata.tags or [],
            "input": trajectory.metadata.task_description,
            "output": self._get_trajectory_output(trajectory),
            "generations": [
                self._turn_to_generation(turn) for turn in trajectory.turns
            ],
            "scores": [
                {"name": "final_reward", "value": trajectory.final_reward}
            ] if trajectory.final_reward is not None else [],
        }

    def _turn_to_generation(self, turn: OTSTurn) -> Dict[str, Any]:
        """Convert turn to Langfuse generation format."""
        messages = []
        output = None

        for msg in turn.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content.text or str(msg.content.data),
            })
            if msg.role.value == "assistant":
                output = msg.content.text or str(msg.content.data)

        return {
            "id": f"turn-{turn.turn_id}",
            "name": f"Turn {turn.turn_id}",
            "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
            "input": messages,
            "output": output,
            "spans": [
                self._decision_to_span(d) for d in turn.decisions
            ],
        }

    def _decision_to_span(self, decision: OTSDecision) -> Dict[str, Any]:
        """Convert decision to Langfuse span format."""
        return {
            "id": decision.decision_id,
            "name": f"Decision: {decision.choice.action}",
            "input": {
                "state": decision.state.context_summary if decision.state else None,
            },
            "output": {
                "action": decision.choice.action,
                "arguments": decision.choice.arguments,
                "success": decision.consequence.success,
            },
            "metadata": {
                "decision_type": decision.decision_type.value,
            },
        }


class OTelTrajectoryExporter:
    """
    Exports OTS trajectories as OpenTelemetry spans.

    Integrates with Letta's existing OTel infrastructure to provide:
    - Trace continuity across agent operations
    - Decision-level span granularity
    - Standard OTLP export to any backend

    OTS concepts map to OTel:
    - OTSTrajectory → Root span
    - OTSTurn → Child span
    - OTSDecision → Nested span with tool.* attributes
    - Annotations → Span events

    Usage:
        exporter = OTelTrajectoryExporter()
        exporter.export_trajectory(ots_trajectory)
    """

    def __init__(self, tracer_name: str = "letta.trajectories.ots"):
        """
        Initialize OTel exporter.

        Args:
            tracer_name: Name for the tracer
        """
        from opentelemetry import trace
        self.tracer = trace.get_tracer(tracer_name)

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

        # Start root span for trajectory
        with self.tracer.start_as_current_span(
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

        with self.tracer.start_as_current_span(
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

        with self.tracer.start_as_current_span(
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
                        "impact": decision.credit_assignment.impact,
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


# Convenience functions

async def export_to_langfuse(
    trajectory: OTSTrajectory,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> str:
    """
    Export a trajectory to Langfuse.

    Args:
        trajectory: OTS trajectory to export
        public_key: Langfuse public key (or use env var)
        secret_key: Langfuse secret key (or use env var)

    Returns:
        Langfuse trace ID
    """
    exporter = LangfuseExporter(public_key=public_key, secret_key=secret_key)
    return await exporter.export_trajectory(trajectory)


def export_to_otel(trajectory: OTSTrajectory) -> str:
    """
    Export a trajectory as OTel spans.

    Args:
        trajectory: OTS trajectory to export

    Returns:
        OTel trace ID
    """
    exporter = OTelTrajectoryExporter()
    return exporter.export_trajectory(trajectory)
