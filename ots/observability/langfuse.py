"""
Langfuse exporter for OTS trajectories (EXPORT-ONLY).

Exports OTS trajectories to Langfuse for visualization and analysis.

This is a one-way export to visualize OTS trajectories in Langfuse.
It does NOT import Langfuse traces into OTS, because Langfuse traces
lack the reasoning context (rationale, alternatives) that makes OTS
valuable for context learning.

Use this when:
- You want to visualize OTS trajectories in Langfuse's UI
- You want to use Langfuse's cost tracking and scoring features
- Your team already uses Langfuse and wants trajectory visibility there

Do NOT use this to:
- Import existing Langfuse traces into OTS (loses reasoning context)
- Replace OTS storage (use TrajectoryStore for context learning)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ots.models import (
    OTSDecision,
    OTSTrajectory,
    OTSTurn,
)

logger = logging.getLogger(__name__)


class LangfuseExporter:
    """
    Exports OTS trajectories to Langfuse format.

    Langfuse is an open-source LLM observability platform that provides:
    - Trace visualization
    - Cost tracking
    - Quality scoring
    - Prompt management

    OTS trajectories map to Langfuse concepts:
    - OTSTrajectory -> Langfuse Trace
    - OTSTurn -> Langfuse Generation (for LLM calls)
    - OTSDecision -> Langfuse Span (for tool calls)
    - Annotations -> Langfuse Scores

    Requires: pip install ots[langfuse]

    Example:
        exporter = LangfuseExporter(public_key="...", secret_key="...")
        trace_id = await exporter.export_trajectory(ots_trajectory)
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
    ) -> None:
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
                    "langfuse package not installed. Install with: pip install ots[langfuse]"
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


async def export_to_langfuse(
    trajectory: OTSTrajectory,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> str:
    """
    Export a trajectory to Langfuse.

    Convenience function for quick export.

    Args:
        trajectory: OTS trajectory to export
        public_key: Langfuse public key (or use env var)
        secret_key: Langfuse secret key (or use env var)

    Returns:
        Langfuse trace ID
    """
    exporter = LangfuseExporter(public_key=public_key, secret_key=secret_key)
    return await exporter.export_trajectory(trajectory)
