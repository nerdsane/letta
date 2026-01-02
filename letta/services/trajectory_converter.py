"""
Trajectory converter service for creating trajectories from agent runs.

Converts completed agent runs (with messages, steps, and metadata) into
trajectory format for continual learning.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from letta.log import get_logger
from letta.orm.run import Run
from letta.orm.step import Step
from letta.orm.message import Message
from letta.schemas.enums import MessageRole
from letta.schemas.trajectory import TrajectoryCreate

logger = get_logger(__name__)


class TrajectoryConverter:
    """
    Converts agent execution data into trajectory format.

    Trajectories capture what agents DID (decisions, reasoning, outcomes)
    to enable continual learning through retrieval and analysis.
    """

    def __init__(self):
        pass

    async def from_run(
        self,
        run: Run,
        steps: List[Step],
        messages: List[Message],
    ) -> TrajectoryCreate:
        """
        Convert a completed Run (with steps and messages) into a trajectory.

        Args:
            run: Completed Run ORM object
            steps: All Steps from this run (in order)
            messages: All Messages from this run (in order)

        Returns:
            TrajectoryCreate object ready for trajectory service
        """
        # Extract metadata
        metadata = self._extract_metadata(run, steps, messages)

        # Structure turns (group messages by step)
        turns = self._structure_turns(steps, messages)

        # Determine outcome
        outcome = self._determine_outcome(run, messages)

        # Build trajectory data
        trajectory_data = {
            "run_id": run.id,
            "metadata": metadata,
            "turns": turns,
            "outcome": outcome,
        }

        # Create TrajectoryCreate schema
        return TrajectoryCreate(
            agent_id=run.agent_id,
            data=trajectory_data,
        )

    def _extract_metadata(
        self,
        run: Run,
        steps: List[Step],
        messages: List[Message],
    ) -> Dict[str, Any]:
        """Extract metadata about the run."""
        from datetime import timezone

        # Calculate duration
        duration_ns = None
        if run.completed_at and run.created_at:
            # Ensure both timestamps are timezone-aware for subtraction
            completed = run.completed_at if run.completed_at.tzinfo else run.completed_at.replace(tzinfo=timezone.utc)
            created = run.created_at if run.created_at.tzinfo else run.created_at.replace(tzinfo=timezone.utc)
            duration_ns = int((completed - created).total_seconds() * 1e9)

        # Collect tools used
        tools_used = set()
        for message in messages:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                        tools_used.add(tool_call.function.name)

        # Count tokens
        total_input_tokens = sum(step.prompt_tokens or 0 for step in steps)
        total_output_tokens = sum(step.completion_tokens or 0 for step in steps)

        return {
            "start_time": run.created_at.isoformat() if run.created_at else None,
            "end_time": run.completed_at.isoformat() if run.completed_at else None,
            "duration_ns": duration_ns,
            "status": run.status,
            "stop_reason": run.stop_reason,
            "step_count": len(steps),
            "message_count": len(messages),
            "tools_used": sorted(list(tools_used)),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "models": list(set(step.model for step in steps if step.model)),
            "run_type": run.metadata_.get("run_type") if run.metadata_ else None,
            "error": run.metadata_.get("error") if run.metadata_ else None,
        }

    def _structure_turns(
        self,
        steps: List[Step],
        messages: List[Message],
    ) -> List[Dict[str, Any]]:
        """
        Structure messages into turns (grouped by step).

        Each turn represents one LLM inference cycle.
        """
        turns = []

        # Group messages by step
        messages_by_step = {}
        for message in messages:
            step_id = message.step_id
            if step_id:
                if step_id not in messages_by_step:
                    messages_by_step[step_id] = []
                messages_by_step[step_id].append(message)

        # Create turn for each step
        for step in steps:
            turn_messages = messages_by_step.get(step.id, [])

            turn = {
                "step_id": step.id,
                "model": step.model,
                "input_tokens": step.prompt_tokens,
                "output_tokens": step.completion_tokens,
                "stop_reason": step.stop_reason,
                "messages": [self._message_to_dict(msg) for msg in turn_messages],
            }

            turns.append(turn)

        return turns

    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert Message ORM to dictionary for trajectory."""
        msg_dict = {
            "message_id": message.id,
            "role": message.role,
            "timestamp": message.created_at.isoformat() if message.created_at else None,
        }

        # Handle content (new structured format)
        if message.content:
            msg_dict["content"] = [
                {
                    "type": content.type if hasattr(content, 'type') else "text",
                    "text": content.text if hasattr(content, 'text') else str(content),
                }
                for content in message.content
            ]

        # Fallback to legacy text field
        elif message.text:
            msg_dict["content"] = [{"type": "text", "text": message.text}]

        # Add tool calls if present
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    } if hasattr(tc, 'function') else None,
                }
                for tc in message.tool_calls
            ]

        # Add tool call ID if this is a tool response
        if message.tool_call_id:
            msg_dict["tool_call_id"] = message.tool_call_id

        return msg_dict

    def _determine_outcome(
        self,
        run: Run,
        messages: List[Message],
    ) -> Dict[str, Any]:
        """
        Determine the execution status of the run (completed/incomplete/failed/error).

        This measures whether the run completed without errors, NOT the quality
        or learning value of the interaction (that's measured by outcome_score).

        Uses heuristics based on run status, stop reason, and messages.
        """
        # Default execution status
        execution_status = "unknown"
        confidence = 0.5
        reasoning = []

        # Check run status
        if run.status == "completed":
            execution_status = "completed"  # Renamed from "success"
            confidence = 0.7
            reasoning.append("Run completed without errors")

            # Check stop reason for additional signals
            if run.stop_reason == "end_turn":
                confidence = 0.8
                reasoning.append("Agent naturally ended turn")
            elif run.stop_reason == "max_tokens":
                execution_status = "incomplete"  # Renamed from "partial_success"
                confidence = 0.5
                reasoning.append("Hit token limit (may be incomplete)")

        elif run.status == "failed":
            execution_status = "failed"  # Same name
            confidence = 0.9
            reasoning.append("Run failed with error")

            # Include error details
            if run.metadata_ and run.metadata_.get("error"):
                reasoning.append(f"Error: {run.metadata_['error']}")

        elif run.status == "cancelled":
            execution_status = "failed"  # Same name
            confidence = 0.8
            reasoning.append("Run was cancelled")

        # Check for user engagement signals
        user_message_count = sum(1 for msg in messages if msg.role == MessageRole.user)
        if user_message_count > 3:
            # Multiple user messages suggest continued engagement
            if execution_status == "completed":
                confidence = min(1.0, confidence + 0.1)
                reasoning.append(f"High user engagement ({user_message_count} user messages)")

        # Check for assistant tool usage (indicates active problem-solving)
        tool_calls = sum(1 for msg in messages if msg.tool_calls)
        if tool_calls > 0:
            reasoning.append(f"Used {tool_calls} tool calls")

        # Map execution status to old outcome type for backward compatibility
        status_to_outcome_map = {
            "completed": "success",
            "incomplete": "partial_success",
            "failed": "failure",
            "error": "error",
            "unknown": "unknown",
        }

        # Return both new and old format for backward compatibility
        return {
            # New format (clear semantics - preferred)
            "execution": {
                "status": execution_status,  # completed | incomplete | failed | error | unknown
                "confidence": confidence,  # 0.0-1.0
                "reasoning": reasoning,
            },
            # Old format (deprecated, kept for backward compatibility)
            "type": status_to_outcome_map.get(execution_status, "unknown"),  # success | partial_success | failure | unknown
            "confidence": confidence,  # 0.0-1.0
            "reasoning": reasoning,
        }
