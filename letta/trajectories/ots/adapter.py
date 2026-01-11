"""
Adapter to convert Letta runs to Open Trajectory Specification (OTS) format.

Letta runs are the native execution traces. This adapter converts them directly
to OTS format for continual learning - no intermediate "trajectory" needed.

Run format (native Letta):
{
  "run_id": "run-...",
  "metadata": { start_time, end_time, duration_ns, status, tokens, models },
  "turns": [{ step_id, model, messages }],
  "outcome": { execution, type, confidence }
}

OTS format (output):
{
  "trajectory_id": "...",
  "metadata": { task_description, agent_id, outcome, ... },
  "turns": [{ turn_id, messages, decisions, ... }],
  "final_reward": 0.0-1.0
}
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from letta.log import get_logger
from letta.trajectories.ots.models import (
    ContentType,
    DecisionType,
    MessageRole,
    OutcomeType,
    OTSChoice,
    OTSConsequence,
    OTSContext,
    OTSDecision,
    OTSEntity,
    OTSMessage,
    OTSMessageContent,
    OTSMetadata,
    OTSResource,
    OTSSystemMessage,
    OTSTrajectory,
    OTSTurn,
)

logger = get_logger(__name__)


class OTSAdapter:
    """
    Converts Letta runs to Open Trajectory Specification format.

    Letta runs are the native format. OTS is the standardized output for
    continual learning, display, and analysis.

    Usage:
        # From run dict (native Letta format)
        ots_trajectory = OTSAdapter.from_letta_run(run_data)

        # From Run ORM/schema object
        ots_trajectory = OTSAdapter.from_run_object(run, steps, messages)
    """

    @classmethod
    def from_letta_run(
        cls,
        run_data: Dict[str, Any],
        agent_id: Optional[str] = None,
        extract_decisions: bool = True,
    ) -> OTSTrajectory:
        """
        Convert a Letta run (native format) to OTS.

        Args:
            run_data: Run data dict with run_id, metadata, turns, outcome
            agent_id: Optional agent ID (if not in run_data)
            extract_decisions: Whether to extract decisions from tool calls

        Returns:
            OTSTrajectory object
        """
        adapter = cls()

        # Extract run info
        run_id = run_data.get("run_id", str(uuid4()))
        run_metadata = run_data.get("metadata", {})
        turns_data = run_data.get("turns", [])
        outcome_data = run_data.get("outcome", {})

        # Infer agent_id from various sources
        if not agent_id:
            agent_id = run_metadata.get("agent_id") or "unknown"

        # Convert metadata
        metadata = adapter._convert_run_metadata(
            run_id=run_id,
            run_metadata=run_metadata,
            outcome_data=outcome_data,
            turns_data=turns_data,
            agent_id=agent_id,
        )

        # Extract context (entities, resources from tool calls)
        context = adapter._extract_context(turns_data)

        # Extract system message if present
        system_message = adapter._extract_system_message(turns_data)

        # Convert turns
        turns = adapter._convert_turns(turns_data, extract_decisions)

        # Calculate final reward from outcome confidence
        final_reward = outcome_data.get("confidence")

        return OTSTrajectory(
            trajectory_id=run_id,
            version="0.1-draft",
            metadata=metadata,
            context=context,
            system_message=system_message,
            turns=turns,
            final_reward=final_reward,
        )

    def _convert_run_metadata(
        self,
        run_id: str,
        run_metadata: Dict[str, Any],
        outcome_data: Dict[str, Any],
        turns_data: List[Dict[str, Any]],
        agent_id: str,
    ) -> OTSMetadata:
        """Convert run metadata to OTS metadata."""
        # Parse timestamps
        start_time = run_metadata.get("start_time")
        end_time = run_metadata.get("end_time")

        timestamp_start = (
            datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            if start_time
            else datetime.utcnow()
        )
        timestamp_end = (
            datetime.fromisoformat(str(end_time).replace("Z", "+00:00"))
            if end_time
            else None
        )

        # Calculate duration
        duration_ms = None
        if run_metadata.get("duration_ns"):
            duration_ms = run_metadata["duration_ns"] / 1_000_000

        # Map outcome
        outcome_type = self._map_outcome_type(outcome_data)

        # Infer task description from first user message
        task_description = self._infer_task_description(turns_data)

        # Get model info
        models = run_metadata.get("models", [])
        framework = "letta"

        return OTSMetadata(
            task_description=task_description,
            domain=None,  # Could be set by caller
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_ms=duration_ms,
            agent_id=agent_id,
            framework=framework,
            environment=models[0] if models else None,
            outcome=outcome_type,
            feedback_score=outcome_data.get("confidence"),
            human_reviewed=False,
            tags=[],
            parent_trajectory_id=None,
        )

    def _map_outcome_type(self, outcome_data: Dict[str, Any]) -> OutcomeType:
        """Map run outcome to OTS outcome type."""
        execution = outcome_data.get("execution", {})
        status = execution.get("status") or outcome_data.get("type", "")

        if status in ["completed", "success"]:
            return OutcomeType.SUCCESS
        elif status in ["partial_success", "incomplete"]:
            return OutcomeType.PARTIAL_SUCCESS
        else:
            return OutcomeType.FAILURE

    def _infer_task_description(self, turns_data: List[Dict[str, Any]]) -> str:
        """Infer task description from first user message."""
        for turn in turns_data:
            for msg in turn.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    text = self._extract_text_from_content(content)
                    if text:
                        return text[:500]  # Truncate

        return "Unknown task"

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        return item["text"]
                    if item.get("text"):
                        return item["text"]
                elif isinstance(item, str):
                    return item
        return ""

    def _extract_context(self, turns_data: List[Dict[str, Any]]) -> OTSContext:
        """Extract context (entities, resources) from turns."""
        entities = []
        resources = []
        seen_entity_ids = set()
        seen_uris = set()

        for turn in turns_data:
            for msg in turn.get("messages", []):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_call":
                            # Extract from tool calls
                            tool_calls = item.get("tool_calls", [])
                            for tc in tool_calls:
                                self._extract_entities_from_tool_call(
                                    tc, entities, seen_entity_ids
                                )
                                self._extract_resources_from_tool_call(
                                    tc, resources, seen_uris
                                )

                # Also check top-level tool_calls
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    self._extract_entities_from_tool_call(tc, entities, seen_entity_ids)
                    self._extract_resources_from_tool_call(tc, resources, seen_uris)

        return OTSContext(
            referrer=None,
            user=None,
            entities=entities,
            resources=resources,
            custom_context=None,
        )

    def _extract_entities_from_tool_call(
        self,
        tc: Dict[str, Any],
        entities: List[OTSEntity],
        seen_ids: set,
    ):
        """Extract entities from a tool call."""
        func = tc.get("function", {})
        name = func.get("name", "")
        args = self._parse_arguments(func.get("arguments", {}))

        # DSF-specific entity extraction
        if name in ["world_manager", "story_manager", "asset_manager"]:
            for key in ["world_id", "story_id", "asset_id", "id"]:
                if key in args and args[key] not in seen_ids:
                    entity_type = key.replace("_id", "")
                    entities.append(OTSEntity(
                        type=entity_type,
                        id=args[key],
                        name=args.get("name"),
                        metadata=None,
                    ))
                    seen_ids.add(args[key])

    def _extract_resources_from_tool_call(
        self,
        tc: Dict[str, Any],
        resources: List[OTSResource],
        seen_uris: set,
    ):
        """Extract resources from a tool call."""
        func = tc.get("function", {})
        args = self._parse_arguments(func.get("arguments", {}))

        for key in ["file_path", "path", "url", "uri"]:
            if key in args and args[key] not in seen_uris:
                resource_type = "url" if key == "url" else "file"
                resources.append(OTSResource(
                    type=resource_type,
                    uri=args[key],
                    accessed_at=None,
                ))
                seen_uris.add(args[key])

    def _parse_arguments(self, args: Any) -> Dict[str, Any]:
        """Parse tool call arguments."""
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                import json
                return json.loads(args)
            except:
                return {"raw": args}
        return {}

    def _extract_system_message(
        self,
        turns_data: List[Dict[str, Any]],
    ) -> Optional[OTSSystemMessage]:
        """Extract system message from turns."""
        if not turns_data:
            return None

        for msg in turns_data[0].get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", [])
                text = self._extract_text_from_content(content)
                if text:
                    timestamp = msg.get("timestamp")
                    return OTSSystemMessage(
                        content=text,
                        timestamp=(
                            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            if timestamp
                            else datetime.utcnow()
                        ),
                    )

        return None

    def _convert_turns(
        self,
        turns_data: List[Dict[str, Any]],
        extract_decisions: bool,
    ) -> List[OTSTurn]:
        """Convert run turns to OTS turns."""
        ots_turns = []

        for i, turn in enumerate(turns_data):
            # Convert messages
            messages = self._convert_messages(turn.get("messages", []))

            # Extract decisions from tool calls
            decisions = []
            if extract_decisions:
                decisions = self._extract_decisions(turn, i)

            # Get timing info
            step_id = turn.get("step_id", str(uuid4()))
            timestamp = self._parse_turn_timestamp(turn)

            ots_turn = OTSTurn(
                turn_id=i,
                span_id=step_id,
                parent_span_id=None,
                timestamp=timestamp,
                duration_ms=None,
                error=turn.get("error", False),
                turn_reward=None,
                messages=messages,
                decisions=decisions,
            )

            ots_turns.append(ots_turn)

        return ots_turns

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[OTSMessage]:
        """Convert run messages to OTS messages."""
        ots_messages = []

        for msg in messages:
            role = self._map_role(msg.get("role", "assistant"))
            content = self._convert_content(msg)
            timestamp = msg.get("timestamp")

            # Extract reasoning if present
            reasoning = self._extract_reasoning(msg)

            ots_msg = OTSMessage(
                message_id=msg.get("message_id", str(uuid4())),
                role=role,
                timestamp=(
                    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if timestamp
                    else datetime.utcnow()
                ),
                content=content,
                reasoning=reasoning,
                visibility=None,
                context_snapshot=None,
            )

            ots_messages.append(ots_msg)

        return ots_messages

    def _map_role(self, role: str) -> MessageRole:
        """Map Letta role to OTS role."""
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "tool": MessageRole.TOOL,
        }
        return role_map.get(role, MessageRole.ASSISTANT)

    def _convert_content(self, msg: Dict[str, Any]) -> OTSMessageContent:
        """Convert message content to OTS format."""
        content = msg.get("content", [])
        tool_calls = msg.get("tool_calls", [])

        # Handle tool calls
        if tool_calls:
            return OTSMessageContent(
                type=ContentType.TOOL_CALL,
                data={"tool_calls": tool_calls},
                text=None,
            )

        # Handle tool response
        if msg.get("tool_call_id"):
            text = self._extract_text_from_content(content)
            return OTSMessageContent(
                type=ContentType.TOOL_RESPONSE,
                data={"tool_call_id": msg["tool_call_id"]},
                text=text,
            )

        # Handle regular content (may have reasoning + text)
        text_parts = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    # Skip reasoning items - handled separately
                elif isinstance(item, str):
                    text_parts.append(item)
        elif isinstance(content, str):
            text_parts.append(content)

        return OTSMessageContent(
            type=ContentType.TEXT,
            data=None,
            text="\n".join(text_parts) if text_parts else None,
        )

    def _extract_reasoning(self, msg: Dict[str, Any]) -> Optional[str]:
        """Extract reasoning/thinking from message content."""
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    return item.get("text")
        return None

    def _extract_decisions(
        self,
        turn: Dict[str, Any],
        turn_id: int,
    ) -> List[OTSDecision]:
        """Extract decisions from tool calls in a turn."""
        decisions = []
        messages = turn.get("messages", [])

        for msg in messages:
            tool_calls = msg.get("tool_calls", [])

            # Also check content for tool_call type
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_call":
                        tool_calls.extend(item.get("tool_calls", []))

            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                arguments = self._parse_arguments(func.get("arguments", {}))

                # Find tool result
                success, result_summary = self._find_tool_result(
                    messages,
                    tc.get("id"),
                )

                decision = OTSDecision(
                    decision_id=f"t{turn_id}-d{len(decisions)}",
                    decision_type=DecisionType.TOOL_SELECTION,
                    state=None,  # Requires LLM extraction
                    alternatives=None,  # Requires LLM extraction
                    choice=OTSChoice(
                        action=tool_name,
                        arguments=arguments,
                        rationale=None,  # Requires LLM extraction
                        confidence=None,
                    ),
                    consequence=OTSConsequence(
                        success=success,
                        result_summary=result_summary,
                        error_type=None if success else "tool_error",
                    ),
                    evaluation=None,
                    credit_assignment=None,
                )

                decisions.append(decision)

        return decisions

    def _find_tool_result(
        self,
        messages: List[Dict[str, Any]],
        tool_call_id: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        """Find the result of a tool call."""
        if not tool_call_id:
            return True, None

        for msg in messages:
            if msg.get("tool_call_id") == tool_call_id:
                content = msg.get("content", [])
                text = self._extract_text_from_content(content)

                # Check for error indicators
                is_error = any(
                    indicator in text.lower()
                    for indicator in ["error", "exception", "failed", "failure"]
                )

                return not is_error, text[:500] if text else None

            # Also check role=tool messages
            if msg.get("role") == "tool":
                # Tool messages may have tool_call_id differently
                pass

        return True, None

    def _parse_turn_timestamp(self, turn: Dict[str, Any]) -> datetime:
        """Parse timestamp from turn."""
        messages = turn.get("messages", [])
        if messages:
            ts = messages[0].get("timestamp")
            if ts:
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except:
                    pass

        return datetime.utcnow()
