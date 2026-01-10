"""
Adapter to convert Letta trajectories to Open Trajectory Specification (OTS) format.

This adapter bridges Letta's internal trajectory representation to the
portable OTS format for continual learning.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from letta.log import get_logger
from letta.schemas.trajectory import Trajectory as LettaTrajectory
from letta.trajectories.ots.models import (
    ContentType,
    DecisionType,
    MessageRole,
    OutcomeType,
    OTSChoice,
    OTSConsequence,
    OTSContext,
    OTSContextSnapshot,
    OTSDecision,
    OTSEntity,
    OTSMessage,
    OTSMessageContent,
    OTSMetadata,
    OTSResource,
    OTSSystemMessage,
    OTSTrajectory,
    OTSTurn,
    OTSUser,
)

logger = get_logger(__name__)


class OTSAdapter:
    """
    Converts Letta trajectories to Open Trajectory Specification format.

    Usage:
        adapter = OTSAdapter()
        ots_trajectory = adapter.from_letta_trajectory(letta_trajectory)
    """

    def __init__(self, decision_extractor=None):
        """
        Initialize adapter.

        Args:
            decision_extractor: Optional DecisionExtractor for enriching decisions
        """
        self.decision_extractor = decision_extractor

    def from_letta_trajectory(
        self,
        letta_trajectory: LettaTrajectory,
        extract_decisions: bool = True,
    ) -> OTSTrajectory:
        """
        Convert a Letta trajectory to OTS format.

        Args:
            letta_trajectory: Letta trajectory object
            extract_decisions: Whether to extract decisions (requires LLM for full extraction)

        Returns:
            OTSTrajectory object
        """
        data = letta_trajectory.data

        # Convert metadata
        metadata = self._convert_metadata(letta_trajectory, data)

        # Convert context
        context = self._convert_context(data)

        # Convert system message
        system_message = self._extract_system_message(data)

        # Convert turns
        turns = self._convert_turns(data, extract_decisions)

        # Get final reward from outcome score
        final_reward = letta_trajectory.outcome_score

        return OTSTrajectory(
            trajectory_id=str(letta_trajectory.id),
            version="0.1-draft",
            metadata=metadata,
            context=context,
            system_message=system_message,
            turns=turns,
            final_reward=final_reward,
        )

    def _convert_metadata(
        self,
        letta_trajectory: LettaTrajectory,
        data: Dict[str, Any],
    ) -> OTSMetadata:
        """Convert Letta trajectory to OTS metadata."""
        letta_metadata = data.get("metadata", {})
        outcome_data = data.get("outcome", {})

        # Determine outcome type
        outcome_type = self._map_outcome_type(outcome_data)

        # Get timestamps
        start_time = letta_metadata.get("start_time")
        end_time = letta_metadata.get("end_time")

        timestamp_start = (
            datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            if start_time
            else letta_trajectory.created_at
        )
        timestamp_end = (
            datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            if end_time
            else None
        )

        # Calculate duration
        duration_ms = None
        if letta_metadata.get("duration_ns"):
            duration_ms = letta_metadata["duration_ns"] / 1_000_000

        # Build task description from available sources
        task_description = self._infer_task_description(data, letta_trajectory)

        return OTSMetadata(
            task_description=task_description,
            domain=letta_trajectory.domain_type,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_ms=duration_ms,
            agent_id=str(letta_trajectory.agent_id),
            framework="letta",
            environment=None,  # Could infer from run metadata
            outcome=outcome_type,
            feedback_score=letta_trajectory.outcome_score,
            human_reviewed=False,
            tags=letta_trajectory.tags or [],
            parent_trajectory_id=None,
        )

    def _map_outcome_type(self, outcome_data: Dict[str, Any]) -> OutcomeType:
        """Map Letta outcome to OTS outcome type."""
        # Check new format first
        execution = outcome_data.get("execution", {})
        status = execution.get("status") or outcome_data.get("type")

        if status in ["completed", "success"]:
            return OutcomeType.SUCCESS
        elif status in ["incomplete", "partial_success"]:
            return OutcomeType.PARTIAL_SUCCESS
        else:
            return OutcomeType.FAILURE

    def _infer_task_description(
        self,
        data: Dict[str, Any],
        letta_trajectory: LettaTrajectory,
    ) -> str:
        """Infer task description from available sources."""
        # Use searchable summary if available
        if letta_trajectory.searchable_summary:
            return letta_trajectory.searchable_summary

        # Try to get from first user message
        turns = data.get("turns", [])
        for turn in turns:
            for msg in turn.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if content and isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("text"):
                                return c["text"][:500]  # Truncate
                            elif isinstance(c, str):
                                return c[:500]
                    elif isinstance(content, str):
                        return content[:500]

        return "Unknown task"

    def _convert_context(self, data: Dict[str, Any]) -> OTSContext:
        """Convert Letta data to OTS context."""
        # Extract entities from tool results and metadata
        entities = self._extract_entities(data)
        resources = self._extract_resources(data)

        return OTSContext(
            referrer=None,
            user=None,  # Could extract from run metadata
            entities=entities,
            resources=resources,
            custom_context=None,
        )

    def _extract_entities(self, data: Dict[str, Any]) -> List[OTSEntity]:
        """Extract entities from trajectory data."""
        entities = []
        seen_ids = set()

        # Look for entities in tool results
        for turn in data.get("turns", []):
            for msg in turn.get("messages", []):
                # Check tool calls for entity references
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")

                    # DSF-specific: world_manager, story_manager
                    if name in ["world_manager", "story_manager"]:
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                import json
                                args = json.loads(args)
                            except:
                                args = {}

                        # Extract world_id or story_id
                        for key in ["world_id", "story_id", "id"]:
                            if key in args and args[key] not in seen_ids:
                                entity_type = "world" if "world" in key else "story"
                                entities.append(OTSEntity(
                                    type=entity_type,
                                    id=args[key],
                                    name=args.get("name"),
                                    metadata=None,
                                ))
                                seen_ids.add(args[key])

        return entities

    def _extract_resources(self, data: Dict[str, Any]) -> List[OTSResource]:
        """Extract resources from trajectory data."""
        resources = []
        seen_uris = set()

        # Look for file/URL references in tool calls
        for turn in data.get("turns", []):
            for msg in turn.get("messages", []):
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            import json
                            args = json.loads(args)
                        except:
                            args = {}

                    # Look for file paths or URLs
                    for key in ["file_path", "path", "url", "uri"]:
                        if key in args and args[key] not in seen_uris:
                            resource_type = "url" if key == "url" else "file"
                            resources.append(OTSResource(
                                type=resource_type,
                                uri=args[key],
                                accessed_at=None,
                            ))
                            seen_uris.add(args[key])

        return resources

    def _extract_system_message(self, data: Dict[str, Any]) -> Optional[OTSSystemMessage]:
        """Extract system message from trajectory data."""
        turns = data.get("turns", [])
        if not turns:
            return None

        # Look for system message in first turn
        for msg in turns[0].get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", [])
                if content:
                    text = ""
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("text"):
                                text = c["text"]
                                break
                    elif isinstance(content, str):
                        text = content

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
        data: Dict[str, Any],
        extract_decisions: bool,
    ) -> List[OTSTurn]:
        """Convert Letta turns to OTS turns."""
        ots_turns = []
        letta_turns = data.get("turns", [])

        for i, turn in enumerate(letta_turns):
            # Convert messages
            messages = self._convert_messages(turn.get("messages", []))

            # Extract decisions (programmatic only for now)
            decisions = []
            if extract_decisions:
                decisions = self._extract_tool_decisions(turn, i)

            ots_turn = OTSTurn(
                turn_id=i,
                span_id=turn.get("step_id", str(uuid4())),
                parent_span_id=None,
                timestamp=self._parse_timestamp(turn),
                duration_ms=None,
                error=False,
                turn_reward=None,
                messages=messages,
                decisions=decisions,
            )

            ots_turns.append(ots_turn)

        return ots_turns

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[OTSMessage]:
        """Convert Letta messages to OTS messages."""
        ots_messages = []

        for msg in messages:
            role = self._map_role(msg.get("role", "assistant"))
            content = self._convert_content(msg)
            timestamp = msg.get("timestamp")

            ots_msg = OTSMessage(
                message_id=msg.get("message_id", str(uuid4())),
                role=role,
                timestamp=(
                    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if timestamp
                    else datetime.utcnow()
                ),
                content=content,
                reasoning=None,  # Would need LLM extraction
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
        """Convert Letta message content to OTS format."""
        content = msg.get("content", [])
        tool_calls = msg.get("tool_calls", [])

        # Check for tool calls
        if tool_calls:
            return OTSMessageContent(
                type=ContentType.TOOL_CALL,
                data={"tool_calls": tool_calls},
                text=None,
            )

        # Check for tool response
        if msg.get("tool_call_id"):
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        text = c["text"]
                        break
            elif isinstance(content, str):
                text = content

            return OTSMessageContent(
                type=ContentType.TOOL_RESPONSE,
                data={"tool_call_id": msg["tool_call_id"]},
                text=text,
            )

        # Regular text content
        text = ""
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("text"):
                    text = c["text"]
                    break
        elif isinstance(content, str):
            text = content

        return OTSMessageContent(
            type=ContentType.TEXT,
            data=None,
            text=text,
        )

    def _extract_tool_decisions(
        self,
        turn: Dict[str, Any],
        turn_id: int,
    ) -> List[OTSDecision]:
        """
        Extract decisions from tool calls (programmatic extraction).

        This extracts the structured part of decisions. LLM extraction
        for alternatives/rationale is done separately by DecisionExtractor.
        """
        decisions = []

        for msg in turn.get("messages", []):
            tool_calls = msg.get("tool_calls", [])

            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                arguments = func.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        import json
                        arguments = json.loads(arguments)
                    except:
                        arguments = {"raw": arguments}

                # Find corresponding tool result
                success, result_summary = self._find_tool_result(
                    turn.get("messages", []),
                    tc.get("id"),
                )

                decision = OTSDecision(
                    decision_id=f"t{turn_id}-d{len(decisions)}",
                    decision_type=DecisionType.TOOL_SELECTION,
                    state=None,  # Would need LLM to extract context summary
                    alternatives=None,  # Would need LLM to extract
                    choice=OTSChoice(
                        action=tool_name,
                        arguments=arguments,
                        rationale=None,  # Would need LLM to extract
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
                text = ""
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("text"):
                            text = c["text"]
                            break
                elif isinstance(content, str):
                    text = content

                # Check for error indicators
                is_error = "error" in text.lower() or "exception" in text.lower()

                return not is_error, text[:500] if text else None

        return True, None

    def _parse_timestamp(self, turn: Dict[str, Any]) -> datetime:
        """Parse timestamp from turn or message."""
        # Try turn-level timestamp
        messages = turn.get("messages", [])
        if messages:
            ts = messages[0].get("timestamp")
            if ts:
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except:
                    pass

        return datetime.utcnow()
