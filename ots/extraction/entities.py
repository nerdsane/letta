"""
Generic entity extraction from OTS trajectories.

Provides ToolEntityExtractor which automatically extracts entities
from tool calls without any configuration.
"""

import json
from typing import Any, Dict, List, Set

from ots.models import (
    ContentType,
    OTSEntity,
    OTSTrajectory,
)


class ToolEntityExtractor:
    """
    Generic entity extractor that extracts entities from tool calls.

    Automatically extracts:
    - Tool names as "tool" type entities
    - Resources from tool arguments (files, URLs, IDs)

    This provides zero-config entity tracking. For domain-specific
    entity extraction, implement your own EntityExtractor.

    Example:
        extractor = ToolEntityExtractor()
        entities = extractor.extract(trajectory)
    """

    # Common argument patterns that indicate entities
    ENTITY_ARG_PATTERNS = [
        "id", "name", "file", "path", "url", "uri",
        "resource", "entity", "key", "ref", "target",
    ]

    def extract(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """
        Extract entities from all tool calls in trajectory.

        Args:
            trajectory: OTS trajectory to extract from

        Returns:
            List of extracted entities
        """
        entities: List[OTSEntity] = []
        seen_ids: Set[str] = set()

        for turn in trajectory.turns:
            for message in turn.messages:
                if message.content.type == ContentType.TOOL_CALL:
                    turn_entities = self._extract_from_tool_call(message.content.data)
                    for entity in turn_entities:
                        if entity.id not in seen_ids:
                            entities.append(entity)
                            seen_ids.add(entity.id)

        return entities

    def _extract_from_tool_call(self, data: Dict[str, Any] | None) -> List[OTSEntity]:
        """Extract entities from a tool call data structure."""
        if not data:
            return []

        entities: List[OTSEntity] = []

        # Handle different tool call formats
        tool_calls = data.get("tool_calls", [])
        if not tool_calls and "function" in data:
            # Single tool call format
            tool_calls = [data]

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name", "")
            arguments = func.get("arguments", {})

            if not tool_name:
                continue

            # Extract tool as entity
            entities.append(OTSEntity(
                type="tool",
                id=f"tool:{tool_name}",
                name=tool_name,
                metadata={"call_id": tool_call.get("id")},
            ))

            # Parse arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            # Extract entities from arguments
            arg_entities = self._extract_from_arguments(arguments, tool_name)
            entities.extend(arg_entities)

        return entities

    def _extract_from_arguments(
        self,
        arguments: Dict[str, Any],
        tool_name: str,
    ) -> List[OTSEntity]:
        """Extract entities from tool arguments."""
        entities: List[OTSEntity] = []

        for key, value in arguments.items():
            # Check if this looks like an entity reference
            key_lower = key.lower()
            is_entity_key = any(
                pattern in key_lower
                for pattern in self.ENTITY_ARG_PATTERNS
            )

            if is_entity_key and isinstance(value, str) and value:
                # Determine entity type from key
                entity_type = self._infer_entity_type(key)

                entities.append(OTSEntity(
                    type=entity_type,
                    id=f"{entity_type}:{value}",
                    name=value,
                    metadata={
                        "source_tool": tool_name,
                        "source_arg": key,
                    },
                ))

            # Recursively check nested dicts
            elif isinstance(value, dict):
                nested = self._extract_from_arguments(value, tool_name)
                entities.extend(nested)

            # Check lists for entity-like strings
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested = self._extract_from_arguments(item, tool_name)
                        entities.extend(nested)

        return entities

    def _infer_entity_type(self, key: str) -> str:
        """Infer entity type from argument key."""
        key_lower = key.lower()

        if "file" in key_lower or "path" in key_lower:
            return "file"
        elif "url" in key_lower or "uri" in key_lower:
            return "url"
        elif "id" in key_lower:
            return "resource"
        else:
            return "reference"
