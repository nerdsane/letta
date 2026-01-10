"""
DSFEntityExtractor - Extracts Deep Sci-Fi specific entities from OTS trajectories.

Implements the ots.EntityExtractor protocol for domain-specific entity extraction.

DSF entities include:
- Worlds (with development state, version)
- Stories (with segments, world contributions)
- Rules (scope, certainty, tested status)
- Elements (characters, locations, technology)
- Constraints (physical, social, logical, narrative)
"""

import json
import re
from typing import Any, Dict, List, Set

from ots import ContentType, OTSEntity, OTSTrajectory, OTSTurn

from letta.log import get_logger

logger = get_logger(__name__)


class DSFEntityExtractor:
    """
    Extracts DSF-specific entities from OTS trajectories.

    Implements the ots.EntityExtractor protocol.

    DSF entities have domain-specific semantics:
    - Worlds are versioned, have development state
    - Stories track world contributions
    - Rules have scope, certainty, and are testable
    - Elements have detail levels and relationships

    Example:
        from ots import TrajectoryStore
        from letta.ots import DSFEntityExtractor

        store = TrajectoryStore()
        store.register_extractor(DSFEntityExtractor())
    """

    # DSF entity types
    WORLD = "world"
    STORY = "story"
    RULE = "rule"
    ELEMENT = "element"
    CONSTRAINT = "constraint"
    SEGMENT = "segment"

    # DSF tool names
    WORLD_MANAGER = "world_manager"
    STORY_MANAGER = "story_manager"
    ASSET_MANAGER = "asset_manager"

    def extract(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """
        Extract all DSF entities from a trajectory.

        Args:
            trajectory: OTS trajectory to extract entities from

        Returns:
            List of extracted OTS entities
        """
        entities = []
        seen_ids: Set[str] = set()

        for turn in trajectory.turns:
            turn_entities = self._extract_from_turn(turn)
            for entity in turn_entities:
                if entity.id not in seen_ids:
                    entities.append(entity)
                    seen_ids.add(entity.id)

        return entities

    def extract_worlds(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """Extract only world entities."""
        return [e for e in self.extract(trajectory) if e.type == self.WORLD]

    def extract_stories(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """Extract only story entities."""
        return [e for e in self.extract(trajectory) if e.type == self.STORY]

    def extract_rules(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """Extract only rule entities."""
        return [e for e in self.extract(trajectory) if e.type == self.RULE]

    def extract_elements(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """Extract only element entities (characters, locations, tech)."""
        return [e for e in self.extract(trajectory) if e.type == self.ELEMENT]

    def _extract_from_turn(self, turn: OTSTurn) -> List[OTSEntity]:
        """Extract entities from a single turn."""
        entities = []

        for message in turn.messages:
            # Check for tool calls
            if message.content.type == ContentType.TOOL_CALL and message.content.data:
                tool_calls = message.content.data.get("tool_calls", [])
                for tc in tool_calls:
                    entities.extend(self._extract_from_tool_call(tc))

            # Check for tool responses
            if message.content.type == ContentType.TOOL_RESPONSE and message.content.text:
                entities.extend(self._extract_from_tool_response(message.content.text))

        return entities

    def _extract_from_tool_call(self, tool_call: Dict[str, Any]) -> List[OTSEntity]:
        """Extract entities from a tool call."""
        entities = []
        func = tool_call.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return entities

        if name == self.WORLD_MANAGER:
            entities.extend(self._extract_from_world_manager(args))
        elif name == self.STORY_MANAGER:
            entities.extend(self._extract_from_story_manager(args))

        return entities

    def _extract_from_world_manager(self, args: Dict[str, Any]) -> List[OTSEntity]:
        """Extract entities from world_manager call."""
        entities = []
        operation = args.get("operation", "")

        # Extract checkpoint name as world ID
        checkpoint = args.get("checkpoint_name") or args.get("current_checkpoint")
        if checkpoint:
            entities.append(OTSEntity(
                type=self.WORLD,
                id=f"world:{checkpoint}",
                name=checkpoint,
                metadata={
                    "operation": operation,
                }
            ))

        # For save/update operations, extract world data
        if operation in ["save", "update"] and args.get("world"):
            world_data = args["world"]
            if isinstance(world_data, str):
                try:
                    world_data = json.loads(world_data)
                except json.JSONDecodeError:
                    world_data = {}

            # Extract world metadata
            if checkpoint and isinstance(world_data, dict):
                dev = world_data.get("development", {})
                entities[-1].metadata.update({
                    "development_state": dev.get("state"),
                    "version": dev.get("version"),
                })

            # Extract rules
            foundation = world_data.get("foundation", {})
            for rule in foundation.get("rules", []):
                if isinstance(rule, dict) and rule.get("id"):
                    entities.append(OTSEntity(
                        type=self.RULE,
                        id=f"rule:{rule['id']}",
                        name=rule.get("statement", "")[:100],
                        metadata={
                            "scope": rule.get("scope"),
                            "certainty": rule.get("certainty"),
                            "world": checkpoint,
                        }
                    ))

            # Extract elements
            surface = world_data.get("surface", {})
            for element in surface.get("visible_elements", []):
                if isinstance(element, dict) and element.get("id"):
                    entities.append(OTSEntity(
                        type=self.ELEMENT,
                        id=f"element:{element['id']}",
                        name=element.get("name") or element.get("type"),
                        metadata={
                            "element_type": element.get("type"),
                            "detail_level": element.get("detail_level"),
                            "world": checkpoint,
                        }
                    ))

            # Extract constraints
            for constraint in world_data.get("constraints", []):
                if isinstance(constraint, dict) and constraint.get("id"):
                    entities.append(OTSEntity(
                        type=self.CONSTRAINT,
                        id=f"constraint:{constraint['id']}",
                        name=constraint.get("description", "")[:100],
                        metadata={
                            "constraint_type": constraint.get("type"),
                            "strictness": constraint.get("strictness"),
                            "world": checkpoint,
                        }
                    ))

        return entities

    def _extract_from_story_manager(self, args: Dict[str, Any]) -> List[OTSEntity]:
        """Extract entities from story_manager call."""
        entities = []
        operation = args.get("operation", "")

        # Extract story ID
        story_id = args.get("story_id")
        if story_id:
            entities.append(OTSEntity(
                type=self.STORY,
                id=f"story:{story_id}",
                name=args.get("title") or story_id,
                metadata={
                    "operation": operation,
                    "world_checkpoint": args.get("world_checkpoint"),
                }
            ))

        # For create operation
        if operation == "create":
            if args.get("title"):
                entities.append(OTSEntity(
                    type=self.STORY,
                    id=f"story:{self._generate_id(args['title'])}",
                    name=args["title"],
                    metadata={
                        "operation": "create",
                        "world_checkpoint": args.get("world_checkpoint"),
                    }
                ))

        # For save_segment operation
        if operation == "save_segment" and args.get("segment"):
            segment = args["segment"]
            if isinstance(segment, dict):
                evolution = segment.get("world_evolution", {})
                if evolution.get("rules_applied"):
                    for rule_id in evolution["rules_applied"]:
                        entities.append(OTSEntity(
                            type=self.RULE,
                            id=f"rule:{rule_id}",
                            name=rule_id,
                            metadata={
                                "tested_in_story": story_id,
                                "action": "applied",
                            }
                        ))

                if evolution.get("elements_introduced"):
                    for element_id in evolution["elements_introduced"]:
                        entities.append(OTSEntity(
                            type=self.ELEMENT,
                            id=f"element:{element_id}",
                            name=element_id,
                            metadata={
                                "introduced_in_story": story_id,
                            }
                        ))

        return entities

    def _extract_from_tool_response(self, text: str) -> List[OTSEntity]:
        """Extract entities from tool response text."""
        entities = []

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return entities

            # Check for world data
            if "development" in data and "surface" in data:
                dev = data.get("development", {})
                checkpoint = dev.get("checkpoint_name") or "unknown"

                entities.append(OTSEntity(
                    type=self.WORLD,
                    id=f"world:{checkpoint}",
                    name=checkpoint,
                    metadata={
                        "development_state": dev.get("state"),
                        "version": dev.get("version"),
                        "source": "tool_response",
                    }
                ))

            # Check for story data
            if "segments" in data and "world_checkpoint" in data:
                story_id = data.get("id", "unknown")
                entities.append(OTSEntity(
                    type=self.STORY,
                    id=f"story:{story_id}",
                    name=data.get("metadata", {}).get("title", story_id),
                    metadata={
                        "world_checkpoint": data.get("world_checkpoint"),
                        "world_version": data.get("world_version"),
                        "status": data.get("metadata", {}).get("status"),
                        "segment_count": len(data.get("segments", [])),
                        "source": "tool_response",
                    }
                ))

                # Extract world contributions
                contributions = data.get("world_contributions", {})
                for rule_id in contributions.get("rules_tested", []):
                    entities.append(OTSEntity(
                        type=self.RULE,
                        id=f"rule:{rule_id}",
                        name=rule_id,
                        metadata={
                            "tested_in_story": story_id,
                        }
                    ))

        except (json.JSONDecodeError, TypeError):
            pass

        return entities

    def _generate_id(self, title: str) -> str:
        """Generate ID from title (matching story_manager logic)."""
        slug = re.sub(r'[^a-z0-9\s-]', '', title.lower())
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug.strip('-')[:50]


def extract_dsf_entities(trajectory: OTSTrajectory) -> List[OTSEntity]:
    """
    Convenience function to extract DSF entities from a trajectory.

    Args:
        trajectory: OTS trajectory to extract from

    Returns:
        List of extracted DSF entities
    """
    extractor = DSFEntityExtractor()
    return extractor.extract(trajectory)


def enrich_trajectory_context(trajectory: OTSTrajectory) -> OTSTrajectory:
    """
    Enrich trajectory context with extracted DSF entities.

    Args:
        trajectory: OTS trajectory to enrich

    Returns:
        Trajectory with updated context.entities
    """
    extractor = DSFEntityExtractor()
    entities = extractor.extract(trajectory)

    # Merge with existing entities
    existing_ids = {e.id for e in trajectory.context.entities}
    new_entities = [e for e in entities if e.id not in existing_ids]

    trajectory.context.entities.extend(new_entities)

    return trajectory
