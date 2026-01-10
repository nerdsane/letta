"""
DSF Entity Extractor for OTS trajectories.

Extracts Deep Sci-Fi specific entities from trajectory data:
- Worlds (with development state, version)
- Stories (with segments, contributions)
- Rules (scope, certainty, tested status)
- Elements (characters, locations, technology)
- Constraints (physical, social, logical, narrative)

These entities are used to:
1. Populate trajectory context for retrieval
2. Enable domain-specific filtering
3. Track world/story contributions from decisions
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from letta.log import get_logger
from letta.trajectories.ots.models import OTSEntity, OTSTrajectory, OTSTurn

logger = get_logger(__name__)


class DSFEntityExtractor:
    """
    Extracts DSF-specific entities from OTS trajectories.

    DSF entities have domain-specific semantics:
    - Worlds are versioned, have development state
    - Stories track world contributions
    - Rules have scope, certainty, and are testable
    - Elements have detail levels and relationships

    Usage:
        extractor = DSFEntityExtractor()
        entities = extractor.extract_all(trajectory)
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

    def extract_all(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
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
        return [e for e in self.extract_all(trajectory) if e.type == self.WORLD]

    def extract_stories(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """Extract only story entities."""
        return [e for e in self.extract_all(trajectory) if e.type == self.STORY]

    def extract_rules(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """Extract only rule entities."""
        return [e for e in self.extract_all(trajectory) if e.type == self.RULE]

    def extract_elements(self, trajectory: OTSTrajectory) -> List[OTSEntity]:
        """Extract only element entities (characters, locations, tech)."""
        return [e for e in self.extract_all(trajectory) if e.type == self.ELEMENT]

    def _extract_from_turn(self, turn: OTSTurn) -> List[OTSEntity]:
        """Extract entities from a single turn."""
        entities = []

        for message in turn.messages:
            # Check for tool calls
            if message.content.type.value == "tool_call" and message.content.data:
                tool_calls = message.content.data.get("tool_calls", [])
                for tc in tool_calls:
                    entities.extend(self._extract_from_tool_call(tc))

            # Check for tool responses
            if message.content.type.value == "tool_response" and message.content.text:
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
                # Story created from title
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
                # Track world evolution from segment
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

        # Try to parse as JSON
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return entities

            # Check for world data
            if "development" in data and "surface" in data:
                # This is a world response
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
                # This is a story response
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
        import re
        # Convert to lowercase, replace spaces with hyphens, remove special chars
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
    return extractor.extract_all(trajectory)


def enrich_trajectory_context(trajectory: OTSTrajectory) -> OTSTrajectory:
    """
    Enrich trajectory context with extracted DSF entities.

    Args:
        trajectory: OTS trajectory to enrich

    Returns:
        Trajectory with updated context.entities
    """
    extractor = DSFEntityExtractor()
    entities = extractor.extract_all(trajectory)

    # Merge with existing entities
    existing_ids = {e.id for e in trajectory.context.entities}
    new_entities = [e for e in entities if e.id not in existing_ids]

    trajectory.context.entities.extend(new_entities)

    return trajectory


class DSFEvaluationIntegrator:
    """
    Integrates OTS trajectories with DSF evaluation tools.

    DSF has several evaluation tools:
    - check_logical_consistency: Verify story follows world rules
    - assess_output_quality: Score overall output quality
    - compare_versions: Analyze differences between versions
    - analyze_information_gain: Measure what was learned

    This integrator runs DSF evaluations and stores results as annotations.
    """

    # Evaluation tool names
    CONSISTENCY_CHECKER = "dsf:check_logical_consistency"
    QUALITY_ASSESSOR = "dsf:assess_output_quality"
    VERSION_COMPARATOR = "dsf:compare_versions"
    INFO_GAIN_ANALYZER = "dsf:analyze_information_gain"

    async def evaluate_trajectory(
        self,
        trajectory: OTSTrajectory,
        world_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a trajectory using DSF evaluation tools.

        Args:
            trajectory: OTS trajectory to evaluate
            world_checkpoint: World to check consistency against

        Returns:
            Dict of evaluation results keyed by evaluator ID
        """
        results = {}

        # Extract entities to find world/story context
        extractor = DSFEntityExtractor()
        entities = extractor.extract_all(trajectory)

        worlds = [e for e in entities if e.type == DSFEntityExtractor.WORLD]
        stories = [e for e in entities if e.type == DSFEntityExtractor.STORY]

        # Use provided world or first extracted world
        world_id = world_checkpoint
        if not world_id and worlds:
            world_id = worlds[0].id.replace("world:", "")

        # Evaluate consistency if we have world context
        if world_id:
            results[self.CONSISTENCY_CHECKER] = await self._check_consistency(
                trajectory, world_id
            )

        # Evaluate quality
        results[self.QUALITY_ASSESSOR] = await self._assess_quality(trajectory)

        return results

    async def evaluate_decision(
        self,
        trajectory: OTSTrajectory,
        turn_id: int,
        decision_id: str,
        world_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a specific decision using DSF evaluation tools.

        Args:
            trajectory: OTS trajectory containing the decision
            turn_id: Turn index
            decision_id: Decision ID
            world_checkpoint: World to check consistency against

        Returns:
            Dict of evaluation results
        """
        # Find the decision
        decision = None
        for turn in trajectory.turns:
            if turn.turn_id == turn_id:
                for d in turn.decisions:
                    if d.decision_id == decision_id:
                        decision = d
                        break

        if not decision:
            return {"error": f"Decision {decision_id} not found"}

        results = {}

        # Check if this decision involves world/story operations
        action = decision.choice.action
        if action in [DSFEntityExtractor.WORLD_MANAGER, DSFEntityExtractor.STORY_MANAGER]:
            # Evaluate the decision's impact
            results["decision_impact"] = {
                "action": action,
                "arguments": decision.choice.arguments,
                "success": decision.consequence.success,
                "result": decision.consequence.result_summary,
            }

            # If story operation, check rule adherence
            if action == DSFEntityExtractor.STORY_MANAGER and world_checkpoint:
                results[self.CONSISTENCY_CHECKER] = {
                    "evaluator_id": self.CONSISTENCY_CHECKER,
                    "decision_id": decision_id,
                    # Would call actual evaluation here
                    "pending": True,
                }

        return results

    async def _check_consistency(
        self,
        trajectory: OTSTrajectory,
        world_id: str,
    ) -> Dict[str, Any]:
        """
        Check trajectory consistency with world rules.

        In a full implementation, this would call the actual
        check_logical_consistency tool from Letta.
        """
        # Placeholder - would integrate with actual Letta evaluation tools
        return {
            "evaluator_id": self.CONSISTENCY_CHECKER,
            "world_id": world_id,
            "status": "pending",
            "message": "Integration with Letta evaluation tools pending",
        }

    async def _assess_quality(
        self,
        trajectory: OTSTrajectory,
    ) -> Dict[str, Any]:
        """
        Assess overall trajectory quality.

        In a full implementation, this would call the actual
        assess_output_quality tool from Letta.
        """
        # Placeholder - would integrate with actual Letta evaluation tools
        return {
            "evaluator_id": self.QUALITY_ASSESSOR,
            "status": "pending",
            "message": "Integration with Letta evaluation tools pending",
        }
