"""
Context Learning Retrieval for agents.

Retrieves relevant past decisions to provide as context for new agent actions.
This enables agents to learn from their past experiences without retraining.

Key concepts:
- Decision retrieval: Find similar past decisions at inference time
- Context formatting: Convert decisions to agent-consumable format
- Flexible filtering: Filter by domain, entity_type, entity_id, action_type, tags
- Outcome-aware: Prefer high-scoring, successful decisions; optionally include failures as anti-patterns

IMPORTANT: This module provides a generic `search()` method with flexible filters.
Domain-specific convenience methods (get_similar_world_decisions, etc.) are deprecated
and will be removed in a future version. Use `search()` with appropriate filters instead.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from letta.log import get_logger
from letta.schemas.trajectory import TrajectorySearchRequest
from letta.schemas.user import User as PydanticUser
from letta.services.trajectory_manager import TrajectoryManager
from letta.trajectories.ots.adapter import OTSAdapter
from letta.trajectories.ots.decision_embeddings import DecisionEmbedder, DecisionSearcher
from letta.trajectories.ots.dsf_entity_extractor import DSFEntityExtractor
from letta.trajectories.ots.models import OTSDecision, OTSTrajectory

logger = get_logger(__name__)


@dataclass
class RetrievedExample:
    """A retrieved decision example for context learning."""

    decision: OTSDecision
    trajectory_id: str
    similarity: float
    outcome_score: Optional[float]
    context_info: Dict[str, Any] = field(default_factory=dict)  # Generic context (domain, entities, etc.)
    formatted_text: str = ""  # Pre-formatted for agent consumption

    # Deprecated: Use context_info instead
    @property
    def world_context(self) -> Optional[str]:
        """Deprecated: Use context_info['world'] instead."""
        return self.context_info.get("world")

    @property
    def story_context(self) -> Optional[str]:
        """Deprecated: Use context_info['story'] instead."""
        return self.context_info.get("story")


@dataclass
class ContextLearningResult:
    """Result of context learning retrieval."""

    query: str
    examples: List[RetrievedExample]
    anti_patterns: List[RetrievedExample] = field(default_factory=list)  # Failure examples
    total_candidates: int = 0
    filter_summary: Dict[str, Any] = field(default_factory=dict)
    formatted_context: str = ""  # Ready-to-use context for agent


class ContextLearning:
    """
    Generic context learning retrieval for agents.

    Retrieves and formats relevant past decisions to include in agent context.
    This is the "retrieval" part of in-context learning - providing relevant
    examples from past successful trajectories.

    The primary method is `search()` which accepts flexible filters:
    - domain: Filter by domain (e.g., "dsf")
    - entity_type: Filter by entity type (e.g., "world", "story")
    - entity_id: Filter by specific entity ID
    - action_type: Filter by tool/action name
    - tags: Filter by tags

    Usage:
        learner = ContextLearning()
        result = await learner.search(
            query="creating world rules",
            actor=user,
            filters={"domain": "dsf", "entity_type": "world"},
            include_failures=True,
        )
        # Include result.formatted_context in agent system prompt
    """

    def __init__(
        self,
        trajectory_manager: Optional[TrajectoryManager] = None,
        adapter: Optional[OTSAdapter] = None,
        embedder: Optional[DecisionEmbedder] = None,
    ):
        """
        Initialize context learning.

        Args:
            trajectory_manager: Trajectory manager for retrieval
            adapter: OTS adapter for conversion
            embedder: Decision embedder for similarity
        """
        self.trajectory_manager = trajectory_manager or TrajectoryManager()
        self.adapter = adapter or OTSAdapter()
        self.embedder = embedder or DecisionEmbedder()
        self.searcher = DecisionSearcher(self.embedder)
        self.entity_extractor = DSFEntityExtractor()

    async def search(
        self,
        query: str,
        actor: PydanticUser,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.7,
        include_failures: bool = False,
        limit: int = 5,
    ) -> ContextLearningResult:
        """
        Generic trajectory search with flexible filters.

        This is the primary search method. Use this for domain-agnostic queries
        with arbitrary filter combinations.

        Args:
            query: Natural language description of what you're looking for
            actor: User for trajectory access
            filters: Optional filter dictionary. Supported keys:
                - domain (str): Filter by domain (e.g., "dsf")
                - entity_type (str): Filter by entity type (e.g., "world", "story")
                - entity_id (str): Filter by specific entity ID
                - action_type (str): Filter by tool/action name
                - tags (List[str]): Filter by tags (any match)
            min_score: Minimum outcome score (0-1) for positive examples
            include_failures: Whether to also retrieve failures as anti-patterns
            limit: Maximum examples per category (successes and failures)

        Returns:
            ContextLearningResult with examples, anti_patterns, and formatted_context

        Example:
            # Find experiences about creating worlds
            result = await learner.search(
                query="creating world rules",
                actor=user,
                filters={"domain": "dsf", "entity_type": "world"},
                include_failures=True,
            )
        """
        filters = filters or {}

        # Build search query with filter context
        query_parts = [query]
        if filters.get("domain"):
            query_parts.append(f"domain: {filters['domain']}")
        if filters.get("entity_type"):
            query_parts.append(f"entity: {filters['entity_type']}")
        if filters.get("action_type"):
            query_parts.append(f"action: {filters['action_type']}")

        search_query = " | ".join(query_parts)

        # Search for successful trajectories
        success_request = TrajectorySearchRequest(
            query=search_query,
            min_score=min_score,
            domain_type=filters.get("domain"),
            limit=limit * 3,
        )

        success_results = await self.trajectory_manager.search_trajectories_async(
            success_request, actor
        )

        success_examples = self._extract_examples(
            results=success_results,
            filters=filters,
            include_failures=False,
            min_score=min_score,
            limit=limit,
        )

        # Search for failures if requested
        anti_patterns: List[RetrievedExample] = []
        failure_count = 0
        if include_failures:
            failure_request = TrajectorySearchRequest(
                query=search_query,
                max_score=0.5,  # Low scores for failures
                domain_type=filters.get("domain"),
                limit=limit * 3,
            )
            failure_results = await self.trajectory_manager.search_trajectories_async(
                failure_request, actor
            )
            failure_count = len(failure_results)
            anti_patterns = self._extract_examples(
                results=failure_results,
                filters=filters,
                include_failures=True,
                min_score=0.0,
                limit=limit,
            )

        # Format combined context
        formatted_context = self._format_combined_context(success_examples, anti_patterns)

        return ContextLearningResult(
            query=search_query,
            examples=success_examples,
            anti_patterns=anti_patterns,
            total_candidates=len(success_results) + failure_count,
            filter_summary=filters,
            formatted_context=formatted_context,
        )

    def _extract_examples(
        self,
        results: List[Any],
        filters: Dict[str, Any],
        include_failures: bool,
        min_score: float,
        limit: int,
    ) -> List[RetrievedExample]:
        """Extract and filter decision examples from search results."""
        examples: List[RetrievedExample] = []

        action_type = filters.get("action_type")
        entity_type = filters.get("entity_type")
        entity_id = filters.get("entity_id")

        for result in results:
            try:
                ots_traj = OTSAdapter.from_letta_run(
                    result.trajectory.data,
                    agent_id=result.trajectory.agent_id,
                )
                outcome_score = result.trajectory.outcome_score

                # Filter by outcome score
                if include_failures:
                    if outcome_score is not None and outcome_score >= 0.5:
                        continue
                else:
                    if outcome_score is not None and outcome_score < min_score:
                        continue

                # Extract entity context
                context_info = self._extract_context_info(ots_traj)

                # Extract decisions from trajectory
                for turn in ots_traj.turns:
                    for decision in turn.decisions:
                        # Filter by action type
                        if action_type and decision.choice.action != action_type:
                            continue

                        # Filter by entity type/id
                        if entity_type or entity_id:
                            if not self._matches_entity_filter(decision, entity_type, entity_id):
                                continue

                        # Skip failed decisions for success examples
                        if not include_failures and not decision.consequence.success:
                            continue

                        # For failure examples, prefer actual failures
                        if include_failures and decision.consequence.success:
                            continue

                        formatted = self._format_decision_for_agent(
                            decision, context_info, include_failures
                        )

                        examples.append(RetrievedExample(
                            decision=decision,
                            trajectory_id=result.trajectory.id,
                            similarity=result.similarity,
                            outcome_score=outcome_score,
                            context_info=context_info,
                            formatted_text=formatted,
                        ))

            except Exception as e:
                logger.warning(f"Failed to process trajectory {result.trajectory.id}: {e}")
                continue

        # Sort by similarity and limit
        examples.sort(key=lambda x: x.similarity, reverse=True)
        return examples[:limit]

    def _extract_context_info(self, ots_traj: OTSTrajectory) -> Dict[str, Any]:
        """Extract context info from trajectory."""
        context_info: Dict[str, Any] = {}

        try:
            entities = self.entity_extractor.extract_all(ots_traj)
            for entity in entities:
                if entity.type == DSFEntityExtractor.WORLD:
                    context_info["world"] = entity.name
                elif entity.type == DSFEntityExtractor.STORY:
                    context_info["story"] = entity.name
                elif entity.type == DSFEntityExtractor.RULE:
                    context_info.setdefault("rules", []).append(entity.name)
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")

        return context_info

    def _matches_entity_filter(
        self,
        decision: OTSDecision,
        entity_type: Optional[str],
        entity_id: Optional[str],
    ) -> bool:
        """Check if decision matches entity filters."""
        if decision.choice.arguments:
            args = decision.choice.arguments
            if entity_type:
                if args.get("type") == entity_type:
                    return True
                if args.get("entity_type") == entity_type:
                    return True
            if entity_id:
                if args.get("id") == entity_id:
                    return True
                if args.get("entity_id") == entity_id:
                    return True
                if entity_type and args.get(f"{entity_type}_id") == entity_id:
                    return True

        return not (entity_type or entity_id)

    async def get_context_for_action(
        self,
        current_situation: str,
        actor: PydanticUser,
        action_type: Optional[str] = None,
        world_checkpoint: Optional[str] = None,
        story_id: Optional[str] = None,
        min_score: float = 0.7,
        max_examples: int = 3,
        include_failures: bool = False,
    ) -> ContextLearningResult:
        """
        Get context examples for an upcoming agent action.

        Convenience method that wraps search() with DSF-specific parameters.
        For more flexible filtering, use search() directly with a filters dict.

        Args:
            current_situation: Description of the current situation
            actor: User for trajectory access
            action_type: Filter by action type (world_manager, story_manager)
            world_checkpoint: Filter by world (DSF-specific)
            story_id: Filter by story (DSF-specific)
            min_score: Minimum outcome score (0-1)
            max_examples: Maximum examples to return
            include_failures: Whether to include failed decisions (for anti-patterns)

        Returns:
            ContextLearningResult with formatted examples
        """
        # Build query with DSF context
        query_parts = [current_situation]
        if world_checkpoint:
            query_parts.append(f"world: {world_checkpoint}")
        if story_id:
            query_parts.append(f"story: {story_id}")
        query = " | ".join(query_parts)

        # Build filters
        filters: Dict[str, Any] = {"domain": "dsf"}
        if action_type:
            filters["action_type"] = action_type

        return await self.search(
            query=query,
            actor=actor,
            filters=filters,
            min_score=min_score,
            include_failures=include_failures,
            limit=max_examples,
        )

    async def get_similar_world_decisions(
        self,
        world_checkpoint: str,
        operation: str,
        actor: PydanticUser,
        max_examples: int = 3,
    ) -> List[RetrievedExample]:
        """
        DEPRECATED: Use search() with filters instead.

        Get similar world management decisions.

        Args:
            world_checkpoint: World to find similar operations for
            operation: world_manager operation (save, load, diff, update)
            actor: User for trajectory access
            max_examples: Maximum examples

        Returns:
            List of relevant world management examples
        """
        warnings.warn(
            "get_similar_world_decisions is deprecated. Use search() with "
            "filters={'entity_type': 'world', 'entity_id': world_checkpoint} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = await self.search(
            query=f"Managing world {world_checkpoint} with {operation} operation",
            actor=actor,
            filters={
                "domain": "dsf",
                "entity_type": "world",
                "action_type": DSFEntityExtractor.WORLD_MANAGER,
            },
            limit=max_examples,
        )
        return result.examples

    async def get_similar_story_decisions(
        self,
        world_checkpoint: str,
        story_id: Optional[str],
        operation: str,
        actor: PydanticUser,
        max_examples: int = 3,
    ) -> List[RetrievedExample]:
        """
        DEPRECATED: Use search() with filters instead.

        Get similar story management decisions.

        Args:
            world_checkpoint: World the story is in
            story_id: Story being worked on (if any)
            operation: story_manager operation
            actor: User for trajectory access
            max_examples: Maximum examples

        Returns:
            List of relevant story management examples
        """
        warnings.warn(
            "get_similar_story_decisions is deprecated. Use search() with "
            "filters={'entity_type': 'story', 'entity_id': story_id} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        situation = f"Managing story in world {world_checkpoint} with {operation} operation"
        if story_id:
            situation += f" (story: {story_id})"

        result = await self.search(
            query=situation,
            actor=actor,
            filters={
                "domain": "dsf",
                "entity_type": "story",
                "entity_id": story_id,
                "action_type": DSFEntityExtractor.STORY_MANAGER,
            },
            limit=max_examples,
        )
        return result.examples

    async def get_rule_application_examples(
        self,
        rule_id: str,
        actor: PydanticUser,
        max_examples: int = 3,
    ) -> List[RetrievedExample]:
        """
        DEPRECATED: Use search() with filters instead.

        Get examples of a specific rule being applied.

        Args:
            rule_id: Rule ID to find applications of
            actor: User for trajectory access
            max_examples: Maximum examples

        Returns:
            List of examples where this rule was used
        """
        warnings.warn(
            "get_rule_application_examples is deprecated. Use search() with "
            "filters={'entity_type': 'rule', 'entity_id': rule_id} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = await self.search(
            query=f"Applying rule {rule_id} in story",
            actor=actor,
            filters={"domain": "dsf", "entity_type": "rule", "entity_id": rule_id},
            limit=max_examples * 2,
        )

        # Filter for examples that applied this rule
        filtered = []
        for example in result.examples:
            args = example.decision.choice.arguments or {}
            segment = args.get("segment", {})
            if isinstance(segment, dict):
                evolution = segment.get("world_evolution", {})
                if rule_id in evolution.get("rules_applied", []):
                    filtered.append(example)

        return filtered[:max_examples]

    def _format_decision_for_agent(
        self,
        decision: OTSDecision,
        context_info: Dict[str, Any],
        is_failure: bool = False,
    ) -> str:
        """Format a decision for inclusion in agent context."""
        lines = []

        # Context from context_info
        world_context = context_info.get("world")
        story_context = context_info.get("story")
        if world_context or story_context:
            ctx_parts = []
            if world_context:
                ctx_parts.append(f"World: {world_context}")
            if story_context:
                ctx_parts.append(f"Story: {story_context}")
            lines.append(f"**Context**: {', '.join(ctx_parts)}")

        # State/Situation
        if decision.state and decision.state.context_summary:
            lines.append(f"**Situation**: {decision.state.context_summary}")

        # Choice/Action
        lines.append(f"**Action**: {decision.choice.action}")
        if decision.choice.rationale:
            lines.append(f"**Reasoning**: {decision.choice.rationale}")

        # Outcome
        if is_failure:
            lines.append(f"**Result**: {decision.consequence.result_summary[:200] if decision.consequence.result_summary else 'Failed'}")
            if decision.consequence.error_type:
                lines.append(f"**Error**: {decision.consequence.error_type}")
            if decision.evaluation and decision.evaluation.feedback:
                lines.append(f"**Lesson**: {decision.evaluation.feedback[:200]}")
        else:
            lines.append(f"**Result**: {decision.consequence.result_summary[:200] if decision.consequence.result_summary else 'Success'}")

        return "\n".join(lines)

    def _format_context(
        self,
        examples: List[RetrievedExample],
        include_failures: bool,
    ) -> str:
        """Format examples of a single type into agent-ready context."""
        if not examples:
            return ""

        lines = []

        if include_failures:
            lines.append("## Anti-Patterns to Avoid")
            lines.append("")
            lines.append("The following decisions led to poor outcomes:")
        else:
            lines.append("## Relevant Past Decisions")
            lines.append("")
            lines.append("Consider these successful past decisions as guidance:")

        lines.append("")

        for i, example in enumerate(examples, 1):
            outcome = "✗ Failed" if include_failures else "✓ Success"
            lines.append(f"### Example {i} (similarity: {example.similarity:.2f}, outcome: {outcome})")
            lines.append(example.formatted_text)
            lines.append("")

        return "\n".join(lines)

    def _format_combined_context(
        self,
        success_examples: List[RetrievedExample],
        anti_patterns: List[RetrievedExample],
    ) -> str:
        """Format both success examples and anti-patterns into agent-ready context."""
        sections = []

        # Success examples section
        if success_examples:
            lines = [
                "## Relevant Past Decisions",
                "",
                "Consider these successful past approaches:",
                "",
            ]
            for i, example in enumerate(success_examples, 1):
                lines.append(f"### Example {i} (similarity: {example.similarity:.2f}, outcome: ✓ Success)")
                lines.append(example.formatted_text)
                lines.append("")
            sections.append("\n".join(lines))

        # Anti-patterns section
        if anti_patterns:
            lines = [
                "## Anti-Patterns to Avoid",
                "",
                "The following approaches led to poor outcomes:",
                "",
            ]
            for i, example in enumerate(anti_patterns, 1):
                lines.append(f"### Example {i} (similarity: {example.similarity:.2f}, outcome: ✗ Failed)")
                lines.append(example.formatted_text)
                lines.append("")
            sections.append("\n".join(lines))

        return "\n".join(sections)


# Backwards compatibility alias
DSFContextLearning = ContextLearning


# Convenience functions

async def search_trajectories(
    query: str,
    actor: PydanticUser,
    filters: Optional[Dict[str, Any]] = None,
    include_failures: bool = False,
    limit: int = 5,
) -> ContextLearningResult:
    """
    Search trajectories for relevant past decisions.

    This is the primary convenience function for context learning.
    Returns both successful examples and optional anti-patterns.

    Args:
        query: Natural language description of what you're looking for
        actor: User for trajectory access
        filters: Optional filter dictionary (domain, entity_type, entity_id, action_type, tags)
        include_failures: Whether to also retrieve failures as anti-patterns
        limit: Maximum examples per category

    Returns:
        ContextLearningResult with examples, anti_patterns, and formatted_context

    Example:
        result = await search_trajectories(
            query="creating world rules",
            actor=user,
            filters={"domain": "dsf", "entity_type": "world"},
            include_failures=True,
        )
        # Include result.formatted_context in agent system prompt
    """
    learner = ContextLearning()
    return await learner.search(
        query=query,
        actor=actor,
        filters=filters,
        include_failures=include_failures,
        limit=limit,
    )


async def get_dsf_context(
    situation: str,
    actor: PydanticUser,
    world_checkpoint: Optional[str] = None,
    max_examples: int = 3,
) -> str:
    """
    DEPRECATED: Use search_trajectories() instead.

    Get formatted context for a DSF agent action.

    Args:
        situation: Description of what the agent is about to do
        actor: User for trajectory access
        world_checkpoint: Filter by world if relevant
        max_examples: Maximum examples to include

    Returns:
        Formatted context string for agent
    """
    warnings.warn(
        "get_dsf_context is deprecated. Use search_trajectories() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    learner = ContextLearning()
    result = await learner.get_context_for_action(
        current_situation=situation,
        actor=actor,
        world_checkpoint=world_checkpoint,
        max_examples=max_examples,
    )
    return result.formatted_context


async def get_anti_patterns(
    situation: str,
    actor: PydanticUser,
    max_examples: int = 2,
) -> str:
    """
    DEPRECATED: Use search_trajectories(include_failures=True) instead.

    Get formatted anti-patterns to avoid.

    Args:
        situation: Description of what the agent is about to do
        actor: User for trajectory access
        max_examples: Maximum anti-patterns to include

    Returns:
        Formatted anti-patterns string for agent
    """
    warnings.warn(
        "get_anti_patterns is deprecated. Use search_trajectories(include_failures=True) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    learner = ContextLearning()
    result = await learner.search(
        query=situation,
        actor=actor,
        include_failures=True,
        limit=max_examples,
    )
    return result.formatted_context
