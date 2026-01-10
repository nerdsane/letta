"""
Context Learning Retrieval for DSF agents.

Retrieves relevant past decisions to provide as context for new agent actions.
This enables agents to learn from their past experiences without retraining.

Key concepts:
- Decision retrieval: Find similar past decisions at inference time
- Context formatting: Convert decisions to agent-consumable format
- Domain filtering: Filter by world, story, rule relevance
- Outcome-aware: Prefer high-scoring, successful decisions
"""

from dataclasses import dataclass
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
    world_context: Optional[str]  # Which world this came from
    story_context: Optional[str]  # Which story this came from
    formatted_text: str  # Pre-formatted for agent consumption


@dataclass
class ContextLearningResult:
    """Result of context learning retrieval."""

    query: str
    examples: List[RetrievedExample]
    total_candidates: int
    filter_summary: Dict[str, Any]
    formatted_context: str  # Ready-to-use context for agent


class DSFContextLearning:
    """
    Context learning retrieval for DSF agents.

    Retrieves and formats relevant past decisions to include in agent context.
    This is the "retrieval" part of in-context learning - providing relevant
    examples from past successful trajectories.

    Usage:
        learner = DSFContextLearning()
        context = await learner.get_context_for_action(
            current_situation="Agent is creating a story in the Nexus world",
            action_type="story_manager",
            actor=user,
            world_checkpoint="nexus-v3",
        )
        # Include context.formatted_context in agent system prompt
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

        This is the main entry point for context learning. Call this before
        an agent takes an action to provide relevant past examples.

        Args:
            current_situation: Description of the current situation
            actor: User for trajectory access
            action_type: Filter by action type (world_manager, story_manager)
            world_checkpoint: Filter by world
            story_id: Filter by story
            min_score: Minimum outcome score (0-1)
            max_examples: Maximum examples to return
            include_failures: Whether to include failed decisions (for anti-patterns)

        Returns:
            ContextLearningResult with formatted examples
        """
        # Build search query
        query_parts = [current_situation]
        if action_type:
            query_parts.append(f"action: {action_type}")
        if world_checkpoint:
            query_parts.append(f"world: {world_checkpoint}")
        if story_id:
            query_parts.append(f"story: {story_id}")

        query = " | ".join(query_parts)

        # Search trajectories
        search_request = TrajectorySearchRequest(
            query=query,
            min_score=min_score if not include_failures else None,
            max_score=0.3 if include_failures else None,  # Low scores for failures
            domain_type="dsf",
            limit=max_examples * 3,  # Retrieve more, then filter
        )

        results = await self.trajectory_manager.search_trajectories_async(
            search_request,
            actor,
        )

        # Convert to OTS and extract decisions
        examples: List[RetrievedExample] = []

        for result in results:
            try:
                ots_traj = self.adapter.from_letta_trajectory(result.trajectory)

                # Extract DSF entities for context
                entities = self.entity_extractor.extract_all(ots_traj)
                world_ctx = None
                story_ctx = None
                for entity in entities:
                    if entity.type == DSFEntityExtractor.WORLD:
                        world_ctx = entity.name
                    if entity.type == DSFEntityExtractor.STORY:
                        story_ctx = entity.name

                # Filter decisions by action type if specified
                for turn in ots_traj.turns:
                    for decision in turn.decisions:
                        if action_type and decision.choice.action != action_type:
                            continue

                        # Skip failed decisions unless explicitly requested
                        if not include_failures and not decision.consequence.success:
                            continue

                        # Format for agent
                        formatted = self._format_decision_for_agent(
                            decision,
                            world_ctx,
                            story_ctx,
                        )

                        examples.append(RetrievedExample(
                            decision=decision,
                            trajectory_id=result.trajectory.id,
                            similarity=result.similarity,
                            outcome_score=result.trajectory.outcome_score,
                            world_context=world_ctx,
                            story_context=story_ctx,
                            formatted_text=formatted,
                        ))

            except Exception as e:
                logger.warning(f"Failed to process trajectory {result.trajectory.id}: {e}")
                continue

        # Sort by similarity and limit
        examples.sort(key=lambda x: x.similarity, reverse=True)
        examples = examples[:max_examples]

        # Build formatted context
        formatted_context = self._format_context(examples, include_failures)

        return ContextLearningResult(
            query=query,
            examples=examples,
            total_candidates=len(results),
            filter_summary={
                "action_type": action_type,
                "world_checkpoint": world_checkpoint,
                "story_id": story_id,
                "min_score": min_score,
                "include_failures": include_failures,
            },
            formatted_context=formatted_context,
        )

    async def get_similar_world_decisions(
        self,
        world_checkpoint: str,
        operation: str,
        actor: PydanticUser,
        max_examples: int = 3,
    ) -> List[RetrievedExample]:
        """
        Get similar world management decisions.

        Args:
            world_checkpoint: World to find similar operations for
            operation: world_manager operation (save, load, diff, update)
            actor: User for trajectory access
            max_examples: Maximum examples

        Returns:
            List of relevant world management examples
        """
        situation = f"Managing world {world_checkpoint} with {operation} operation"

        result = await self.get_context_for_action(
            current_situation=situation,
            actor=actor,
            action_type=DSFEntityExtractor.WORLD_MANAGER,
            world_checkpoint=world_checkpoint,
            max_examples=max_examples,
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
        situation = f"Managing story in world {world_checkpoint} with {operation} operation"
        if story_id:
            situation += f" (story: {story_id})"

        result = await self.get_context_for_action(
            current_situation=situation,
            actor=actor,
            action_type=DSFEntityExtractor.STORY_MANAGER,
            world_checkpoint=world_checkpoint,
            story_id=story_id,
            max_examples=max_examples,
        )

        return result.examples

    async def get_rule_application_examples(
        self,
        rule_id: str,
        actor: PydanticUser,
        max_examples: int = 3,
    ) -> List[RetrievedExample]:
        """
        Get examples of a specific rule being applied.

        Args:
            rule_id: Rule ID to find applications of
            actor: User for trajectory access
            max_examples: Maximum examples

        Returns:
            List of examples where this rule was used
        """
        result = await self.get_context_for_action(
            current_situation=f"Applying rule {rule_id} in story",
            actor=actor,
            max_examples=max_examples * 2,  # Retrieve more, filter by rule
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
        world_context: Optional[str],
        story_context: Optional[str],
    ) -> str:
        """Format a decision for inclusion in agent context."""
        lines = []

        # Header
        lines.append(f"**Past Decision ({decision.decision_type.value})**")

        # Context
        if world_context or story_context:
            ctx_parts = []
            if world_context:
                ctx_parts.append(f"World: {world_context}")
            if story_context:
                ctx_parts.append(f"Story: {story_context}")
            lines.append(f"Context: {', '.join(ctx_parts)}")

        # State
        if decision.state and decision.state.context_summary:
            lines.append(f"Situation: {decision.state.context_summary}")

        # Choice
        lines.append(f"Action: {decision.choice.action}")
        if decision.choice.rationale:
            lines.append(f"Reasoning: {decision.choice.rationale}")

        # Outcome
        outcome = "✓ Success" if decision.consequence.success else "✗ Failed"
        lines.append(f"Outcome: {outcome}")
        if decision.consequence.result_summary:
            result = decision.consequence.result_summary[:200]
            lines.append(f"Result: {result}")

        return "\n".join(lines)

    def _format_context(
        self,
        examples: List[RetrievedExample],
        include_failures: bool,
    ) -> str:
        """Format all examples into agent-ready context."""
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
            lines.append(f"### Example {i} (similarity: {example.similarity:.2f})")
            lines.append(example.formatted_text)
            lines.append("")

        return "\n".join(lines)


# Convenience functions

async def get_dsf_context(
    situation: str,
    actor: PydanticUser,
    world_checkpoint: Optional[str] = None,
    max_examples: int = 3,
) -> str:
    """
    Get formatted context for a DSF agent action.

    This is the simplest way to add context learning to an agent.
    Include the returned string in the agent's system prompt.

    Args:
        situation: Description of what the agent is about to do
        actor: User for trajectory access
        world_checkpoint: Filter by world if relevant
        max_examples: Maximum examples to include

    Returns:
        Formatted context string for agent
    """
    learner = DSFContextLearning()
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
    Get formatted anti-patterns to avoid.

    Returns examples of past failures to warn the agent.

    Args:
        situation: Description of what the agent is about to do
        actor: User for trajectory access
        max_examples: Maximum anti-patterns to include

    Returns:
        Formatted anti-patterns string for agent
    """
    learner = DSFContextLearning()
    result = await learner.get_context_for_action(
        current_situation=situation,
        actor=actor,
        min_score=0.0,  # We want failures
        max_examples=max_examples,
        include_failures=True,
    )
    return result.formatted_context
