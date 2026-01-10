"""
Context Learning for OTS trajectories.

Retrieves relevant past decisions to provide as context for new agent actions.
This enables agents to learn from their past experiences without retraining.

Key concepts:
- Decision retrieval: Find similar past decisions at inference time
- Context formatting: Convert decisions to agent-consumable format
- Outcome-aware: Prefer high-scoring, successful decisions
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ots.models import OTSDecision, OTSTrajectory
from ots.protocols import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievedExample:
    """A retrieved decision example for context learning."""

    decision: OTSDecision
    trajectory_id: str
    similarity: float
    outcome_score: Optional[float]
    context_info: Dict[str, Any] = field(default_factory=dict)
    formatted_text: str = ""


@dataclass
class ContextLearningResult:
    """Result of context learning retrieval."""

    query: str
    examples: List[RetrievedExample]
    anti_patterns: List[RetrievedExample]  # Failure examples (if requested)
    total_candidates: int
    filter_summary: Dict[str, Any]
    formatted_context: str  # Ready-to-use context for agent


class ContextLearning:
    """
    Context learning retrieval for agents.

    Retrieves and formats relevant past decisions to include in agent context.
    This is the "retrieval" part of in-context learning - providing relevant
    examples from past successful trajectories.

    Example:
        from ots import TrajectoryStore
        from ots.learning import ContextLearning

        store = TrajectoryStore()
        learner = ContextLearning(store)

        # Get context for an upcoming action
        result = await learner.get_context(
            situation="Agent is about to make an API call",
            action_type="api_call",
            max_examples=3,
        )

        # Include in agent prompt
        system_prompt = f'''
        {base_prompt}

        ## Relevant Past Experience
        {result.formatted_context}
        '''
    """

    def __init__(self, store: Any) -> None:
        """
        Initialize context learning.

        Args:
            store: TrajectoryStore instance for retrieval
        """
        self.store = store

    async def search(
        self,
        query: str,
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
            filters: Optional filter dictionary. Supported keys:
                - domain (str): Filter by domain (e.g., "dsf", "coding")
                - entity_type (str): Filter by entity type (e.g., "world", "story")
                - entity_id (str): Filter by specific entity ID
                - action_type (str): Filter by tool/action name
                - tags (List[str]): Filter by tags (any match)
            min_score: Minimum outcome score (0-1) for positive examples
            include_failures: Whether to also retrieve failures as anti-patterns
            limit: Maximum examples per category (successes and failures)

        Returns:
            ContextLearningResult with examples and formatted context

        Example:
            # Find experiences about creating worlds
            result = await learner.search(
                query="creating world rules",
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
        success_results = await self.store.search(
            query=search_query,
            limit=limit * 3,  # Retrieve more, then filter
            min_score=min_score,
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
            failure_results = await self.store.search(
                query=search_query,
                limit=limit * 3,
                min_score=0.0,  # Accept any score for failures
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
        formatted_context = self._format_combined_context(
            success_examples, anti_patterns
        )

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
        results: List[SearchResult],
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
            trajectory = result.trajectory
            outcome_score = trajectory.metadata.feedback_score

            # Filter by outcome score
            if include_failures:
                # For failures, we want low scores
                if outcome_score is not None and outcome_score >= 0.5:
                    continue
            else:
                # For successes, we want high scores
                if outcome_score is not None and outcome_score < min_score:
                    continue

            # Extract decisions from trajectory
            for turn in trajectory.turns:
                for decision in turn.decisions:
                    # Filter by action type
                    if action_type and decision.choice.action != action_type:
                        continue

                    # Filter by entity type/id (check in decision state or args)
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
                        decision, trajectory, include_failures
                    )

                    examples.append(RetrievedExample(
                        decision=decision,
                        trajectory_id=trajectory.trajectory_id,
                        similarity=result.similarity,
                        outcome_score=outcome_score,
                        context_info={
                            "domain": trajectory.metadata.domain,
                            "tags": trajectory.metadata.tags,
                        },
                        formatted_text=formatted,
                    ))

        # Sort by similarity and limit
        examples.sort(key=lambda x: x.similarity, reverse=True)
        return examples[:limit]

    def _matches_entity_filter(
        self,
        decision: OTSDecision,
        entity_type: Optional[str],
        entity_id: Optional[str],
    ) -> bool:
        """Check if decision matches entity filters."""
        # Check in choice arguments
        if decision.choice.arguments:
            args = decision.choice.arguments
            if entity_type:
                # Look for entity type in common arg patterns
                if args.get("type") == entity_type:
                    return True
                if args.get("entity_type") == entity_type:
                    return True
            if entity_id:
                if args.get("id") == entity_id:
                    return True
                if args.get("entity_id") == entity_id:
                    return True
                # Check for common id patterns
                if args.get(f"{entity_type}_id") == entity_id:
                    return True

        # Check in state entities
        if decision.state and decision.state.active_entities:
            for entity in decision.state.active_entities:
                if entity_type and entity.entity_type != entity_type:
                    continue
                if entity_id and entity.entity_id != entity_id:
                    continue
                return True

        # If no entity filter specified, match
        return not (entity_type or entity_id)

    async def get_context(
        self,
        situation: str,
        action_type: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_score: float = 0.7,
        max_examples: int = 3,
        include_failures: bool = False,
    ) -> ContextLearningResult:
        """
        Get context examples for an upcoming agent action.

        Convenience method that wraps search() with individual filter parameters.
        For more flexible filtering, use search() directly with a filters dict.

        Args:
            situation: Description of the current situation
            action_type: Filter by action type (tool name)
            domain: Filter by domain
            tags: Filter by tags
            min_score: Minimum outcome score (0-1) for positive examples
            max_examples: Maximum examples to return
            include_failures: Whether to include failed decisions (for anti-patterns)

        Returns:
            ContextLearningResult with formatted examples
        """
        # Build filters dict from individual parameters
        filters: Dict[str, Any] = {}
        if action_type:
            filters["action_type"] = action_type
        if domain:
            filters["domain"] = domain
        if tags:
            filters["tags"] = tags

        return await self.search(
            query=situation,
            filters=filters,
            min_score=min_score,
            include_failures=include_failures,
            limit=max_examples,
        )

    async def get_similar_decisions(
        self,
        action_type: str,
        max_examples: int = 3,
    ) -> List[RetrievedExample]:
        """
        Get similar decisions for a specific action type.

        Args:
            action_type: Action/tool type to find examples for
            max_examples: Maximum examples

        Returns:
            List of relevant decision examples
        """
        result = await self.get_context(
            situation=f"Using {action_type}",
            action_type=action_type,
            max_examples=max_examples,
        )
        return result.examples

    async def get_anti_patterns(
        self,
        situation: str,
        filters: Optional[Dict[str, Any]] = None,
        max_examples: int = 2,
    ) -> ContextLearningResult:
        """
        Get examples of past failures to warn the agent.

        Returns examples of past failures to avoid.

        Args:
            situation: Description of what the agent is about to do
            filters: Optional filter dictionary (see search() for supported keys)
            max_examples: Maximum anti-patterns to include

        Returns:
            ContextLearningResult with failure examples in anti_patterns field
        """
        return await self.search(
            query=situation,
            filters=filters,
            min_score=0.0,
            include_failures=True,
            limit=max_examples,
        )

    def _format_decision_for_agent(
        self,
        decision: OTSDecision,
        trajectory: OTSTrajectory,
        is_failure: bool = False,
    ) -> str:
        """Format a decision for inclusion in agent context."""
        lines = []

        # Context from trajectory
        if trajectory.metadata.domain:
            lines.append(f"**Domain**: {trajectory.metadata.domain}")

        # State/Context
        if decision.state and decision.state.context_summary:
            lines.append(f"**Context**: {decision.state.context_summary}")

        # Choice/Action
        lines.append(f"**Action**: {decision.choice.action}")
        if decision.choice.rationale:
            lines.append(f"**Reasoning**: {decision.choice.rationale}")

        # Outcome
        if is_failure:
            lines.append(f"**Result**: {decision.consequence.result_summary[:200] if decision.consequence.result_summary else 'Failed'}")
            # For failures, include lesson learned if available
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
