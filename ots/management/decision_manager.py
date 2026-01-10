"""
Decision Manager for OTS trajectories.

Provides utilities for extracting decisions from trajectories,
generating embeddings, and searching by semantic similarity.

This is the core OTS logic - framework-specific storage should
wrap this with their own persistence layer.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ots.models import (
    DecisionType,
    OTSDecision,
    OTSTrajectory,
    OTSTurn,
)
from ots.protocols import EmbeddingProvider


@dataclass
class DecisionRecord:
    """
    Flattened decision with trajectory context.

    Used for storage and search - extracts key fields from nested
    OTSDecision structure into a flat, searchable format.
    """

    decision_id: str
    trajectory_id: str
    turn_index: int
    decision_index: int

    # Core decision info
    action: str
    decision_type: Optional[DecisionType] = None
    rationale: Optional[str] = None

    # Outcome
    success: Optional[bool] = None
    result_summary: Optional[str] = None
    error_type: Optional[str] = None

    # Search
    searchable_text: str = ""
    embedding: Optional[List[float]] = None

    # Optional metadata
    confidence: Optional[float] = None
    contribution_to_outcome: Optional[float] = None
    pivotal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "trajectory_id": self.trajectory_id,
            "turn_index": self.turn_index,
            "decision_index": self.decision_index,
            "action": self.action,
            "decision_type": self.decision_type.value if self.decision_type else None,
            "rationale": self.rationale,
            "success": self.success,
            "result_summary": self.result_summary,
            "error_type": self.error_type,
            "searchable_text": self.searchable_text,
            "confidence": self.confidence,
            "contribution_to_outcome": self.contribution_to_outcome,
            "pivotal": self.pivotal,
        }


@dataclass
class DecisionSearchResult:
    """Result from decision search with similarity score."""

    decision: DecisionRecord
    similarity: float
    trajectory_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.to_dict(),
            "similarity": self.similarity,
            "trajectory_summary": self.trajectory_summary,
        }


@dataclass
class DecisionFilter:
    """Filter criteria for decision search."""

    action: Optional[str] = None
    decision_type: Optional[DecisionType] = None
    success: Optional[bool] = None
    min_confidence: Optional[float] = None
    has_error: Optional[bool] = None
    pivotal_only: bool = False


class DecisionManager:
    """
    Manages decision extraction, embedding, and search.

    This is the core OTS logic. Framework-specific storage (Letta, LangChain, etc.)
    should wrap this class with their own persistence layer.

    Usage:
        manager = DecisionManager(embedding_provider=OpenAIEmbeddingProvider())

        # Extract decisions from trajectory
        records = manager.extract_decisions(trajectory)

        # Generate embeddings
        records = await manager.embed_decisions(records)

        # Search (with external storage)
        # Framework provides similarity search implementation
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        searchable_text_builder: Optional[Callable[[OTSDecision], str]] = None,
    ):
        """
        Initialize DecisionManager.

        Args:
            embedding_provider: Provider for generating embeddings
            searchable_text_builder: Custom function to build searchable text
                                    (defaults to _build_searchable_text)
        """
        self.embedding_provider = embedding_provider
        self._build_text = searchable_text_builder or self._build_searchable_text

    def extract_decisions(
        self,
        trajectory: OTSTrajectory,
        include_empty_turns: bool = False,
    ) -> List[DecisionRecord]:
        """
        Extract all decisions from a trajectory into flat records.

        Args:
            trajectory: The OTS trajectory to extract from
            include_empty_turns: If True, include turns without decisions

        Returns:
            List of DecisionRecord objects with searchable_text populated
        """
        records = []

        for turn_idx, turn in enumerate(trajectory.turns):
            if not turn.decisions and not include_empty_turns:
                continue

            for dec_idx, decision in enumerate(turn.decisions):
                record = self._decision_to_record(
                    trajectory_id=trajectory.trajectory_id,
                    turn_idx=turn_idx,
                    dec_idx=dec_idx,
                    decision=decision,
                )
                records.append(record)

        return records

    def extract_decisions_from_turn(
        self,
        trajectory_id: str,
        turn: OTSTurn,
        turn_index: int,
    ) -> List[DecisionRecord]:
        """
        Extract decisions from a single turn.

        Useful for incremental extraction during trajectory capture.

        Args:
            trajectory_id: Parent trajectory ID
            turn: The turn to extract from
            turn_index: Index of this turn in the trajectory

        Returns:
            List of DecisionRecord objects
        """
        return [
            self._decision_to_record(
                trajectory_id=trajectory_id,
                turn_idx=turn_index,
                dec_idx=dec_idx,
                decision=decision,
            )
            for dec_idx, decision in enumerate(turn.decisions)
        ]

    async def embed_decisions(
        self,
        records: List[DecisionRecord],
        batch_size: int = 100,
    ) -> List[DecisionRecord]:
        """
        Generate embeddings for decision records.

        Args:
            records: List of DecisionRecord objects
            batch_size: Number of records to embed in each batch

        Returns:
            Same records with embedding field populated
        """
        if not self.embedding_provider:
            return records

        # Process in batches for efficiency
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            texts = [r.searchable_text for r in batch]

            # Use batch embedding if available
            if hasattr(self.embedding_provider, "embed_batch"):
                embeddings = await self.embedding_provider.embed_batch(texts)
            else:
                # Fall back to individual embedding
                embeddings = [await self.embedding_provider.embed(t) for t in texts]

            for record, embedding in zip(batch, embeddings):
                record.embedding = embedding

        return records

    def embed_decisions_sync(
        self,
        records: List[DecisionRecord],
    ) -> List[DecisionRecord]:
        """
        Generate embeddings synchronously (for frameworks without async).

        Args:
            records: List of DecisionRecord objects

        Returns:
            Same records with embedding field populated
        """
        if not self.embedding_provider:
            return records

        for record in records:
            if hasattr(self.embedding_provider, "embed_sync"):
                record.embedding = self.embedding_provider.embed_sync(
                    record.searchable_text
                )

        return records

    def filter_records(
        self,
        records: List[DecisionRecord],
        filter_criteria: DecisionFilter,
    ) -> List[DecisionRecord]:
        """
        Filter decision records by criteria.

        Args:
            records: List of DecisionRecord objects
            filter_criteria: Filter criteria to apply

        Returns:
            Filtered list of records
        """
        result = records

        if filter_criteria.action:
            result = [r for r in result if r.action == filter_criteria.action]

        if filter_criteria.decision_type:
            result = [
                r for r in result if r.decision_type == filter_criteria.decision_type
            ]

        if filter_criteria.success is not None:
            result = [r for r in result if r.success == filter_criteria.success]

        if filter_criteria.min_confidence is not None:
            result = [
                r
                for r in result
                if r.confidence is not None
                and r.confidence >= filter_criteria.min_confidence
            ]

        if filter_criteria.has_error is not None:
            if filter_criteria.has_error:
                result = [r for r in result if r.error_type is not None]
            else:
                result = [r for r in result if r.error_type is None]

        if filter_criteria.pivotal_only:
            result = [r for r in result if r.pivotal]

        return result

    def _decision_to_record(
        self,
        trajectory_id: str,
        turn_idx: int,
        dec_idx: int,
        decision: OTSDecision,
    ) -> DecisionRecord:
        """Convert OTSDecision to flat DecisionRecord."""
        searchable_text = self._build_text(decision)

        # Extract credit assignment if available
        contribution = None
        pivotal = False
        if decision.credit_assignment:
            contribution = decision.credit_assignment.contribution_to_outcome
            pivotal = decision.credit_assignment.pivotal

        return DecisionRecord(
            decision_id=decision.decision_id,
            trajectory_id=trajectory_id,
            turn_index=turn_idx,
            decision_index=dec_idx,
            action=decision.choice.action,
            decision_type=decision.decision_type,
            rationale=decision.choice.rationale,
            success=decision.consequence.success if decision.consequence else None,
            result_summary=decision.consequence.result_summary
            if decision.consequence
            else None,
            error_type=decision.consequence.error_type if decision.consequence else None,
            searchable_text=searchable_text,
            confidence=decision.choice.confidence,
            contribution_to_outcome=contribution,
            pivotal=pivotal,
        )

    def _build_searchable_text(self, decision: OTSDecision) -> str:
        """
        Build text representation for embedding.

        Combines action, type, rationale, and outcome into a single
        searchable string.
        """
        parts = [f"Action: {decision.choice.action}"]

        if decision.decision_type:
            parts.append(f"Type: {decision.decision_type.value}")

        if decision.choice.rationale:
            # Truncate long rationales
            rationale = decision.choice.rationale[:500]
            parts.append(f"Rationale: {rationale}")

        if decision.consequence:
            outcome = "success" if decision.consequence.success else "failure"
            parts.append(f"Outcome: {outcome}")

            if decision.consequence.error_type:
                parts.append(f"Error: {decision.consequence.error_type}")

            if decision.consequence.result_summary:
                # Truncate long summaries
                summary = decision.consequence.result_summary[:200]
                parts.append(f"Result: {summary}")

        return " | ".join(parts)


# Utility functions for common operations


def count_decisions_by_action(records: List[DecisionRecord]) -> Dict[str, int]:
    """Count decisions by action name."""
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.action] = counts.get(record.action, 0) + 1
    return counts


def count_decisions_by_type(records: List[DecisionRecord]) -> Dict[str, int]:
    """Count decisions by decision type."""
    counts: Dict[str, int] = {}
    for record in records:
        dtype = record.decision_type.value if record.decision_type else "unknown"
        counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def compute_success_rate(records: List[DecisionRecord]) -> float:
    """Compute overall success rate of decisions."""
    with_outcome = [r for r in records if r.success is not None]
    if not with_outcome:
        return 0.0
    return sum(1 for r in with_outcome if r.success) / len(with_outcome)


def compute_success_rate_by_action(records: List[DecisionRecord]) -> Dict[str, float]:
    """Compute success rate per action."""
    by_action: Dict[str, List[bool]] = {}
    for record in records:
        if record.success is not None:
            if record.action not in by_action:
                by_action[record.action] = []
            by_action[record.action].append(record.success)

    return {
        action: sum(outcomes) / len(outcomes) if outcomes else 0.0
        for action, outcomes in by_action.items()
    }
