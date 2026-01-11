"""
OTS Storage utilities for storing and retrieving OTS trajectories.

Integrates OTS format with Letta's existing trajectory infrastructure,
enabling storage, retrieval, and semantic search of decision traces.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from letta.log import get_logger
from letta.schemas.trajectory import Trajectory as LettaTrajectory, TrajectoryCreate
from letta.schemas.trajectory_annotation import (
    EvaluatorType,
    TrajectoryAnnotation,
    TrajectoryAnnotationCreate,
)
from letta.schemas.user import User as PydanticUser
from letta.services.annotation_manager import AnnotationManager
from letta.services.trajectory_manager import TrajectoryManager
from letta.trajectories.ots.adapter import OTSAdapter
from letta.trajectories.ots.decision_extractor import DecisionExtractor
from letta.trajectories.ots.models import (
    OTSAnnotation,
    OTSDecision,
    OTSDecisionEvaluation,
    OTSTrajectory,
)

logger = get_logger(__name__)


class OTSStore:
    """
    Storage layer for OTS trajectories.

    Bridges the OTS format with Letta's trajectory storage, providing:
    - Store/retrieve OTS trajectories
    - Store/retrieve annotations at trajectory/turn/decision levels
    - Semantic search for similar decisions
    - Decision-level embedding generation

    Usage:
        store = OTSStore()
        trajectory_id = await store.store_ots_trajectory(ots_traj, actor)
        ots_traj = await store.get_ots_trajectory(trajectory_id, actor)
    """

    def __init__(
        self,
        trajectory_manager: Optional[TrajectoryManager] = None,
        annotation_manager: Optional[AnnotationManager] = None,
    ):
        """
        Initialize OTS store.

        Args:
            trajectory_manager: Trajectory manager instance (creates new if None)
            annotation_manager: Annotation manager instance (creates new if None)
        """
        self.trajectory_manager = trajectory_manager or TrajectoryManager()
        self.annotation_manager = annotation_manager or AnnotationManager()
        self.adapter = OTSAdapter()
        self.decision_extractor = DecisionExtractor()

    async def store_ots_trajectory(
        self,
        ots_trajectory: OTSTrajectory,
        actor: PydanticUser,
        auto_process: bool = True,
    ) -> str:
        """
        Store an OTS trajectory in Letta's storage.

        Args:
            ots_trajectory: OTS trajectory to store
            actor: User storing the trajectory
            auto_process: Whether to trigger async LLM processing

        Returns:
            ID of the stored trajectory
        """
        # Convert OTS to Letta storage format
        letta_data = self._ots_to_letta_data(ots_trajectory)

        # Create trajectory in Letta storage
        trajectory_create = TrajectoryCreate(
            agent_id=ots_trajectory.metadata.agent_id,
            data=letta_data,
        )

        letta_trajectory = await self.trajectory_manager.create_and_process_async(
            trajectory_create,
            actor,
            auto_process=auto_process,
        )

        logger.info(f"Stored OTS trajectory as {letta_trajectory.id}")

        # Store any embedded evaluations as annotations
        await self._store_decision_evaluations(
            letta_trajectory.id,
            ots_trajectory,
            actor,
        )

        return letta_trajectory.id

    async def get_ots_trajectory(
        self,
        trajectory_id: str,
        actor: PydanticUser,
        include_annotations: bool = True,
    ) -> Optional[OTSTrajectory]:
        """
        Retrieve a trajectory in OTS format.

        Args:
            trajectory_id: ID of the trajectory
            actor: User requesting the trajectory
            include_annotations: Whether to include annotations in decisions

        Returns:
            OTS trajectory or None if not found
        """
        letta_trajectory = await self.trajectory_manager.get_trajectory_async(
            trajectory_id,
            actor,
        )

        if not letta_trajectory:
            return None

        # Convert run data to OTS format
        ots_trajectory = OTSAdapter.from_letta_run(
            letta_trajectory.data,
            agent_id=letta_trajectory.agent_id,
        )

        # Enrich with annotations if requested
        if include_annotations:
            ots_trajectory = await self._enrich_with_annotations(
                ots_trajectory,
                actor,
            )

        return ots_trajectory

    async def store_annotation(
        self,
        annotation: OTSAnnotation,
        actor: PydanticUser,
    ) -> str:
        """
        Store an OTS annotation.

        Args:
            annotation: OTS annotation to store
            actor: User storing the annotation

        Returns:
            ID of the stored annotation
        """
        annotation_create = TrajectoryAnnotationCreate(
            trajectory_id=annotation.trajectory_id,
            turn_id=annotation.turn_id,
            decision_id=annotation.decision_id,
            evaluator_id=annotation.evaluator.id,
            evaluator_type=EvaluatorType(annotation.evaluator.type.value),
            evaluator_version=annotation.evaluator.version,
            score=annotation.score,
            label=annotation.label,
            feedback=annotation.feedback,
        )

        letta_annotation = await self.annotation_manager.create_annotation_async(
            annotation_create,
            actor,
        )

        return letta_annotation.id

    async def get_trajectory_annotations(
        self,
        trajectory_id: str,
        actor: PydanticUser,
    ) -> List[TrajectoryAnnotation]:
        """
        Get all annotations for a trajectory.

        Args:
            trajectory_id: ID of the trajectory
            actor: User requesting annotations

        Returns:
            List of annotations at all granularity levels
        """
        return await self.annotation_manager.get_trajectory_annotations_async(
            trajectory_id,
            actor,
        )

    async def annotate_decision(
        self,
        trajectory_id: str,
        turn_id: int,
        decision_id: str,
        evaluator_id: str,
        evaluator_type: EvaluatorType,
        score: float,
        label: Optional[str] = None,
        feedback: Optional[str] = None,
        actor: Optional[PydanticUser] = None,
    ) -> str:
        """
        Add an annotation to a specific decision.

        Args:
            trajectory_id: ID of the trajectory
            turn_id: Turn index containing the decision
            decision_id: ID of the decision
            evaluator_id: ID of the evaluator
            evaluator_type: Type of evaluator
            score: Quality score (0-1)
            label: Optional categorical label
            feedback: Optional detailed feedback
            actor: User creating the annotation

        Returns:
            ID of the created annotation
        """
        annotation_create = TrajectoryAnnotationCreate(
            trajectory_id=trajectory_id,
            turn_id=turn_id,
            decision_id=decision_id,
            evaluator_id=evaluator_id,
            evaluator_type=evaluator_type,
            score=score,
            label=label,
            feedback=feedback,
        )

        letta_annotation = await self.annotation_manager.create_annotation_async(
            annotation_create,
            actor,
        )

        return letta_annotation.id

    async def search_similar_decisions(
        self,
        query: str,
        actor: PydanticUser,
        min_score: Optional[float] = None,
        domain_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Tuple[OTSDecision, float]]:
        """
        Search for similar decisions using semantic search.

        This searches trajectory summaries and returns the decisions
        from matching trajectories. For true decision-level search,
        decision embeddings would need to be stored separately.

        Args:
            query: Natural language query
            actor: User performing the search
            min_score: Minimum trajectory score threshold
            domain_type: Filter by domain type
            limit: Maximum results

        Returns:
            List of (decision, similarity_score) tuples
        """
        from letta.schemas.trajectory import TrajectorySearchRequest

        # Search trajectories
        search_request = TrajectorySearchRequest(
            query=query,
            min_score=min_score,
            domain_type=domain_type,
            limit=limit,
        )

        results = await self.trajectory_manager.search_trajectories_async(
            search_request,
            actor,
        )

        # Extract decisions from matching trajectories
        decisions = []
        for result in results:
            # Convert run data to OTS and get decisions
            ots_traj = OTSAdapter.from_letta_run(
                result.trajectory.data,
                agent_id=result.trajectory.agent_id,
            )
            for turn in ots_traj.turns:
                for decision in turn.decisions:
                    decisions.append((decision, result.similarity))

        return decisions[:limit]

    async def extract_and_store_decisions(
        self,
        trajectory_id: str,
        actor: PydanticUser,
        use_llm: bool = False,
    ) -> int:
        """
        Extract decisions from a stored trajectory and update storage.

        Args:
            trajectory_id: ID of the trajectory
            actor: User performing extraction
            use_llm: Whether to use LLM for rich extraction (rationale, alternatives)

        Returns:
            Number of decisions extracted
        """
        # Get trajectory
        letta_trajectory = await self.trajectory_manager.get_trajectory_async(
            trajectory_id,
            actor,
        )

        if not letta_trajectory:
            raise ValueError(f"Trajectory {trajectory_id} not found")

        # Convert and extract decisions
        ots_trajectory = OTSAdapter.from_letta_run(
            letta_trajectory.data,
            agent_id=letta_trajectory.agent_id,
            extract_decisions=True,
        )

        # Optionally enrich with LLM
        if use_llm:
            ots_trajectory = await self.decision_extractor.enrich_decisions(
                ots_trajectory
            )

        # Update the trajectory data with extracted decisions
        letta_data = self._ots_to_letta_data(ots_trajectory)

        from letta.schemas.trajectory import TrajectoryUpdate

        await self.trajectory_manager.update_trajectory_async(
            trajectory_id,
            TrajectoryUpdate(data=letta_data),
            actor,
        )

        # Count decisions
        decision_count = sum(len(turn.decisions) for turn in ots_trajectory.turns)
        logger.info(f"Extracted {decision_count} decisions from trajectory {trajectory_id}")

        return decision_count

    async def get_decision_by_id(
        self,
        trajectory_id: str,
        decision_id: str,
        actor: PydanticUser,
    ) -> Optional[OTSDecision]:
        """
        Get a specific decision by ID.

        Args:
            trajectory_id: ID of the trajectory
            decision_id: ID of the decision (e.g., "t0-d1")
            actor: User requesting the decision

        Returns:
            Decision or None if not found
        """
        ots_trajectory = await self.get_ots_trajectory(trajectory_id, actor)

        if not ots_trajectory:
            return None

        for turn in ots_trajectory.turns:
            for decision in turn.decisions:
                if decision.decision_id == decision_id:
                    return decision

        return None

    def _ots_to_letta_data(self, ots_trajectory: OTSTrajectory) -> Dict[str, Any]:
        """
        Convert OTS trajectory to Letta storage format.

        The OTS format is stored in the `data` JSONB field.
        """
        return ots_trajectory.to_dict()

    async def _store_decision_evaluations(
        self,
        trajectory_id: str,
        ots_trajectory: OTSTrajectory,
        actor: PydanticUser,
    ) -> int:
        """
        Store decision evaluations as annotations.

        If decisions have embedded evaluations, store them as linked annotations.
        """
        count = 0

        for turn in ots_trajectory.turns:
            for decision in turn.decisions:
                if decision.evaluation:
                    await self.annotation_manager.create_annotation_async(
                        TrajectoryAnnotationCreate(
                            trajectory_id=trajectory_id,
                            turn_id=turn.turn_id,
                            decision_id=decision.decision_id,
                            evaluator_id=decision.evaluation.evaluator_id,
                            evaluator_type=EvaluatorType.HEURISTIC,
                            score=decision.evaluation.score,
                            feedback=decision.evaluation.feedback,
                        ),
                        actor,
                    )
                    count += 1

        return count

    async def _enrich_with_annotations(
        self,
        ots_trajectory: OTSTrajectory,
        actor: PydanticUser,
    ) -> OTSTrajectory:
        """
        Enrich OTS trajectory with stored annotations.
        """
        annotations = await self.annotation_manager.get_trajectory_annotations_async(
            ots_trajectory.trajectory_id,
            actor,
        )

        # Index annotations by decision_id
        decision_annotations: Dict[str, List[TrajectoryAnnotation]] = {}
        for ann in annotations:
            if ann.decision_id:
                if ann.decision_id not in decision_annotations:
                    decision_annotations[ann.decision_id] = []
                decision_annotations[ann.decision_id].append(ann)

        # Apply to decisions
        for turn in ots_trajectory.turns:
            for decision in turn.decisions:
                if decision.decision_id in decision_annotations:
                    # Use the highest-scored annotation as the evaluation
                    best_ann = max(
                        decision_annotations[decision.decision_id],
                        key=lambda a: a.score,
                    )
                    decision.evaluation = OTSDecisionEvaluation(
                        evaluator_id=best_ann.evaluator_id,
                        score=best_ann.score,
                        feedback=best_ann.feedback,
                    )

        return ots_trajectory


async def store_ots_trajectory(
    ots_trajectory: OTSTrajectory,
    actor: PydanticUser,
    auto_process: bool = True,
) -> str:
    """
    Convenience function to store an OTS trajectory.

    Args:
        ots_trajectory: OTS trajectory to store
        actor: User storing the trajectory
        auto_process: Whether to trigger async LLM processing

    Returns:
        ID of the stored trajectory
    """
    store = OTSStore()
    return await store.store_ots_trajectory(ots_trajectory, actor, auto_process)


async def get_ots_trajectory(
    trajectory_id: str,
    actor: PydanticUser,
) -> Optional[OTSTrajectory]:
    """
    Convenience function to get an OTS trajectory.

    Args:
        trajectory_id: ID of the trajectory
        actor: User requesting the trajectory

    Returns:
        OTS trajectory or None if not found
    """
    store = OTSStore()
    return await store.get_ots_trajectory(trajectory_id, actor)
