"""
Trajectory API endpoints for capturing and retrieving agent execution traces.

Trajectories enable continual learning by capturing what agents DID (decisions, reasoning, outcomes)
and making that experience searchable and learnable.

Supports cross-organization sharing with privacy-preserving anonymization.
"""

import json
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from letta.log import get_logger
from letta.schemas.trajectory import (
    DecisionSummary,
    Trajectory,
    TrajectoryCreate,
    TrajectorySearchRequest,
    TrajectorySearchResponse,
    TrajectorySearchResult,
    TrajectoryShareUpdate,
    TrajectoryUpdate,
    TrajectoryVisibility,
    TrajectoryWithDecisions,
    EntitySummary,
)
from pydantic import BaseModel
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server
from letta.server.server import SyncServer

logger = get_logger(__name__)

router = APIRouter(prefix="/trajectories", tags=["trajectories"])


# Langfuse export models


class LangfuseExportRequest(BaseModel):
    """Request body for Langfuse export"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None


class LangfuseExportResponse(BaseModel):
    """Response from Langfuse export"""
    success: bool
    langfuse_trace_id: str
    message: Optional[str] = None


@router.post("/", response_model=Trajectory, status_code=201)
async def create_trajectory(
    trajectory_create: TrajectoryCreate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Create a new trajectory.

    Trajectories are typically created when an agent starts executing a task.
    The data field is flexible - store whatever structure makes sense for your agent type.

    Processing (summary generation, scoring, embedding) happens asynchronously.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectory = await server.trajectory_manager.create_trajectory_async(trajectory_create=trajectory_create, actor=actor)
    return trajectory


@router.get("/stats", response_model=dict)
async def get_processing_stats(
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get processing statistics for trajectories.

    Returns counts by processing status:
    - total: Total number of trajectories
    - pending: Awaiting processing
    - processing: Currently being processed
    - completed: Successfully processed
    - failed: Processing failed

    Useful for monitoring async processing health.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    stats = await server.trajectory_manager.get_processing_stats_async(actor=actor)
    return stats


@router.get("/queue", response_model=dict)
async def get_queue_status(
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get current queue status for active background processing tasks.

    Returns:
    - active_count: Number of trajectories currently being processed in background
    - tasks: List of active tasks with their status

    Useful for monitoring and debugging async processing.
    """
    # No actor needed - queue status is server-level
    queue_status = server.trajectory_manager.get_queue_status()
    return queue_status


@router.get("/", response_model=List[Trajectory])
async def list_trajectories(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    min_score: Optional[float] = Query(None, description="Minimum outcome score (0-1)", ge=0.0, le=1.0),
    max_score: Optional[float] = Query(None, description="Maximum outcome score (0-1)", ge=0.0, le=1.0),
    limit: int = Query(50, description="Maximum number of results", ge=1, le=500),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    List trajectories with optional filtering.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectories = await server.trajectory_manager.list_trajectories_async(
        actor=actor,
        agent_id=agent_id,
        min_score=min_score,
        max_score=max_score,
        limit=limit,
        offset=offset,
    )
    return trajectories


@router.patch("/{trajectory_id}", response_model=Trajectory)
async def update_trajectory(
    trajectory_id: str,
    trajectory_update: TrajectoryUpdate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Update an existing trajectory.

    Typically used to:
    - Add more data as execution progresses
    - Update outcome information when complete
    - Trigger reprocessing (set searchable_summary=None to regenerate)
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectory = await server.trajectory_manager.update_trajectory_async(
        trajectory_id=trajectory_id,
        trajectory_update=trajectory_update,
        actor=actor,
    )

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    return trajectory


@router.delete("/{trajectory_id}", status_code=204)
async def delete_trajectory(
    trajectory_id: str,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Delete a trajectory.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    success = await server.trajectory_manager.delete_trajectory_async(trajectory_id=trajectory_id, actor=actor)

    if not success:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")


@router.post("/search", response_model=TrajectorySearchResponse)
async def search_trajectories(
    search_request: TrajectorySearchRequest = Body(...),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Search for similar trajectories using semantic similarity.

    This is the core retrieval mechanism for context learning:
    - Find trajectories where the agent handled similar situations
    - Filter by outcome score to find successes (min_score=0.7) or failures (max_score=0.4)
    - Use results as few-shot examples for current execution

    Cross-Organization Support:
    Set `include_cross_org=true` and specify `domain_type` to include anonymized
    trajectories from other organizations that share the same domain.

    Example:
    {
      "query": "User wants story about consciousness upload with identity fragmentation",
      "agent_id": "agent-123",
      "domain_type": "story_agent",
      "include_cross_org": true,
      "min_score": 0.7,
      "limit": 10
    }

    Returns trajectories ordered by semantic similarity to the query.
    Cross-org results have `visibility: "anonymized"` and redacted content.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    results = await server.trajectory_manager.search_trajectories_async(search_request=search_request, actor=actor)
    return TrajectorySearchResponse(results=results, query=search_request.query)


@router.patch("/{trajectory_id}/sharing", response_model=Trajectory)
async def update_trajectory_sharing(
    trajectory_id: str,
    share_update: TrajectoryShareUpdate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Enable or disable cross-organization sharing for a trajectory.

    When enabled (share_cross_org=true):
    - Trajectory becomes searchable by other organizations
    - Other orgs see an anonymized version (PII redacted, identifiers hashed)
    - Your organization still sees full detail

    Privacy protection:
    - Message content is redacted
    - Tool arguments are redacted
    - Identifiers (agent_id, org_id) are hashed
    - Only learning signal preserved (summary, score, tags, structural metadata)
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectory = await server.trajectory_manager.set_trajectory_sharing_async(
        trajectory_id=trajectory_id,
        share_cross_org=share_update.share_cross_org,
        actor=actor,
    )

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    return trajectory


@router.get("/by-domain/{domain_type}", response_model=List[TrajectorySearchResult])
async def list_trajectories_by_domain(
    domain_type: str,
    include_cross_org: bool = Query(False, description="Include anonymized cross-org trajectories"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    List trajectories by domain type with optional cross-org inclusion.

    Domain types are custom categories like "story_agent", "code_agent", "research_agent"
    that group trajectories for cross-organization sharing.

    If include_cross_org=true:
    - Same-org trajectories: full detail, visibility="full"
    - Cross-org trajectories: anonymized, visibility="anonymized"

    Cross-org trajectories are only included if they have share_cross_org=true.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    results = await server.trajectory_manager.list_trajectories_by_domain_async(
        domain_type=domain_type,
        actor=actor,
        include_cross_org=include_cross_org,
        limit=limit,
    )
    return results


@router.post("/{trajectory_id}/process", response_model=Trajectory)
async def process_trajectory(
    trajectory_id: str,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Manually trigger LLM processing for a trajectory.

    This generates:
    - searchable_summary: Natural language summary for search
    - outcome_score: Quality rating 0-1
    - score_reasoning: Explanation of the score
    - embedding: Vector for similarity search

    Normally happens automatically, but you can trigger it manually if needed.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectory = await server.trajectory_manager.process_trajectory_async(trajectory_id=trajectory_id, actor=actor)

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    return trajectory


@router.post("/{trajectory_id}/export/langfuse", response_model=LangfuseExportResponse)
async def export_trajectory_to_langfuse(
    trajectory_id: str,
    export_request: LangfuseExportRequest = Body(default=LangfuseExportRequest()),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Export a trajectory to Langfuse for visualization.

    This converts the Letta trajectory to OTS format and sends it to Langfuse,
    where you can visualize:
    - Trace timeline showing turn progression
    - LLM generations for each turn
    - Tool call spans with inputs/outputs
    - Quality scores and metadata

    Requires Langfuse environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY

    The export is one-way (OTS -> Langfuse) and preserves:
    - Turn structure and messages
    - Decision details (tool calls, arguments)
    - Quality scores and tags
    - Timing information

    Note: Decision alternatives and credit assignment (OTS-specific features)
    are included in span metadata but may not be fully visualized in Langfuse.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectory = await server.trajectory_manager.get_trajectory_async(trajectory_id=trajectory_id, actor=actor)

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    try:
        # Convert run data to OTS format
        from letta.trajectories.ots import OTSAdapter, LangfuseExporter

        ots_trajectory = OTSAdapter.from_letta_run(
            trajectory.data,
            agent_id=trajectory.agent_id,
            extract_decisions=True,
        )

        # Export to Langfuse
        exporter = LangfuseExporter()
        trace_id = await exporter.export_trajectory(
            trajectory=ots_trajectory,
            user_id=export_request.user_id,
            session_id=export_request.session_id,
            tags=export_request.tags,
        )

        return LangfuseExportResponse(
            success=True,
            langfuse_trace_id=trace_id,
            message=f"Trajectory exported to Langfuse. View at https://cloud.langfuse.com/trace/{trace_id}",
        )

    except ImportError as e:
        logger.error(f"Langfuse import failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Langfuse package not installed. Install with: pip install langfuse",
        )
    except Exception as e:
        logger.error(f"Langfuse export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export trajectory to Langfuse: {str(e)}",
        )


# Analytics endpoints


class TrajectoryWithEmbedding(BaseModel):
    """Trajectory with embedding included for visualization"""
    id: str
    agent_id: str
    searchable_summary: Optional[str]
    outcome_score: Optional[float]
    tags: Optional[List[str]]
    task_category: Optional[str]
    complexity_level: Optional[str]
    embedding: Optional[List[float]]
    created_at: str
    processing_status: str
    data: dict  # Include metadata for turn counts, etc.


class AnalyticsAggregations(BaseModel):
    """Aggregated statistics for analytics dashboard"""
    total_count: int
    score_distribution: dict  # bins -> counts
    turn_distribution: dict  # turn counts -> frequency
    tool_usage: dict  # tool name -> count
    tags_frequency: dict  # tag -> count
    category_breakdown: dict  # category -> count
    complexity_breakdown: dict  # complexity -> count
    daily_counts: List[dict]  # date -> count, avg_score
    agent_stats: dict  # agent_id -> stats


@router.get("/analytics/embeddings", response_model=List[TrajectoryWithEmbedding])
async def get_trajectories_with_embeddings(
    limit: int = Query(500, ge=1, le=1000, description="Maximum number of trajectories to return"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Filter by minimum outcome score"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get trajectories with embeddings for visualization.

    Returns trajectory data including the embedding vectors needed for
    semantic map visualization. Embeddings are normally excluded from
    API responses but included here for analytics.

    Use this endpoint to build 2D semantic maps where trajectories cluster
    by similarity.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)

    # Get trajectories from database with embeddings
    from letta.orm.trajectory import Trajectory
    from sqlalchemy import select
    from letta.server.db import db_registry

    async with db_registry.async_session() as session:
        query = select(Trajectory).where(
            Trajectory.organization_id == actor.organization_id
        )

        if agent_id:
            query = query.where(Trajectory.agent_id == agent_id)
        if min_score is not None:
            query = query.where(Trajectory.outcome_score >= min_score)

        # Only return processed trajectories with embeddings
        query = query.where(
            Trajectory.processing_status == "completed",
            Trajectory.embedding.isnot(None)
        ).limit(limit)

        result = await session.execute(query)
        trajectories = result.scalars().all()

        return [
            TrajectoryWithEmbedding(
                id=t.id,
                agent_id=t.agent_id,
                searchable_summary=t.searchable_summary,
                outcome_score=t.outcome_score,
                tags=t.tags,
                task_category=t.task_category,
                complexity_level=t.complexity_level,
                embedding=t.embedding,
                created_at=t.created_at.isoformat(),
                processing_status=t.processing_status,
                data=t.data,
            )
            for t in trajectories
        ]


@router.get("/analytics/aggregations", response_model=AnalyticsAggregations)
async def get_analytics_aggregations(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get aggregated statistics for analytics dashboard.

    Returns pre-computed aggregations for charts and graphs:
    - Score distribution (histogram bins)
    - Turn count distribution
    - Tool usage frequency
    - Tags word cloud data
    - Category and complexity breakdowns
    - Time-series data (daily counts and scores)
    - Per-agent statistics
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)

    from letta.orm.trajectory import Trajectory
    from sqlalchemy import select, func
    from letta.server.db import db_registry
    from collections import Counter, defaultdict
    from datetime import datetime, timedelta, timezone

    async with db_registry.async_session() as session:
        # Base query
        query = select(Trajectory).where(
            Trajectory.organization_id == actor.organization_id
        )
        if agent_id:
            query = query.where(Trajectory.agent_id == agent_id)

        result = await session.execute(query)
        trajectories = result.scalars().all()

        total_count = len(trajectories)

        # Score distribution (10 bins: 0-0.1, 0.1-0.2, ..., 0.9-1.0)
        score_bins = defaultdict(int)
        for t in trajectories:
            if t.outcome_score is not None:
                bin_idx = min(int(t.outcome_score * 10), 9)
                bin_range = f"{bin_idx * 0.1:.1f}-{(bin_idx + 1) * 0.1:.1f}"
                score_bins[bin_range] += 1

        # Turn distribution
        turn_counts = Counter()
        for t in trajectories:
            turn_count = len(t.data.get("turns", []))
            turn_counts[turn_count] += 1

        # Tool usage
        tool_usage = Counter()
        for t in trajectories:
            tools = t.data.get("metadata", {}).get("tools_used", [])
            for tool in tools:
                tool_usage[tool] += 1

        # Tags frequency
        tags_freq = Counter()
        for t in trajectories:
            if t.tags:
                for tag in t.tags:
                    tags_freq[tag] += 1

        # Category breakdown
        category_breakdown = Counter()
        for t in trajectories:
            if t.task_category:
                category_breakdown[t.task_category] += 1

        # Complexity breakdown
        complexity_breakdown = Counter()
        for t in trajectories:
            if t.complexity_level:
                complexity_breakdown[t.complexity_level] += 1

        # Daily counts and scores (last 30 days)
        daily_data = defaultdict(lambda: {"count": 0, "total_score": 0, "score_count": 0})
        cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=30)

        for t in trajectories:
            if t.created_at >= cutoff_date:
                date_key = t.created_at.date().isoformat()
                daily_data[date_key]["count"] += 1
                if t.outcome_score is not None:
                    daily_data[date_key]["total_score"] += t.outcome_score
                    daily_data[date_key]["score_count"] += 1

        # Calculate averages
        daily_counts = []
        for date_str in sorted(daily_data.keys()):
            data = daily_data[date_str]
            avg_score = data["total_score"] / data["score_count"] if data["score_count"] > 0 else None
            daily_counts.append({
                "date": date_str,
                "count": data["count"],
                "avg_score": avg_score,
            })

        # Agent stats
        agent_stats = defaultdict(lambda: {"count": 0, "total_score": 0, "score_count": 0})
        for t in trajectories:
            agent_stats[t.agent_id]["count"] += 1
            if t.outcome_score is not None:
                agent_stats[t.agent_id]["total_score"] += t.outcome_score
                agent_stats[t.agent_id]["score_count"] += 1

        # Calculate agent averages
        agent_stats_formatted = {}
        for agent_id, stats in agent_stats.items():
            avg_score = stats["total_score"] / stats["score_count"] if stats["score_count"] > 0 else None
            agent_stats_formatted[agent_id] = {
                "count": stats["count"],
                "avg_score": avg_score,
            }

        return AnalyticsAggregations(
            total_count=total_count,
            score_distribution=dict(score_bins),
            turn_distribution=dict(turn_counts),
            tool_usage=dict(tool_usage),
            tags_frequency=dict(tags_freq),
            category_breakdown=dict(category_breakdown),
            complexity_breakdown=dict(complexity_breakdown),
            daily_counts=daily_counts,
            agent_stats=agent_stats_formatted,
        )


# Pure OTS Analytics (no LLM enrichment required)


class OTSAnalyticsResponse(BaseModel):
    """Analytics computed purely from OTS trajectory data (no LLM enrichment)."""
    decision_success_rate: dict  # action -> success rate (0-1)
    action_frequency: dict  # action -> count
    decision_type_breakdown: dict  # decision type -> count
    turn_distribution: dict  # turn count -> frequency
    error_type_frequency: dict  # error type -> count
    trajectory_outcomes: dict  # outcome type -> count
    total_trajectories: int
    total_turns: int
    total_decisions: int
    total_messages: int
    avg_turns_per_trajectory: float
    avg_decisions_per_turn: float
    overall_success_rate: float


@router.get("/analytics/ots", response_model=OTSAnalyticsResponse)
async def get_ots_analytics(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    limit: int = Query(500, ge=1, le=1000, description="Maximum trajectories to analyze"),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get pure OTS analytics (no LLM enrichment required).

    These analytics are computed directly from the OTS trajectory structure:
    - Decision success rates per action
    - Action frequency distribution
    - Decision type breakdown (tool_selection, parameter_choice, etc.)
    - Error type frequency
    - Trajectory outcome distribution

    Unlike the enriched analytics, these work immediately without waiting
    for LLM processing and don't require API calls.

    Useful for:
    - Monitoring tool call patterns
    - Identifying error hotspots
    - Tracking decision success rates
    - Understanding agent behavior at the decision level
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)

    # Import OTS analytics functions
    from ots.analytics import get_pure_ots_analytics
    from ots.models import (
        OTSTrajectory,
        OTSTurn,
        OTSDecision,
        OTSChoice,
        OTSConsequence,
        OTSMetadata,
        OTSContext,
        DecisionType,
        OutcomeType,
    )
    from datetime import datetime, timezone

    from letta.orm.trajectory import Trajectory
    from sqlalchemy import select
    from letta.server.db import db_registry

    async with db_registry.async_session() as session:
        # Get trajectories
        query = select(Trajectory).where(
            Trajectory.organization_id == actor.organization_id
        )
        if agent_id:
            query = query.where(Trajectory.agent_id == agent_id)
        query = query.limit(limit)

        result = await session.execute(query)
        trajectories_orm = result.scalars().all()

        # Convert Letta trajectories to OTS format for analytics
        ots_trajectories = []
        for traj_orm in trajectories_orm:
            ots_traj = _letta_to_ots_for_analytics(traj_orm)
            ots_trajectories.append(ots_traj)

        # Compute OTS analytics
        analytics = get_pure_ots_analytics(ots_trajectories)

        return OTSAnalyticsResponse(
            decision_success_rate=analytics.decision_success_rate,
            action_frequency=analytics.action_frequency,
            decision_type_breakdown=analytics.decision_type_breakdown,
            turn_distribution=analytics.turn_distribution,
            error_type_frequency=analytics.error_type_frequency,
            trajectory_outcomes=analytics.trajectory_outcomes,
            total_trajectories=analytics.total_trajectories,
            total_turns=analytics.total_turns,
            total_decisions=analytics.total_decisions,
            total_messages=analytics.total_messages,
            avg_turns_per_trajectory=analytics.avg_turns_per_trajectory,
            avg_decisions_per_turn=analytics.avg_decisions_per_turn,
            overall_success_rate=analytics.overall_success_rate,
        )


def _letta_to_ots_for_analytics(trajectory_orm) -> "OTSTrajectory":
    """
    Convert Letta trajectory ORM to OTS format for analytics.

    This is a lightweight conversion focused on extracting decisions
    for analytics purposes.
    """
    from ots.models import (
        OTSTrajectory,
        OTSTurn,
        OTSDecision,
        OTSChoice,
        OTSConsequence,
        OTSMetadata,
        OTSContext,
        OTSMessage,
        OTSMessageContent,
        DecisionType,
        OutcomeType,
        MessageRole,
        ContentType,
    )
    from datetime import datetime, timezone

    data = trajectory_orm.data or {}
    turns_data = data.get("turns", [])

    ots_turns = []
    for turn_idx, turn in enumerate(turns_data):
        messages = turn.get("messages", [])

        # Extract decisions from tool calls
        ots_decisions = []
        ots_messages = []

        for msg_idx, msg in enumerate(messages):
            # Convert messages
            role_map = {
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT,
                "system": MessageRole.SYSTEM,
                "tool": MessageRole.TOOL,
            }
            role = role_map.get(msg.get("role", ""), MessageRole.ASSISTANT)

            # Handle content that can be string or list of content blocks
            raw_content = msg.get("content", "")
            if isinstance(raw_content, list):
                # Extract text from content blocks (e.g., reasoning blocks)
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict):
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                text_content = "\n".join(text_parts)
            else:
                text_content = raw_content if isinstance(raw_content, str) else ""

            ots_messages.append(
                OTSMessage(
                    role=role,
                    timestamp=datetime.now(timezone.utc),
                    content=OTSMessageContent(
                        type=ContentType.TEXT,
                        text=text_content,
                    ),
                )
            )

            # Extract tool calls as decisions
            tool_calls = msg.get("tool_calls") or []
            for tc_idx, tc in enumerate(tool_calls):
                tool_name = tc.get("function", {}).get("name", "unknown")
                tool_call_id = tc.get("id")

                # Determine success from tool response
                success = _extract_tool_success_for_analytics(messages, tool_call_id)

                # Parse arguments - can be dict or JSON string
                raw_args = tc.get("function", {}).get("arguments")
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                else:
                    parsed_args = {}

                ots_decisions.append(
                    OTSDecision(
                        decision_id=f"{trajectory_orm.id}-{turn_idx}-{msg_idx}-{tc_idx}",
                        decision_type=DecisionType.TOOL_SELECTION,
                        choice=OTSChoice(
                            action=tool_name,
                            arguments=parsed_args,
                        ),
                        consequence=OTSConsequence(
                            success=success if success is not None else True,
                            result_summary=None,
                            error_type=_extract_error_type(messages, tool_call_id) if success is False else None,
                        ),
                    )
                )

        ots_turns.append(
            OTSTurn(
                turn_id=turn_idx,
                timestamp=datetime.now(timezone.utc),
                messages=ots_messages,
                decisions=ots_decisions,
            )
        )

    # Determine outcome from score
    score = trajectory_orm.outcome_score
    if score is not None:
        if score >= 0.7:
            outcome = OutcomeType.SUCCESS
        elif score >= 0.4:
            outcome = OutcomeType.PARTIAL_SUCCESS
        else:
            outcome = OutcomeType.FAILURE
    else:
        outcome = OutcomeType.SUCCESS  # Default

    return OTSTrajectory(
        trajectory_id=trajectory_orm.id,
        metadata=OTSMetadata(
            task_description=trajectory_orm.searchable_summary or "",
            domain=trajectory_orm.domain_type,
            timestamp_start=trajectory_orm.created_at or datetime.now(timezone.utc),
            agent_id=trajectory_orm.agent_id,
            outcome=outcome,
            tags=trajectory_orm.tags or [],
        ),
        context=OTSContext(),
        turns=ots_turns,
        final_reward=trajectory_orm.outcome_score,
    )


def _extract_tool_success_for_analytics(messages, tool_call_id):
    """Extract tool success from messages."""
    if not tool_call_id:
        return None

    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
            content = msg.get("content", "")
            if isinstance(content, str):
                if any(err in content.lower() for err in ["error", "failed", "exception", "traceback"]):
                    return False
            return True

    return None


def _extract_error_type(messages, tool_call_id):
    """Extract error type from tool response."""
    if not tool_call_id:
        return None

    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
            content = msg.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if "timeout" in content_lower:
                    return "Timeout"
                elif "validation" in content_lower:
                    return "ValidationError"
                elif "permission" in content_lower or "auth" in content_lower:
                    return "PermissionError"
                elif "not found" in content_lower:
                    return "NotFoundError"
                elif "error" in content_lower or "exception" in content_lower:
                    return "Error"

    return None


def _extract_decisions_from_trajectory(trajectory: Trajectory) -> List[DecisionSummary]:
    """
    Extract OTS-style decisions from trajectory data.

    If the trajectory has stored OTS decisions (from GPT-5 mini extraction),
    use those for rich data (rationale, alternatives, confidence).
    Otherwise, fall back to programmatic extraction from tool calls.
    """
    # Check if we have stored OTS decisions (rich data from LLM extraction)
    if trajectory.ots_decisions:
        return _convert_ots_decisions_to_summaries(trajectory.ots_decisions, trajectory.id)

    # Fallback: programmatic extraction from tool calls
    return _extract_decisions_programmatic(trajectory)


def _convert_ots_decisions_to_summaries(ots_decisions: List[Dict], trajectory_id: str) -> List[DecisionSummary]:
    """
    Convert stored OTS decisions to DecisionSummary objects.

    OTS decisions have rich data including:
    - rationale: Why this decision was made
    - alternatives: What other options were considered
    - confidence: How confident the agent was
    - context_summary: What the agent understood at decision time
    """
    decisions = []
    for idx, d in enumerate(ots_decisions):
        choice = d.get("choice", {}) or {}
        consequence = d.get("consequence", {}) or {}
        state = d.get("state", {}) or {}
        alternatives_data = d.get("alternatives", {})

        # Convert alternatives to AlternativeConsidered objects
        alternatives_considered = None
        if alternatives_data and alternatives_data.get("considered"):
            from letta.schemas.trajectory import AlternativeConsidered
            alternatives_considered = [
                AlternativeConsidered(
                    action=alt.get("action", ""),
                    rationale=alt.get("rationale"),
                    rejected_reason=alt.get("rejected_reason"),
                )
                for alt in alternatives_data.get("considered", [])
            ]

        decisions.append(DecisionSummary(
            decision_id=d.get("decision_id", f"{trajectory_id}-ots-{idx}"),
            turn_index=d.get("turn_index", 0),
            decision_type=d.get("decision_type", "tool_selection"),
            action=choice.get("action", ""),
            arguments=choice.get("arguments"),
            rationale=choice.get("rationale"),
            alternatives_considered=alternatives_considered,
            confidence=choice.get("confidence"),
            context_summary=state.get("context_summary"),
            success=consequence.get("success", True),
            error_type=consequence.get("error_type"),
            result_summary=consequence.get("result_summary"),
        ))

    return decisions


def _extract_decisions_programmatic(trajectory: Trajectory) -> List[DecisionSummary]:
    """
    Extract decisions programmatically from tool calls in trajectory data.

    This is the fallback when no stored OTS decisions are available.
    Provides basic decision info without LLM-extracted rationale/alternatives.
    """
    decisions = []
    data = trajectory.data or {}
    turns = data.get("turns", [])

    for turn_idx, turn in enumerate(turns):
        messages = turn.get("messages", [])

        for msg_idx, msg in enumerate(messages):
            tool_calls = msg.get("tool_calls") or []

            for tc_idx, tc in enumerate(tool_calls):
                tool_name = tc.get("function", {}).get("name", "unknown")
                tool_call_id = tc.get("id")
                arguments = tc.get("function", {}).get("arguments")

                # Parse arguments if it's a string
                if isinstance(arguments, str):
                    try:
                        import json
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw": arguments}

                # Determine success from tool response
                success = _extract_tool_success_for_analytics(messages, tool_call_id)
                if success is None:
                    success = True  # Default to success if no response found

                error_type = None
                result_summary = None

                if not success:
                    error_type = _extract_error_type(messages, tool_call_id)

                # Try to extract result summary from tool response
                for response_msg in messages:
                    if response_msg.get("role") == "tool" and response_msg.get("tool_call_id") == tool_call_id:
                        content = response_msg.get("content", "")
                        if isinstance(content, str) and len(content) > 0:
                            # Truncate long results
                            result_summary = content[:200] + "..." if len(content) > 200 else content
                        break

                decisions.append(DecisionSummary(
                    decision_id=f"{trajectory.id}-{turn_idx}-{msg_idx}-{tc_idx}",
                    turn_index=turn_idx,
                    decision_type="tool_selection",
                    action=tool_name,
                    arguments=arguments,
                    rationale=None,  # Not available without LLM extraction
                    alternatives_considered=None,
                    confidence=None,
                    context_summary=None,
                    success=success,
                    error_type=error_type,
                    result_summary=result_summary,
                ))

    return decisions


def _extract_entities_from_trajectory(trajectory: Trajectory) -> List[EntitySummary]:
    """
    Extract entities from stored OTS data.

    Returns entities extracted during trajectory processing (LLM + programmatic).
    Returns empty list if no OTS entities were extracted.
    """
    if not trajectory.ots_entities:
        return []

    entities = []
    for e in trajectory.ots_entities:
        entities.append(EntitySummary(
            entity_type=e.get("entity_type", "concept"),
            entity_id=e.get("entity_id", ""),
            name=e.get("name", ""),
            metadata=e.get("metadata"),
        ))

    return entities


# IMPORTANT: This route must be at the END of the file because it's a catch-all
# that matches any string as trajectory_id. If placed earlier, it shadows
# specific routes like /analytics/ots, /analytics/embeddings, etc.
@router.get("/{trajectory_id}", response_model=TrajectoryWithDecisions)
async def get_trajectory(
    trajectory_id: str,
    include_decisions: bool = Query(True, description="Include extracted OTS-style decisions"),
    include_entities: bool = Query(True, description="Include extracted entities"),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get a single trajectory by ID with optional OTS data extraction.

    When OTS LLM processing is enabled (GPT-5 mini), returns rich data:
    - decisions: Tool calls with rationale, alternatives considered, confidence
    - entities: Services, files, concepts, etc. mentioned in the trajectory

    Falls back to programmatic extraction if OTS data not available.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectory = await server.trajectory_manager.get_trajectory_async(trajectory_id=trajectory_id, actor=actor)

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    # Extract decisions if requested
    decisions = []
    if include_decisions:
        decisions = _extract_decisions_from_trajectory(trajectory)

    # Extract entities if requested
    entities = []
    if include_entities:
        entities = _extract_entities_from_trajectory(trajectory)

    # Return trajectory with decisions and entities
    return TrajectoryWithDecisions(
        id=trajectory.id,
        agent_id=trajectory.agent_id,
        domain_type=trajectory.domain_type,
        share_cross_org=trajectory.share_cross_org,
        data=trajectory.data,
        searchable_summary=trajectory.searchable_summary,
        outcome_score=trajectory.outcome_score,
        score_reasoning=trajectory.score_reasoning,
        tags=trajectory.tags,
        task_category=trajectory.task_category,
        complexity_level=trajectory.complexity_level,
        trajectory_metadata=trajectory.trajectory_metadata,
        ots_decisions=trajectory.ots_decisions,
        ots_entities=trajectory.ots_entities,
        processing_status=trajectory.processing_status,
        processing_started_at=trajectory.processing_started_at,
        processing_completed_at=trajectory.processing_completed_at,
        processing_error=trajectory.processing_error,
        created_at=trajectory.created_at,
        updated_at=trajectory.updated_at,
        organization_id=trajectory.organization_id,
        decisions=decisions,
        entities=entities,
    )
