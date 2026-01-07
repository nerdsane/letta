"""
Trajectory API endpoints for capturing and retrieving agent execution traces.

Trajectories enable continual learning by capturing what agents DID (decisions, reasoning, outcomes)
and making that experience searchable and learnable.
"""

from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from letta.schemas.trajectory import (
    Trajectory,
    TrajectoryCreate,
    TrajectorySearchRequest,
    TrajectorySearchResponse,
    TrajectorySearchResult,
    TrajectoryUpdate,
)
from pydantic import BaseModel
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server
from letta.server.server import SyncServer

router = APIRouter(prefix="/trajectories", tags=["trajectories"])


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


@router.get("/{trajectory_id}", response_model=Trajectory)
async def get_trajectory(
    trajectory_id: str,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get a single trajectory by ID.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    trajectory = await server.trajectory_manager.get_trajectory_async(trajectory_id=trajectory_id, actor=actor)

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    return trajectory


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

    Example:
    {
      "query": "User wants story about consciousness upload with identity fragmentation",
      "agent_id": "agent-123",
      "min_score": 0.7,
      "limit": 3
    }

    Returns trajectories ordered by semantic similarity to the query.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    results = await server.trajectory_manager.search_trajectories_async(search_request=search_request, actor=actor)
    return TrajectorySearchResponse(results=results, query=search_request.query)


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
        query = select(Trajectory).where(Trajectory.organization_id == actor.organization_id)

        if agent_id:
            query = query.where(Trajectory.agent_id == agent_id)
        if min_score is not None:
            query = query.where(Trajectory.outcome_score >= min_score)

        # Only return processed trajectories with embeddings
        query = query.where(Trajectory.processing_status == "completed", Trajectory.embedding.isnot(None)).limit(limit)

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
        query = select(Trajectory).where(Trajectory.organization_id == actor.organization_id)
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
            daily_counts.append(
                {
                    "date": date_str,
                    "count": data["count"],
                    "avg_score": avg_score,
                }
            )

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
