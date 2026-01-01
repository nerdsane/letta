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
