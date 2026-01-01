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
from letta.services.trajectory_service import TrajectoryService

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
    # Get user_id from headers.actor_id
    user_id = headers.actor_id
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    # Use server's database registry
    from letta.server.db_registry import db_registry
    async with db_registry.async_session() as db:
        service = TrajectoryService(db, user_id)
        trajectory = await service.create_trajectory(trajectory_create)
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
    service = TrajectoryService(db, user_id)
    trajectory = await service.get_trajectory(trajectory_id)

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
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """
    List trajectories with optional filtering.
    """
    service = TrajectoryService(db, user_id)
    trajectories = await service.list_trajectories(
        agent_id=agent_id, min_score=min_score, max_score=max_score, limit=limit, offset=offset
    )
    return trajectories


@router.patch("/{trajectory_id}", response_model=Trajectory)
async def update_trajectory(
    trajectory_id: str,
    trajectory_update: TrajectoryUpdate = Body(...),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """
    Update an existing trajectory.

    Typically used to:
    - Add more data as execution progresses
    - Update outcome information when complete
    - Trigger reprocessing (set searchable_summary=None to regenerate)
    """
    service = TrajectoryService(db, user_id)
    trajectory = await service.update_trajectory(trajectory_id, trajectory_update)

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    return trajectory


@router.delete("/{trajectory_id}", status_code=204)
async def delete_trajectory(
    trajectory_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """
    Delete a trajectory.
    """
    service = TrajectoryService(db, user_id)
    success = await service.delete_trajectory(trajectory_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")


@router.post("/search", response_model=TrajectorySearchResponse)
async def search_trajectories(
    search_request: TrajectorySearchRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
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
    service = TrajectoryService(db, user_id)
    results = await service.search_trajectories(search_request)
    return TrajectorySearchResponse(results=results, query=search_request.query)


@router.post("/{trajectory_id}/process", response_model=Trajectory)
async def process_trajectory(
    trajectory_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
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
    service = TrajectoryService(db, user_id)
    trajectory = await service.process_trajectory(trajectory_id)

    if not trajectory:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    return trajectory
