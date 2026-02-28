"""
GET /v1/topology — Agent topology graph.

Returns a directed graph of agent-to-agent relationships, aggregated
performance statistics, and risk scores for every observed agent.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key
from ...services.topology import compute_topology

router = APIRouter()


@router.get(
    "/topology",
    summary="Agent topology graph",
    response_description=(
        "Directed graph with nodes (agents) and edges (inter-agent calls). "
        "Each node carries aggregated performance and security metrics."
    ),
)
async def get_topology(
    include_isolated: bool = Query(
        True,
        description="Include agents with no cross-agent edges (stand-alone agents).",
    ),
    min_edge_calls: int = Query(
        1,
        ge=1,
        description="Minimum cross-agent call count to include an edge.",
    ),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Compute and return the current agent topology graph, scoped to the
    authenticated organisation.
    """
    graph = await compute_topology(
        db,
        org_id=org_ctx.org_id,
        include_isolated=include_isolated,
        min_edge_calls=min_edge_calls,
    )
    return graph.to_dict()
