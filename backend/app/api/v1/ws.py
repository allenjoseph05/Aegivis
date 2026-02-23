"""
WebSocket /ws/alerts - Real-time push of violations and anomaly detections.

Clients connect and receive JSON messages as they arrive.
The ws_broadcast() helper is imported by ingest.py to push events.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

# In-process subscriber set. All connected dashboard clients live here.
_subscribers: set[WebSocket] = set()


async def ws_broadcast(message: dict) -> None:
    """
    Fire-and-forget push to all connected WebSocket clients.

    Dead connections are silently pruned. This function never raises.
    Call via asyncio.create_task() from ingest to avoid blocking.
    """
    if not _subscribers:
        return
    data = json.dumps(message, default=str)
    dead: set[WebSocket] = set()
    for ws in list(_subscribers):
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    if dead:
        _subscribers.difference_update(dead)
        logger.debug(f"Pruned {len(dead)} dead WebSocket subscriber(s)")


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time violation and anomaly alerts.

    Connect from the dashboard to receive live push notifications.
    Clients may send any text (ping frames) which are ignored.
    """
    await websocket.accept()
    _subscribers.add(websocket)
    logger.info(f"WebSocket client connected — {len(_subscribers)} subscriber(s)")
    try:
        while True:
            # Keep connection alive; handle client-side pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _subscribers.discard(websocket)
        logger.info(f"WebSocket client disconnected — {len(_subscribers)} subscriber(s)")
