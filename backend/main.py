"""FastAPI app: serves the dashboard and drives the orchestrator over WebSocket.

Run with:  uvicorn backend.main:app --reload
Requires:  ANTHROPIC_API_KEY in the environment (or a .env file).
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .agents import AGENTS
from .orchestrator import Orchestrator

load_dotenv()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Multi-Agent Agentic OS")

# A single shared async client (the SDK manages its own connection pool).
_client = anthropic.AsyncAnthropic()
_orchestrator = Orchestrator(_client)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/agents")
async def list_agents() -> dict:
    """Roster for the dashboard to render agent cards."""
    return {
        "agents": [
            {"id": a.id, "name": a.name, "title": a.title, "emoji": a.emoji}
            for a in AGENTS.values()
        ]
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()

    async def emit(event: dict) -> None:
        await websocket.send_json(event)

    try:
        while True:
            msg = await websocket.receive_json()
            goal = (msg or {}).get("goal", "").strip()
            if not goal:
                await emit({"type": "error", "message": "Empty goal."})
                continue
            try:
                await _orchestrator.run(goal, emit)
            except Exception as exc:  # surface failures to the UI rather than dropping the socket
                await emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                await emit({"type": "done"})
    except WebSocketDisconnect:
        return
