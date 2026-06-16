"""FastAPI app: serves the dashboard and drives the orchestrator over WebSocket.

All activity is persisted to SQLite (agents.db) so sessions survive restarts and
previous history can be replayed in the dashboard.

Run with:  python3 -m uvicorn backend.main:app --reload
Requires:  ANTHROPIC_API_KEY in the environment (or a .env file).
"""

from __future__ import annotations

from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from . import db
from .agents import AGENTS
from .orchestrator import Orchestrator

load_dotenv()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Multi-Agent Agentic OS")

_client = anthropic.AsyncAnthropic()
_orchestrator = Orchestrator(_client)

# Pricing / limits for the token-usage panel. claude-opus-4-8: $5 / $25 per 1M.
PRICING = {
    "input_per_mtok_usd": 5.0,
    "output_per_mtok_usd": 25.0,
    "usd_to_thb": 36.5,          # approximate; adjust to taste
    "daily_token_limit": 2_000_000,  # warn as this is approached
    "warn_ratio": 0.8,
}


def usage_payload() -> dict:
    return {
        "type": "usage",
        "today": db.usage_today(),
        "series": db.usage_series(7),
        "pricing": PRICING,
    }


@app.on_event("startup")
async def _startup() -> None:
    db.init()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/agents")
async def list_agents() -> dict:
    return {
        "agents": [
            {"id": a.id, "name": a.name, "title": a.title, "emoji": a.emoji, "tags": list(a.tags)}
            for a in AGENTS.values()
        ]
    }


@app.get("/api/usage")
async def usage() -> dict:
    return usage_payload()


class Recorder:
    """Wraps the WebSocket ``emit`` and persists each event to SQLite.

    Accumulates streamed chunks per agent and flushes the full text to the DB
    when the agent finishes, so we store coherent messages rather than fragments.
    """

    def __init__(self, send, session_id: int, goal: str):
        self._send = send
        self.session_id = session_id
        self.goal = goal
        self._buffers: dict[str, list[str]] = {}
        self._task_ids: dict[str, int] = {}

    async def __call__(self, event: dict) -> None:
        self._persist(event)
        await self._send(event)

    def _persist(self, e: dict) -> None:
        t = e.get("type")
        if t == "plan":
            for st in e["subtasks"]:
                self._task_ids[st["agent"]] = db.create_task(
                    self.session_id, st["agent"], st["task"]
                )
        elif t == "agent_output":
            self._buffers.setdefault(e["agent"], []).append(e["chunk"])
        elif t == "agent_status":
            db.save_status(e["agent"], e["status"])
            agent = e["agent"]
            if e["status"] == "working":
                self._buffers[agent] = []
                if agent in self._task_ids:
                    db.update_task(self._task_ids[agent], status="working")
            elif e["status"] == "done":
                text = "".join(self._buffers.get(agent, []))
                # The CEO's synthesis is stored via the dedicated "final" event,
                # so skip its buffered copy to avoid duplicates.
                if text and agent != "ceo":
                    db.save_message(self.session_id, agent, "assistant", text)
                if agent in self._task_ids:
                    db.update_task(self._task_ids[agent], status="done", result=text)
        elif t == "token_usage":
            db.save_token_usage(
                self.session_id, e["agent"], e["input_tokens"], e["output_tokens"]
            )
        elif t == "final":
            db.save_message(self.session_id, "ceo", "final", e["output"])


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()

    async def send(event: dict) -> None:
        await websocket.send_json(event)

    # Continue the most recent session (requirement: pick up where we left off).
    current = db.get_or_create_current()
    current_id = current["id"]

    async def send_session_state(session: dict) -> None:
        await send({"type": "sessions", "sessions": db.list_sessions()})
        await send({"type": "session", "id": session["id"], "name": session["name"]})
        await send({"type": "history", **db.get_history(session["id"])})

    await send_session_state(current)
    await send(usage_payload())

    try:
        while True:
            msg = await websocket.receive_json()
            action = (msg or {}).get("action")

            if action == "new_session":
                current = db.create_session()
                current_id = current["id"]
                await send_session_state(current)
                continue

            if action == "load_session":
                sid = msg.get("id")
                if sid:
                    current_id = sid
                    name = next(
                        (s["name"] for s in db.list_sessions() if s["id"] == sid), ""
                    )
                    await send_session_state({"id": sid, "name": name})
                continue

            goal = (msg or {}).get("goal", "").strip()
            if not goal:
                await send({"type": "error", "message": "Empty goal."})
                continue

            db.save_message(current_id, "user", "user", goal)
            recorder = Recorder(send, current_id, goal)
            try:
                await _orchestrator.run(goal, recorder)
            except Exception as exc:  # surface to the UI rather than dropping the socket
                await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                await send({"type": "done"})
            # Refresh the session list and usage totals after a run.
            await send({"type": "sessions", "sessions": db.list_sessions()})
            await send(usage_payload())
    except WebSocketDisconnect:
        return
