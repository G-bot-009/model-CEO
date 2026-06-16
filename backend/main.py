"""FastAPI app: serves the dashboard and drives the orchestrator over WebSocket.

All activity is persisted to SQLite (agents.db) so sessions survive restarts and
previous history can be replayed in the dashboard.

Run with:  python3 -m uvicorn backend.main:app --reload
Requires:  ANTHROPIC_API_KEY in the environment (or a .env file).
"""

from __future__ import annotations

import re
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
    s = db.get_settings()  # user-defined name overrides (agent_name_<id>)
    return {
        "agents": [
            {
                "id": a.id,
                "name": s.get(f"agent_name_{a.id}") or a.name,
                "title": a.title,
                "emoji": a.emoji,
                "tags": list(a.tags),
            }
            for a in AGENTS.values()
        ]
    }


@app.get("/api/usage")
async def usage() -> dict:
    return usage_payload()


@app.get("/api/console")
async def console() -> dict:
    connectors = db.list_connectors()
    connected = sum(1 for v in connectors.values() if v == "connected")
    summary = db.approvals_summary()
    return {
        "sops": db.list_sops(),
        "routines": db.list_routines(),
        "approvals": db.list_approvals("pending"),
        "decisions": db.list_decisions(),
        "overview": {"total": db.decisions_count(), **summary},
        "workforce": db.workforce(),
        "connectors": connectors,
        "stats": {
            "pending": summary["pending"],
            "decisions_today": db.decisions_today(),
            "agents": len(AGENTS),
            "systems": connected,
            "bot": "OFF",
        },
        "status": {
            "database": True,
            "metaapi": connectors.get("MetaTrader 5 (Forex)") == "connected",
            "trading_bot": False,
            "egress": connected > 0,
        },
    }


@app.post("/api/sop")
async def api_add_sop(p: dict) -> dict:
    db.add_sop((p.get("title") or "SOP").strip(), (p.get("body") or "").strip())
    return {"ok": True, "sops": db.list_sops()}


@app.post("/api/routine")
async def api_add_routine(p: dict) -> dict:
    db.add_routine((p.get("goal") or "").strip(), p.get("cadence") or "daily")
    return {"ok": True, "routines": db.list_routines()}


@app.post("/api/approval")
async def api_decide(p: dict) -> dict:
    db.decide_approval(int(p["id"]), p.get("action", "approved"))
    db.add_decision("approval", f'{p.get("action","approved")} #{p["id"]}')
    return {"ok": True, "approvals": db.list_approvals("pending")}


@app.post("/api/connector")
async def api_connector(p: dict) -> dict:
    name = (p.get("name") or "").strip()
    status = p.get("status", "connected")
    db.set_connector(name, status, p.get("config", ""))
    db.add_decision("connector", f"{name} → {status}")
    return {"ok": True, "connectors": db.list_connectors()}


@app.get("/api/agent/{agent_id}")
async def agent_detail(agent_id: str, period: str = "day") -> dict:
    if agent_id not in AGENTS:
        return {"error": "unknown agent"}
    a = AGENTS[agent_id]
    s = db.get_settings()
    return {
        "id": agent_id,
        "name": s.get(f"agent_name_{agent_id}") or a.name,
        "title": a.title,
        "emoji": a.emoji,
        "period": period,
        "summary": db.agent_summary(agent_id, period),
        "works": db.agent_works(agent_id),
        "totals": db.agent_totals(agent_id),
        "images": db.list_images(agent_id, 12),
    }


# --- Image studio: Claude generates self-contained SVG artwork --------------
IMG_SIZES = {"1:1": (600, 600), "16:9": (800, 450), "9:16": (450, 800), "banner": (900, 300)}

def _extract_svg(text: str) -> str:
    i, j = text.find("<svg"), text.rfind("</svg>")
    svg = text[i:j + 6] if (i != -1 and j != -1) else ""
    # basic sanitization for embedding in the page
    svg = re.sub(r"<script[\s\S]*?</script>", "", svg, flags=re.I)
    svg = re.sub(r"<foreignObject[\s\S]*?</foreignObject>", "", svg, flags=re.I)
    svg = re.sub(r"\son\w+\s*=\s*\"[^\"]*\"", "", svg, flags=re.I)
    svg = re.sub(r"\son\w+\s*=\s*'[^']*'", "", svg, flags=re.I)
    return svg


@app.post("/api/generate-image")
async def generate_image(p: dict) -> dict:
    agent = p.get("agent") if p.get("agent") in AGENTS else "designer"
    prompt = (p.get("prompt") or "").strip()
    size = p.get("size", "1:1")
    if not prompt:
        return {"error": "empty prompt"}
    w, h = IMG_SIZES.get(size, (600, 600))
    instr = (
        f"Create one piece of artwork as a SINGLE self-contained SVG. "
        f"Output ONLY raw SVG markup — no markdown, no code fences, no explanation. "
        f'Start with <svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}"> and end with </svg>. Use gradients, clean shapes, and '
        f"readable text where helpful. Do not use external images, fonts, <script>, or "
        f"<foreignObject>.\n\nSubject: {prompt}"
    )
    try:
        resp = await _client.messages.create(
            model="claude-opus-4-8",
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=AGENTS[agent].system,
            messages=[{"role": "user", "content": instr}],
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    svg = _extract_svg(text)
    if not svg:
        return {"error": "โมเดลไม่ได้คืนค่าเป็น SVG — ลองใหม่อีกครั้ง"}
    sess = db.get_or_create_current()["id"]
    iid = db.add_image(agent, prompt, size, svg)
    u = resp.usage
    db.save_token_usage(sess, agent, getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0)
    tid = db.create_task(sess, agent, f"สร้างภาพ: {prompt[:60]}")
    db.update_task(tid, status="done", result="(ภาพ SVG)")
    db.add_decision("image", f"{agent}: {prompt[:60]}")
    return {"ok": True, "id": iid, "svg": svg, "prompt": prompt, "size": size}


@app.get("/api/pause")
async def get_pause() -> dict:
    return db.pause_state()


@app.post("/api/pause")
async def set_pause(p: dict) -> dict:
    target = p.get("target")
    paused = bool(p.get("paused"))
    db.set_pause(target, paused)
    db.add_decision("pause", f"{target} → {'พัก' if paused else 'ทำงานต่อ'}")
    return {"ok": True, **db.pause_state()}


@app.get("/api/settings")
async def get_settings() -> dict:
    return db.get_settings()


@app.post("/api/settings")
async def post_settings(p: dict) -> dict:
    db.set_settings(p)
    return {"ok": True, "settings": db.get_settings()}


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
    orch = _orchestrator   # may be swapped to a BYOK client for this connection

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

            # BYOK: use the user's own API key for this connection (Anthropic only for now).
            if action == "set_key":
                key = (msg.get("api_key") or "").strip()
                provider = msg.get("provider", "anthropic")
                if not key:
                    orch = _orchestrator
                    await send({"type": "byok", "ok": True, "provider": "default"})
                elif provider == "anthropic":
                    orch = Orchestrator(anthropic.AsyncAnthropic(api_key=key))
                    await send({"type": "byok", "ok": True, "provider": provider})
                else:
                    await send({"type": "byok", "ok": False,
                                "message": "ตอนนี้รองรับเฉพาะ Anthropic (Claude) — provider อื่นกำลังจะเพิ่ม"})
                continue

            # Direct task to a single agent (skip CEO planning/delegation).
            if action == "agent_task":
                agent = msg.get("agent")
                task = (msg.get("task") or "").strip()
                if agent not in AGENTS or not task:
                    await send({"type": "error", "message": "Invalid agent task."})
                    continue
                ps = db.pause_state()
                if ps["company"] or ps["agents"].get(agent):
                    await send({"type": "error", "message": f"⏸️ {AGENTS[agent].name} ถูกพักงานอยู่ — กดทำงานต่อก่อนสั่งงาน"})
                    await send({"type": "done"})
                    continue
                db.save_message(current_id, "user", "user", f"[{agent}] {task}")
                rec = Recorder(send, current_id, task)
                rec._task_ids[agent] = db.create_task(current_id, agent, task)
                try:
                    await send({"type": "plan", "subtasks": [{"agent": agent, "task": task}]})
                    await orch.run_specialist(agent, task, task, rec)
                    db.add_decision("agent_task", f"{agent}: {task[:80]}")
                except Exception as exc:
                    await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                await send({"type": "done"})
                await send(usage_payload())
                continue

            goal = (msg or {}).get("goal", "").strip()
            if not goal:
                await send({"type": "error", "message": "Empty goal."})
                continue

            ps = db.pause_state()
            if ps["company"]:
                await send({"type": "error", "message": "⏸️ บริษัทหยุดชั่วคราวอยู่ — กดเริ่มบริษัทก่อนสั่งงาน"})
                await send({"type": "done"})
                continue
            paused_set = {a for a, v in ps["agents"].items() if v}

            db.save_message(current_id, "user", "user", goal)
            recorder = Recorder(send, current_id, goal)
            try:
                await orch.run(goal, recorder, paused=paused_set)
                db.add_decision("run", goal[:120])
            except Exception as exc:  # surface to the UI rather than dropping the socket
                await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                await send({"type": "done"})
            # Refresh the session list and usage totals after a run.
            await send({"type": "sessions", "sessions": db.list_sessions()})
            await send(usage_payload())
    except WebSocketDisconnect:
        return
