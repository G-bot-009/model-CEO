"""FastAPI app: serves the dashboard and drives the orchestrator over WebSocket.

All activity is persisted to SQLite (agents.db) so sessions survive restarts and
previous history can be replayed in the dashboard.

Run with:  python3 -m uvicorn backend.main:app --reload
Requires:  ANTHROPIC_API_KEY in the environment (or a .env file).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from . import db
from . import skills
from .agents import AGENTS, JARVIS, all_agents, compose_custom_system, SPECIALIST_FOOTER as _SPECIALIST_FOOTER
from .orchestrator import Orchestrator, get_model, set_model, thinking_kwargs

JARVIS_ROUTINE = (
    "รันรูทีนเช้าของฉันให้ครบ 5 ข้อ แล้วสรุปแบบผู้ช่วยส่วนตัว:\n"
    "1) ดึงรายได้แอปจาก RevenueCat แล้วสรุปผลงานเมื่อคืน\n"
    "2) เช็ค Meta Ads แล้วสรุป: งบที่ใช้, ROAS, และโฆษณาที่ดีที่สุด\n"
    "3) อ่านอีเมลลูกค้าที่ยังไม่ได้อ่าน แล้วร่างคำตอบให้ฉันรีวิว\n"
    "4) เสนอไอเดียคอนเทนต์สั้น 3 ชิ้นจากเทรนด์ปัจจุบัน\n"
    "5) เตรียมตั้งเวลาโพสต์คอนเทนต์ที่อนุมัติแล้วไป Buffer ทุกแพลตฟอร์ม (รอฉันยืนยันก่อนโพสต์)"
)

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


def _cache_on() -> bool:
    return db.get_settings().get("cache_enabled", "true") != "false"

def _cache_key(agent: str, prompt: str) -> str:
    return hashlib.sha256(f"{get_model()}|{agent}|{prompt}".encode("utf-8")).hexdigest()


@app.on_event("startup")
async def _startup() -> None:
    db.init()
    set_model(db.get_settings().get("model"))  # apply saved model choice


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
                "color": a.color,
                "custom": a.id.startswith("x_"),
                "category": a.category,
                "skills": list(a.skills),
            }
            for a in all_agents().values()
        ]
    }


@app.get("/api/skills")
async def list_skills() -> dict:
    """Skill catalog (categories + skills) for the agent-creation picker."""
    return {"categories": skills.catalog()}


@app.post("/api/agent/create")
async def create_agent(p: dict) -> dict:
    name = (p.get("name") or "").strip()
    if not name:
        return {"error": "name required"}
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:24] or "agent"
    existing = all_agents()
    aid = base = "x_" + slug
    i = 1
    while aid in existing:
        aid = f"{base}_{i}"; i += 1
    role = (p.get("title") or "ผู้ช่วยทั่วไป").strip()
    emoji = (p.get("emoji") or "🧩").strip()[:4]
    color = (p.get("color") or "#64748b").strip()
    category = (p.get("category") or "").strip()
    chosen = p.get("skills") or []
    if isinstance(chosen, str):
        chosen = [s for s in chosen.split(",") if s]
    chosen = skills.valid_skill_ids(chosen)[:12]   # cap to protect token budget
    # auto-fill skill chips on the card from the first few chosen skills
    if chosen:
        chip_titles = [skills.skill_meta(c)["th"] for c in chosen[:3]]
        tags = ",".join(chip_titles)
    else:
        tags = ",".join([t.strip() for t in (p.get("tags") or "").split(",") if t.strip()][:3])
    system = skills.compose_system(name, role, _SPECIALIST_FOOTER, chosen)
    db.add_custom_agent(aid, name, role, emoji, color, tags, system, category, ",".join(chosen))
    db.add_decision("agent", f"created {name} ({len(chosen)} skills)")
    return {"ok": True, "id": aid, "skills": len(chosen)}


@app.post("/api/agent/delete")
async def delete_agent(p: dict) -> dict:
    aid = p.get("id", "")
    if not aid.startswith("x_"):
        return {"error": "ลบได้เฉพาะเอเจนต์ที่สร้างเอง"}
    db.delete_custom_agent(aid)
    db.add_decision("agent", f"deleted {aid}")
    return {"ok": True}


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
            "agents": len(all_agents()),
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
    _ag = all_agents()
    if agent_id not in _ag:
        return {"error": "unknown agent"}
    a = _ag[agent_id]
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
    _ag = all_agents()
    agent = p.get("agent") if p.get("agent") in _ag else "designer"
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
            model=get_model(),
            max_tokens=8000,
            **thinking_kwargs(),
            system=_ag[agent].system,
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
    if p.get("model"):
        set_model(p["model"])
    return {"ok": True, "settings": db.get_settings()}


@app.get("/api/cache")
async def cache_info() -> dict:
    return {"count": db.cache_count(), "enabled": _cache_on()}


@app.post("/api/cache/clear")
async def cache_clear() -> dict:
    db.cache_clear()
    return {"ok": True, "count": 0}


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


async def run_jarvis(send, goal: str, client, session_id: int) -> None:
    """Stream a Jarvis (personal-assistant persona) response."""
    await send({"type": "jarvis_status", "status": "working"})
    collected: list[str] = []
    async with client.messages.stream(
        model=get_model(),
        max_tokens=8000,
        **thinking_kwargs(),
        system=JARVIS.system,
        messages=[{"role": "user", "content": goal}],
    ) as stream:
        async for chunk in stream.text_stream:
            collected.append(chunk)
            await send({"type": "jarvis_output", "chunk": chunk})
        final = await stream.get_final_message()
    text = "".join(collected)
    db.save_message(session_id, "jarvis", "assistant", text)
    u = final.usage
    db.save_token_usage(session_id, "jarvis", getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0)
    db.add_decision("jarvis", goal[:80])
    await send({"type": "jarvis_done"})
    return text


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

            # BYOK: use the user's own API key, or point at a local/custom
            # Anthropic-compatible endpoint (e.g. Ollama via a LiteLLM proxy = $0).
            if action == "set_key":
                key = (msg.get("api_key") or "").strip()
                provider = msg.get("provider", "anthropic")
                base_url = (msg.get("base_url") or "").strip()
                model = (msg.get("model") or "").strip()
                if base_url:
                    orch = Orchestrator(anthropic.AsyncAnthropic(api_key=key or "local", base_url=base_url))
                    if model:
                        set_model(model)
                    await send({"type": "byok", "ok": True, "provider": "local (" + base_url + ")"})
                elif not key:
                    orch = _orchestrator
                    await send({"type": "byok", "ok": True, "provider": "default"})
                elif provider == "anthropic":
                    orch = Orchestrator(anthropic.AsyncAnthropic(api_key=key))
                    if model:
                        set_model(model)
                    await send({"type": "byok", "ok": True, "provider": provider})
                else:
                    await send({"type": "byok", "ok": False,
                                "message": "provider นี้ต้องใช้โหมด Local/Base URL (Anthropic-compatible) — ใส่ Base URL"})
                continue

            # Direct task to a single agent (skip CEO planning/delegation).
            if action == "agent_task":
                agent = msg.get("agent")
                task = (msg.get("task") or "").strip()
                if agent not in all_agents() or not task:
                    await send({"type": "error", "message": "Invalid agent task."})
                    continue
                ps = db.pause_state()
                if ps["company"] or ps["agents"].get(agent):
                    await send({"type": "error", "message": f"⏸️ {all_agents()[agent].name} ถูกพักงานอยู่ — กดทำงานต่อก่อนสั่งงาน"})
                    await send({"type": "done"})
                    continue
                db.save_message(current_id, "user", "user", f"[{agent}] {task}")
                await send({"type": "plan", "subtasks": [{"agent": agent, "task": task}]})
                tkey = _cache_key(agent, task)
                tcached = db.cache_get(tkey) if _cache_on() else None
                if tcached:
                    await send({"type": "agent_status", "agent": agent, "status": "working"})
                    await send({"type": "agent_output", "agent": agent, "chunk": "💾 (จากแคช · 0 token)\n\n" + tcached})
                    await send({"type": "agent_status", "agent": agent, "status": "done"})
                    db.save_message(current_id, agent, "assistant", tcached)
                    tid = db.create_task(current_id, agent, task); db.update_task(tid, status="done", result=tcached)
                    await send({"type": "done"})
                    continue
                rec = Recorder(send, current_id, task)
                rec._task_ids[agent] = db.create_task(current_id, agent, task)
                try:
                    result = await orch.run_specialist(agent, task, task, rec)
                    if result:
                        db.cache_set(tkey, agent, task, result)
                    db.add_decision("agent_task", f"{agent}: {task[:80]}")
                except Exception as exc:
                    await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                await send({"type": "done"})
                await send(usage_payload())
                continue

            # Jarvis mode (personal assistant) — chat or full morning routine.
            if action in ("jarvis", "jarvis_routine"):
                jgoal = JARVIS_ROUTINE if action == "jarvis_routine" else (msg.get("goal") or "").strip()
                if not jgoal:
                    await send({"type": "jarvis_done"})
                    continue
                db.save_message(current_id, "user", "user", "[jarvis] " + jgoal[:200])
                ckey = _cache_key("jarvis", jgoal)
                cached = db.cache_get(ckey) if _cache_on() else None
                if cached:
                    await send({"type": "jarvis_status", "status": "working"})
                    await send({"type": "jarvis_output", "chunk": "💾 (จากแคช · 0 token)\n\n" + cached})
                    await send({"type": "jarvis_done"})
                    db.save_message(current_id, "jarvis", "assistant", cached)
                    continue
                try:
                    out = await run_jarvis(send, jgoal, orch.client, current_id)
                    if out:
                        db.cache_set(ckey, "jarvis", jgoal, out)
                except Exception as exc:
                    await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                    await send({"type": "jarvis_done"})
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
