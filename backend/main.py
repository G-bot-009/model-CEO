"""FastAPI app: serves the dashboard and drives the orchestrator over WebSocket.

All activity is persisted to SQLite (agents.db) so sessions survive restarts and
previous history can be replayed in the dashboard.

Run with:  python3 -m uvicorn backend.main:app --reload
Requires:  ANTHROPIC_API_KEY in the environment (or a .env file).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

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

_client = anthropic.AsyncAnthropic()          # env default — never lost ("API ห้ามหาย")
_client_cache: dict[str, anthropic.AsyncAnthropic] = {}


def _build_client(base_url: str, secret: str) -> anthropic.AsyncAnthropic:
    """Cache/build an Anthropic client for a stored key (Anthropic key or
    Anthropic-compatible base URL). Falls back to the env default."""
    base, secret = (base_url or "").strip(), (secret or "").strip()
    ck = f"{base}\n{secret}"
    c = _client_cache.get(ck)
    if c is None:
        if base:
            c = anthropic.AsyncAnthropic(api_key=secret or "local", base_url=base)
        elif secret:
            c = anthropic.AsyncAnthropic(api_key=secret)
        else:
            c = _client
        _client_cache[ck] = c
    return c


def client_for_agent(agent_id: str, default: Optional[anthropic.AsyncAnthropic] = None) -> anthropic.AsyncAnthropic:
    """Resolve the client an agent should use: its assigned key, else ``default``/env."""
    default = default or _client
    try:
        kid = db.agent_key_id(agent_id)
        if kid:
            row = db.get_api_key(kid)
            if row:
                return _build_client(row.get("base_url"), row.get("secret"))
    except Exception:
        pass
    return default


_orchestrator = Orchestrator(lambda aid: client_for_agent(aid))

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


def _spend_usd(inp: int, out: int) -> float:
    return inp / 1e6 * PRICING["input_per_mtok_usd"] + out / 1e6 * PRICING["output_per_mtok_usd"]


def _resolve_range(period: str, start, end):
    """Return (start_day, end_day, label, period) for a named period or a custom from/to."""
    today = date.today()
    if start and end:
        return start, end, f"{start} → {end}", "custom"
    if period == "yesterday":
        d = (today - timedelta(days=1)).isoformat()
        return d, d, "เมื่อวาน", period
    if period == "7d":
        return (today - timedelta(days=6)).isoformat(), today.isoformat(), "7 วันล่าสุด", period
    if period in ("30d", "month", "1month"):
        return (today - timedelta(days=29)).isoformat(), today.isoformat(), "1 เดือน (30 วัน)", "30d"
    return today.isoformat(), today.isoformat(), "วันนี้", "today"


def _cache_on() -> bool:
    return db.get_settings().get("cache_enabled", "true") != "false"

def _cache_key(agent: str, prompt: str) -> str:
    return hashlib.sha256(f"{get_model()}|{agent}|{prompt}".encode("utf-8")).hexdigest()


@app.on_event("startup")
async def _startup() -> None:
    db.init()
    set_model(db.get_settings().get("model"))  # apply saved model choice


LOGIN_USER = os.getenv("LOGIN_USER", "admin")
LOGIN_PASS = os.getenv("LOGIN_PASS", "admin")


def _auth_token() -> str:
    """Stable per-install token; the login cookie must match it."""
    s = db.get_settings()
    tok = s.get("auth_token")
    if not tok:
        tok = secrets.token_hex(16)
        db.set_settings({"auth_token": tok})
    return tok


@app.get("/")
async def index(request: Request):
    # gate the dashboard behind login (local single-user)
    if request.cookies.get("mf_auth") != _auth_token():
        return RedirectResponse("/login")
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/login")
async def login_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "login.html")


@app.post("/api/login")
async def api_login(p: dict):
    if (p.get("username") or "").strip() == LOGIN_USER and (p.get("password") or "") == LOGIN_PASS:
        resp = JSONResponse({"ok": True})
        resp.set_cookie("mf_auth", _auth_token(), httponly=True, samesite="lax",
                        max_age=60 * 60 * 24 * 30, path="/")
        return resp
    return JSONResponse({"error": "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"}, status_code=401)


@app.post("/api/logout")
async def api_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("mf_auth", path="/")
    return resp


# --- Inbound webhook: external services (TradingView/Zapier/n8n) trigger the team ---
def _inbound_path() -> Optional[str]:
    tok = db.get_settings().get("inbound_token")
    return f"/api/inbound/{tok}" if tok else None


@app.get("/api/inbound")
async def inbound_info() -> dict:
    p = _inbound_path()
    return {"enabled": bool(p), "path": p}


@app.post("/api/inbound/enable")
async def inbound_enable() -> dict:
    tok = db.get_settings().get("inbound_token")
    if not tok:
        tok = secrets.token_urlsafe(12)
        db.set_settings({"inbound_token": tok})
    return {"ok": True, "path": f"/api/inbound/{tok}"}


# --- Social OAuth (BYO app per platform) ------------------------------------
# Each platform: provider auth/token URLs + scope. The user registers their own
# OAuth app (client_id/secret) and sets the redirect URI we show them.
SOCIAL = {
    "instagram": {"label": "Instagram", "auth": "https://www.facebook.com/v19.0/dialog/oauth", "token": "https://graph.facebook.com/v19.0/oauth/access_token", "scope": "instagram_basic,instagram_content_publish,pages_show_list"},
    "facebook":  {"label": "Facebook", "auth": "https://www.facebook.com/v19.0/dialog/oauth", "token": "https://graph.facebook.com/v19.0/oauth/access_token", "scope": "public_profile,pages_show_list,pages_manage_posts"},
    "tiktok":    {"label": "TikTok", "auth": "https://www.tiktok.com/v2/auth/authorize/", "token": "https://open.tiktokapis.com/v2/oauth/token/", "scope": "user.info.basic,video.publish", "client_param": "client_key"},
    "youtube":   {"label": "YouTube", "auth": "https://accounts.google.com/o/oauth2/v2/auth", "token": "https://oauth2.googleapis.com/token", "scope": "https://www.googleapis.com/auth/youtube.upload", "extra": {"access_type": "offline", "prompt": "consent"}},
    "x":         {"label": "X", "auth": "https://twitter.com/i/oauth2/authorize", "token": "https://api.twitter.com/2/oauth2/token", "scope": "tweet.read tweet.write users.read offline.access", "pkce": True, "basic": True},
    "linkedin":  {"label": "LinkedIn", "auth": "https://www.linkedin.com/oauth/v2/authorization", "token": "https://www.linkedin.com/oauth/v2/accessToken", "scope": "openid profile w_member_social"},
    "threads":   {"label": "Threads", "auth": "https://threads.net/oauth/authorize", "token": "https://graph.threads.net/oauth/access_token", "scope": "threads_basic,threads_content_publish"},
    "pinterest": {"label": "Pinterest", "auth": "https://www.pinterest.com/oauth/", "token": "https://api.pinterest.com/v5/oauth/token", "scope": "pins:write,boards:read", "basic": True},
    "google":    {"label": "Google", "auth": "https://accounts.google.com/o/oauth2/v2/auth", "token": "https://oauth2.googleapis.com/token", "scope": "https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/spreadsheets", "extra": {"access_type": "offline", "prompt": "consent"}},
}
_oauth_state: dict = {}   # state -> (platform, code_verifier)


def _redirect_uri(request: Request, platform: str) -> str:
    return str(request.base_url).rstrip("/") + f"/oauth/{platform}/callback"


@app.get("/api/social")
async def social_list(request: Request) -> dict:
    connected = db.list_social()
    return {
        "platforms": [{"id": k, "label": v["label"], "connected": connected.get(k, False)} for k, v in SOCIAL.items()],
        "redirect_note": str(request.base_url).rstrip("/") + "/oauth/<platform>/callback",
    }


@app.post("/api/social/creds")
async def social_creds(p: dict) -> dict:
    plat = p.get("platform")
    if plat not in SOCIAL:
        return {"error": "unknown platform"}
    db.set_social_creds(plat, (p.get("client_id") or "").strip(), (p.get("client_secret") or "").strip())
    return {"ok": True, "redirect_uri": _redirect_uri_str(plat)}


def _redirect_uri_str(plat: str) -> str:
    return f"/oauth/{plat}/callback"


@app.post("/api/social/disconnect")
async def social_disconnect(p: dict) -> dict:
    db.delete_social(p.get("platform", ""))
    return {"ok": True}


@app.get("/oauth/{platform}/start")
async def oauth_start(platform: str, request: Request):
    import base64 as _b64, hashlib as _hl
    from urllib.parse import urlencode
    cfg = SOCIAL.get(platform)
    creds = db.get_social(platform)
    if not cfg or not creds or not creds.get("client_id"):
        return HTMLResponse("<h3>ยังไม่ได้ใส่ client_id/secret ของแพลตฟอร์มนี้ — ปิดหน้าต่างแล้วกรอกก่อน</h3>", status_code=400)
    state = secrets.token_urlsafe(16)
    params = {
        "response_type": "code",
        cfg.get("client_param", "client_id"): creds["client_id"],
        "redirect_uri": _redirect_uri(request, platform),
        "scope": cfg["scope"],
        "state": state,
    }
    verifier = ""
    if cfg.get("pkce"):
        verifier = secrets.token_urlsafe(64)
        challenge = _b64.urlsafe_b64encode(_hl.sha256(verifier.encode()).digest()).decode().rstrip("=")
        params["code_challenge"] = challenge
        params["code_challenge_method"] = "S256"
    params.update(cfg.get("extra", {}))
    _oauth_state[state] = (platform, verifier)
    return RedirectResponse(cfg["auth"] + "?" + urlencode(params))


@app.get("/oauth/{platform}/callback")
async def oauth_callback(platform: str, request: Request, code: str = "", state: str = ""):
    import httpx
    cfg = SOCIAL.get(platform)
    creds = db.get_social(platform)
    saved = _oauth_state.pop(state, None)
    if not cfg or not creds or not code or not saved or saved[0] != platform:
        return HTMLResponse("<h3>เชื่อมไม่สำเร็จ (state/โค้ดไม่ถูกต้อง) — ปิดหน้าต่างแล้วลองใหม่</h3>", status_code=400)
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": _redirect_uri(request, platform),
        cfg.get("client_param", "client_id"): creds["client_id"],
        "client_secret": creds["client_secret"],
    }
    if saved[1]:
        body["code_verifier"] = saved[1]
    headers = {"Accept": "application/json"}
    auth = (creds["client_id"], creds["client_secret"]) if cfg.get("basic") else None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(cfg["token"], data=body, headers=headers, auth=auth)
        try:
            tok = r.json()
        except Exception:
            tok = {}
        access = tok.get("access_token", "")
        if not access:
            return HTMLResponse(f"<h3>แลก token ไม่สำเร็จ</h3><pre>{r.status_code}: {r.text[:400]}</pre>", status_code=400)
        db.set_social_token(platform, access, tok.get("refresh_token", ""), json.dumps(tok)[:2000])
        db.add_decision("social", f"connected {platform}")
    except Exception as exc:
        return HTMLResponse(f"<h3>เชื่อมไม่สำเร็จ</h3><pre>{type(exc).__name__}: {exc}</pre>", status_code=400)
    return HTMLResponse("<!doctype html><meta charset='utf-8'>"
                        "<body style='font-family:Inter,sans-serif;text-align:center;padding:48px'>"
                        f"<h2>เชื่อม {cfg['label']} สำเร็จ ✅</h2><p>ปิดหน้าต่างนี้ได้เลย</p>"
                        "<script>try{window.opener&&window.opener.postMessage('social-connected','*')}catch(e){};setTimeout(()=>window.close(),900)</script></body>")


@app.post("/api/inbound/{token}")
async def inbound_receive(token: str, request: Request) -> dict:
    if token != db.get_settings().get("inbound_token"):
        return JSONResponse({"error": "invalid token"}, status_code=404)
    # accept any JSON or raw body; pull a human-readable message
    try:
        payload = await request.json()
    except Exception:
        payload = {"text": (await request.body()).decode("utf-8", "ignore")}
    text = ""
    if isinstance(payload, dict):
        text = str(payload.get("text") or payload.get("message") or payload.get("content") or payload)
    else:
        text = str(payload)
    sess = db.get_or_create_current()["id"]
    db.save_message(sess, "inbound", "user", "[inbound] " + text[:1000])
    db.add_decision("inbound", text[:120])
    return {"ok": True}


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
async def usage(period: str = "today", start: Optional[str] = None, end: Optional[str] = None) -> dict:
    s, e, label, period = _resolve_range(period, start, end)
    rng = db.usage_between(s, e)
    today = date.today()
    month = db.usage_totals("substr(timestamp,1,7)=?", (today.isoformat()[:7],))
    allt = db.usage_totals()
    budget = 0.0
    try:
        budget = float(db.get_settings().get("token_budget_usd") or 0)
    except (TypeError, ValueError):
        budget = 0.0
    spent_all = _spend_usd(allt["input"], allt["output"])
    # per-key breakdown for the selected range
    labels = {k["id"]: k["label"] for k in db.list_api_keys()}
    by_key = []
    for row in db.usage_by_key(s, e):
        kid = row["key_id"]
        by_key.append({
            "key_id": kid,
            "label": labels.get(kid, "ค่าเริ่มต้น / ENV") if kid else "ค่าเริ่มต้น / ENV",
            "input": row["input"], "output": row["output"],
            "usd": round(_spend_usd(row["input"], row["output"]), 4),
        })
    by_key.sort(key=lambda x: -(x["input"] + x["output"]))
    return {
        "type": "usage",
        "today": {"per_agent": rng["per_agent"], "input": rng["input"], "output": rng["output"]},
        "series": rng["series"],
        "pricing": PRICING,
        "range": {"period": period, "start": s, "end": e, "label": label},
        "spend_usd": round(_spend_usd(rng["input"], rng["output"]), 4),
        "month_usd": round(_spend_usd(month["input"], month["output"]), 4),
        "alltime_usd": round(spent_all, 4),
        "budget_usd": budget,
        "remaining_usd": round(budget - spent_all, 4) if budget else None,
        "by_key": by_key,
    }


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


async def _send_to_connector(name: str, cfg_raw: str, text: str) -> dict:
    """Real outbound for connectors that work with a pasted credential (no OAuth)."""
    import base64
    import hashlib
    import hmac
    import time
    import asyncio
    import httpx

    try:
        data = json.loads(cfg_raw)
    except Exception:
        data = None
    if not isinstance(data, dict):
        data = {}
    cfg = (cfg_raw or "").strip()

    async with httpx.AsyncClient(timeout=20) as client:
        if name == "Slack":
            r = await client.post(cfg, json={"text": text})
        elif name == "Discord":
            r = await client.post(cfg, json={"content": text})
        elif name == "Telegram":
            tok, chat = data.get("bot_token", ""), data.get("chat_id", "")
            if not tok or not chat:
                return {"error": "ต้องมี Bot Token และ Chat ID — เชื่อมใหม่"}
            r = await client.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id": chat, "text": text})
        elif name == "Webhook → Make/Zapier":
            r = await client.post(cfg, json={"title": "ToonOffice", "text": text, "caption": text, "imageUrl": "", "mediaUrl": ""})
        elif name == "LINE OA":
            tok = data.get("access_token") or cfg
            r = await client.post("https://api.line.me/v2/bot/message/broadcast",
                                  headers={"Authorization": f"Bearer {tok}"},
                                  json={"messages": [{"type": "text", "text": text}]})
        elif name == "Notion":
            tok, parent = data.get("token", ""), (data.get("parent", "") or "").replace("-", "")
            if not tok or not parent:
                return {"error": "ต้องมี Integration token และ Parent Page ID"}
            r = await client.post("https://api.notion.com/v1/pages",
                                  headers={"Authorization": f"Bearer {tok}", "Notion-Version": "2022-06-28"},
                                  json={"parent": {"type": "page_id", "page_id": parent},
                                        "properties": {"title": {"title": [{"text": {"content": text[:200]}}]}}})
        elif name == "WordPress":
            url = (data.get("url", "") or "").rstrip("/")
            r = await client.post(f"{url}/wp-json/wp/v2/posts",
                                  auth=(data.get("user", ""), data.get("pass", "")),
                                  json={"title": "ToonOffice test", "content": text, "status": "draft"})
        elif name == "GitHub":
            tok, repo = data.get("token", ""), data.get("repo", "")
            if not tok or "/" not in repo:
                return {"error": "ต้องมี token และ repo แบบ owner/name"}
            path = f"toonoffice/test-{int(time.time())}.md"
            r = await client.put(f"https://api.github.com/repos/{repo}/contents/{path}",
                                 headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"},
                                 json={"message": "ToonOffice test", "content": base64.b64encode(text.encode()).decode()})
        elif name == "Email":
            import smtplib
            from email.message import EmailMessage
            def _smtp():
                msg = EmailMessage()
                msg["From"] = data.get("user", ""); msg["To"] = data.get("to", "")
                msg["Subject"] = "ToonOffice test"; msg.set_content(text)
                with smtplib.SMTP(data.get("host", ""), int(data.get("port") or 587), timeout=20) as s:
                    s.starttls(); s.login(data.get("user", ""), data.get("pass", "")); s.send_message(msg)
            await asyncio.to_thread(_smtp)
            return {"ok": True, "msg": "ส่งอีเมลแล้ว"}
        elif name == "Binance Futures":
            key, sec = data.get("api_key", ""), data.get("api_secret", "")
            q = f"recvWindow=5000&timestamp={int(time.time()*1000)}"
            sig = hmac.new(sec.encode(), q.encode(), hashlib.sha256).hexdigest()
            r = await client.get(f"https://api.binance.com/api/v3/account?{q}&signature={sig}",
                                 headers={"X-MBX-APIKEY": key})
            if r.status_code < 400:
                return {"ok": True, "msg": "อ่านบัญชี Binance ได้ ✅"}
        elif name == "Bybit":
            key, sec = data.get("api_key", ""), data.get("api_secret", "")
            ts, recv, qs = str(int(time.time()*1000)), "5000", "accountType=UNIFIED"
            sign = hmac.new(sec.encode(), (ts + key + recv + qs).encode(), hashlib.sha256).hexdigest()
            r = await client.get(f"https://api.bybit.com/v5/account/wallet-balance?{qs}",
                                 headers={"X-BAPI-API-KEY": key, "X-BAPI-TIMESTAMP": ts,
                                          "X-BAPI-RECV-WINDOW": recv, "X-BAPI-SIGN": sign})
            if r.status_code < 400 and (r.json().get("retCode") == 0):
                return {"ok": True, "msg": "อ่านวอลเล็ต Bybit ได้ ✅"}
            return {"error": f"Bybit: {r.text[:180]}"}
        else:
            return {"error": f"{name} ต้องใช้ OAuth/ตัวกลาง — ยังเชื่อมจริงไม่ได้ด้วยการวางคีย์ (เก็บไว้แล้ว)"}
        if r.status_code >= 400:
            return {"error": f"{name} ตอบกลับ {r.status_code}: {r.text[:180]}"}
    return {"ok": True}


@app.post("/api/connector/send")
async def connector_send(p: dict) -> dict:
    name = (p.get("name") or "").strip()
    text = (p.get("text") or "🔔 ทดสอบจาก ToonOffice — เชื่อมต่อสำเร็จ!").strip()
    row = db.get_connector(name)
    if not row or row.get("status") != "connected":
        return {"error": "ยังไม่ได้เชื่อม connector นี้"}
    try:
        res = await _send_to_connector(name, row.get("config") or "", text)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if res.get("ok"):
        db.add_decision("connector_send", f"{name}: {text[:60]}")
    return res


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
        resp = await client_for_agent(agent).messages.create(
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


@lru_cache(maxsize=1)
def _n8n_data() -> list:
    p = Path(__file__).resolve().parent / "n8n" / "automations.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


@app.get("/api/n8n")
async def n8n_list() -> dict:
    """n8n automation catalog (only entries that have a download link)."""
    return {"items": _n8n_data()}


@app.get("/api/cache")
async def cache_info() -> dict:
    return {"count": db.cache_count(), "enabled": _cache_on()}


@app.post("/api/cache/clear")
async def cache_clear() -> dict:
    db.cache_clear()
    return {"ok": True, "count": 0}


# --- API keys (Admin Usage: multi-key + per-agent assignment) ---------------
@app.get("/api/keys")
async def api_keys() -> dict:
    """Stored keys (masked), plus the agent→key assignment map. Secrets are
    never returned to the client."""
    return {"keys": db.list_api_keys(), "map": db.agent_key_map()}


@app.post("/api/keys")
async def add_api_key(p: dict) -> dict:
    label = (p.get("label") or "").strip()
    provider = (p.get("provider") or "anthropic").strip()
    base_url = (p.get("base_url") or "").strip()
    secret = (p.get("secret") or p.get("api_key") or "").strip()
    if not label:
        label = (provider or "key") + (" · " + base_url if base_url else "")
    if not secret and not base_url:
        return {"error": "ต้องใส่ API key หรือ Base URL อย่างน้อยหนึ่งอย่าง"}
    kid = db.add_api_key(label, provider, base_url, secret)
    db.add_decision("apikey", f"added {label}")
    return {"ok": True, "id": kid, "keys": db.list_api_keys()}


@app.post("/api/keys/delete")
async def del_api_key(p: dict) -> dict:
    kid = p.get("id")
    if not kid:
        return {"error": "id required"}
    db.delete_api_key(int(kid))
    db.add_decision("apikey", f"deleted #{kid}")
    return {"ok": True, "keys": db.list_api_keys(), "map": db.agent_key_map()}


@app.post("/api/keys/assign")
async def assign_api_key(p: dict) -> dict:
    """Pair an agent with a key (key_id=null → back to default/env)."""
    agent_id = (p.get("agent_id") or "").strip()
    if agent_id not in all_agents():
        return {"error": "unknown agent"}
    kid = p.get("key_id")
    db.set_agent_key(agent_id, int(kid) if kid else None)
    return {"ok": True, "map": db.agent_key_map()}


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
    # Per-connection client resolution: an agent's assigned key (persistent,
    # set in Admin Usage) wins; otherwise this connection's BYOK/default client.
    conn = {"default": _client}
    orch = Orchestrator(lambda aid: client_for_agent(aid, default=conn["default"]))

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

            if action == "delete_session":
                sid = msg.get("id")
                if sid:
                    db.delete_session(sid)
                    # if we deleted the active session, fall back to the latest/new one
                    if sid == current_id:
                        current = db.get_or_create_current()
                        current_id = current["id"]
                        await send_session_state(current)
                    else:
                        await send({"type": "sessions", "sessions": db.list_sessions()})
                continue

            # BYOK: use the user's own API key, or point at a local/custom
            # Anthropic-compatible endpoint (e.g. Ollama via a LiteLLM proxy = $0).
            if action == "set_key":
                key = (msg.get("api_key") or "").strip()
                provider = msg.get("provider", "anthropic")
                base_url = (msg.get("base_url") or "").strip()
                model = (msg.get("model") or "").strip()
                if base_url:
                    conn["default"] = _build_client(base_url, key or "local")
                    if model:
                        set_model(model)
                    await send({"type": "byok", "ok": True, "provider": "local (" + base_url + ")"})
                elif not key:
                    conn["default"] = _client
                    await send({"type": "byok", "ok": True, "provider": "default"})
                elif provider == "anthropic":
                    conn["default"] = _build_client("", key)
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
                    out = await run_jarvis(send, jgoal, orch.client_for("jarvis"), current_id)
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
