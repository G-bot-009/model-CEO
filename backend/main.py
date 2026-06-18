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
import time
import urllib.parse
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from . import db
from . import skills
from . import mcp_catalog
from . import mcp_oauth
from . import trading
from . import content_factory
from . import media
from . import projects as projects_mod
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


def model_for_agent(agent_id: str) -> str:
    """Per-agent model override (set on the agent card); else the global model."""
    try:
        m = db.get_settings().get(f"agent_model_{agent_id}")
        if m:
            return m
    except Exception:
        pass
    return get_model()


def _live_mcp_token(conn: dict) -> str:
    """Current access token for a directory connection, refreshing OAuth if expired."""
    tok = (conn.get("token") or "").strip()
    exp = conn.get("expires_at") or 0
    rt = (conn.get("refresh_token") or "").strip()
    turl = (conn.get("token_url") or "").strip()
    if rt and turl and exp and time.time() > (exp - 60):     # expired (or about to) → refresh
        try:
            r = mcp_oauth.refresh(turl, rt, conn.get("client_id") or "", conn.get("client_secret") or "")
            new = r.get("access_token")
            if new:
                new_exp = time.time() + int(r.get("expires_in", 3600)) if r.get("expires_in") else 0
                db.update_mcp_tokens(conn["conn_id"], new, r.get("refresh_token", ""), new_exp)
                return new
        except Exception:
            pass     # fall back to the (possibly stale) token; the run will retry/normal-fallback
    return tok


def mcp_for_agent(agent_id: str) -> list:
    """Per-agent MCP servers → mcp_servers entries for the Messages API."""
    out = []
    try:
        for m in db.list_agent_mcp(agent_id):
            if not m.get("url"):
                continue
            url = m["url"]
            tok = (m.get("token") or "").strip()
            cid = m.get("conn_id")
            if cid:                                  # directory connector → use the live token
                conn = db.get_mcp_connection(cid)
                if conn:
                    url = conn.get("url") or url
                    tok = _live_mcp_token(conn)
            tok = _resolve_secret(tok) or tok        # accept a secret_ref or a raw token
            e = {"type": "url", "name": m.get("name") or "mcp", "url": url}
            if tok:
                e["authorization_token"] = tok
            out.append(e)
    except Exception:
        pass
    return out


_orchestrator = Orchestrator(lambda aid: client_for_agent(aid), model_for_agent)
_orchestrator.mcp_for = mcp_for_agent

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


def _friendly_err(exc: Exception) -> str:
    """Turn raw SDK errors into a Thai hint the user can act on."""
    name = type(exc).__name__
    s = str(exc)
    if "authentication" in s.lower() or "invalid x-api-key" in s.lower() or name == "AuthenticationError":
        return "❌ API key ไม่ถูกต้อง — ใส่ ANTHROPIC_API_KEY ที่ถูกต้องในไฟล์ .env หรือไปผูกคีย์ที่เมนู Admin Usage แล้วลองใหม่"
    if "credit" in s.lower() or "billing" in s.lower():
        return "❌ เครดิต/ยอดเงินไม่พอ — เติมเครดิตที่ console.anthropic.com แล้วลองใหม่"
    if name in ("RateLimitError", "OverloadedError") or "rate" in s.lower() or "overloaded" in s.lower():
        return "⚠️ ใช้งานถี่เกินไป/ระบบหนาแน่น — รอสักครู่แล้วลองใหม่"
    if name in ("APIConnectionError", "APITimeoutError") or "connect" in s.lower() or "timeout" in s.lower():
        return "⚠️ เชื่อมต่อ Anthropic ไม่ได้ — เช็กอินเทอร์เน็ตแล้วลองใหม่"
    return f"{name}: {s}"


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
    import asyncio
    asyncio.create_task(_trading_loop())        # ticks active trading bots
    asyncio.create_task(_content_loop())        # autonomous content factory


LOGIN_USER = os.getenv("LOGIN_USER", "admin")
LOGIN_PASS = os.getenv("LOGIN_PASS", "admin")

# Public address of THIS app (set when behind a domain), e.g. https://app.iamceo.ai
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
# Your self-hosted n8n instance, e.g. https://n8n.iamceo.ai
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://localhost:5678").strip().rstrip("/")


def _public_base(request) -> str:
    """Base URL for outward-facing links (OAuth redirects, n8n callback).
    Prefers PUBLIC_BASE_URL so they're correct behind a reverse proxy/domain."""
    return PUBLIC_BASE_URL or str(request.base_url).rstrip("/")


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
    "lazada":    {"label": "Lazada", "auth": "https://auth.lazada.com/oauth/authorize", "token": "https://auth.lazada.com/rest/auth/token/create", "scope": "", "signed": True, "doc": "https://open.lazada.com/", "note": "Lazada Open Platform — สร้าง App แล้วรับ App Key/Secret (เรียก API แชท/ออเดอร์ด้วยลายเซ็น sign)"},
    "shopee":    {"label": "Shopee", "auth": "https://partner.shopeemobile.com/api/v2/shop/auth_partner", "token": "https://partner.shopeemobile.com/api/v2/auth/token/get", "scope": "", "signed": True, "doc": "https://open.shopee.com/", "note": "Shopee Open Platform — ใช้ Partner ID/Key + ลายเซ็น HMAC เชื่อมร้านเพื่ออ่าน/ตอบแชท"},
}
_oauth_state: dict = {}   # state -> (platform, code_verifier)


def _redirect_uri(request: Request, platform: str) -> str:
    return _public_base(request) + f"/oauth/{platform}/callback"


@app.get("/api/social")
async def social_list(request: Request) -> dict:
    connected = db.list_social()
    return {
        "platforms": [{"id": k, "label": v["label"], "connected": connected.get(k, False),
                       "signed": bool(v.get("signed")), "note": v.get("note", ""), "doc": v.get("doc", "")}
                      for k, v in SOCIAL.items()],
        "redirect_note": _public_base(request) + "/oauth/<platform>/callback",
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


# --- Social Inbox (unified comments/chats) ----------------------------------
def _line_token() -> str:
    row = db.get_connector("LINE OA")
    if not row or row.get("status") != "connected":
        return ""
    try:
        d = json.loads(row.get("config") or "{}")
        return d.get("access_token") or d.get("channel_id") or ""
    except Exception:
        return (row.get("config") or "").strip()


async def _line_push(to: str, text: str) -> bool:
    tok = _line_token()
    if not tok or not to:
        return False
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post("https://api.line.me/v2/bot/message/push",
                                  headers={"Authorization": f"Bearer {tok}"},
                                  json={"to": to, "messages": [{"type": "text", "text": text}]})
        return r.status_code < 400
    except Exception:
        return False


async def _ai_reply(incoming: str) -> str:
    """Short auto-reply via the model (used when auto-reply is ON)."""
    try:
        resp = await _client.messages.create(
            model=get_model(), max_tokens=400, **thinking_kwargs(),
            system="คุณเป็นแอดมินเพจที่สุภาพ ตอบลูกค้าสั้น กระชับ เป็นกันเอง เป็นภาษาไทย ไม่เกิน 3 ประโยค",
            messages=[{"role": "user", "content": incoming}])
        return next((b.text for b in resp.content if b.type == "text"), "").strip()
    except Exception:
        return ""


@app.get("/api/inbox")
async def inbox_list(filter: str = "all") -> dict:
    s = db.get_settings()
    return {
        "items": db.list_inbox(filter),
        "counts": db.inbox_counts(),
        "autoreply": s.get("inbox_autoreply", "false") == "true",
        "inbound_path": _inbound_path(),
    }


@app.post("/api/inbox/test")
async def inbox_test() -> dict:
    import random
    samples = [("LINE", "คุณน้ำหวาน", "สนใจแพ็กเกจรายเดือนค่ะ มีโปรอะไรบ้างคะ", "chat"),
               ("facebook", "คุณบีม", "ของจริงไหมครับ ใช้แล้วเป็นยังไงบ้าง", "comment"),
               ("instagram", "คุณแพรว", "ส่งของกี่วันถึงคะ", "chat")]
    p = random.choice(samples)
    db.add_inbox(p[0], p[1], p[2], p[3], thread="demo")
    return {"ok": True}


async def _ai_summary(texts: str) -> str:
    try:
        resp = await _client.messages.create(
            model=get_model(), max_tokens=200, **thinking_kwargs(),
            system="สรุปสั้นๆ ว่าลูกค้าคนนี้ต้องการ/สนใจอะไร เป็นภาษาไทย 1 ประโยค",
            messages=[{"role": "user", "content": texts[:2000]}])
        return next((b.text for b in resp.content if b.type == "text"), "").strip()
    except Exception:
        return ""


@app.get("/api/inbox/contacts")
async def inbox_contacts() -> dict:
    rows = db.list_inbox("all", 2000)
    groups: dict = {}
    for m in rows:
        k = (m["platform"], m["sender"])
        g = groups.setdefault(k, {"platform": m["platform"], "sender": m["sender"], "thread": m.get("thread", ""),
                                  "count": 0, "last_text": m["text"], "last_at": m["created_at"]})
        g["count"] += 1
    return {"contacts": sorted(groups.values(), key=lambda x: x["last_at"], reverse=True)}


@app.post("/api/inbox/contact/summary")
async def inbox_contact_summary(p: dict) -> dict:
    sender, platform = p.get("sender", ""), p.get("platform", "")
    rows = [m for m in db.list_inbox("all", 2000) if m["sender"] == sender and m["platform"] == platform]
    texts = "\n".join(m["text"] for m in rows)
    return {"summary": await _ai_summary(texts) if texts else ""}


@app.get("/api/inbox/stats")
async def inbox_stats() -> dict:
    from datetime import timedelta as _td
    rows = db.list_inbox("all", 5000)
    cutoff = (date.today() - _td(days=30)).isoformat()
    last30 = [m for m in rows if (m["created_at"] or "")[:10] >= cutoff]
    def count(items, **kw):
        return sum(1 for m in items if all(m.get(k) == v for k, v in kw.items()))
    return {
        "platforms": db.list_social(),
        "total": len(rows),
        "chats_30": count(last30, kind="chat"),
        "comments_30": count(last30, kind="comment"),
        "pending": count(rows, status="pending"),
        "replied": count(rows, status="replied"),
    }


@app.post("/api/inbox/broadcast")
async def inbox_broadcast(p: dict) -> dict:
    text = (p.get("text") or "").strip()
    if not text:
        return {"error": "พิมพ์ข้อความก่อน"}
    personalize = bool(p.get("personalize"))
    ids = p.get("ids") or []
    rows = db.list_inbox("all", 2000)
    # unique LINE threads (the only platform we can push to directly)
    targets, seen = [], set()
    for m in rows:
        if ids and m["id"] not in ids:
            continue
        if m["platform"] == "LINE" and m.get("thread") and m["thread"] not in seen:
            seen.add(m["thread"]); targets.append(m)
    sent = 0
    for m in targets:
        msg = text
        if personalize:
            try:
                resp = await _client.messages.create(
                    model=get_model(), max_tokens=300, **thinking_kwargs(),
                    system="ปรับข้อความบรอดแคสต์ให้เหมือนส่งหาลูกค้าคนนี้โดยตรง คงความหมาย/ราคา/ลิงก์เดิมทุกอย่าง ภาษาไทย",
                    messages=[{"role": "user", "content": f"ลูกค้า: {m['sender']} (เคยถาม: {m['text'][:120]})\nข้อความ: {text}"}])
                msg = next((b.text for b in resp.content if b.type == "text"), text).strip() or text
            except Exception:
                msg = text
        if await _line_push(m["thread"], msg):
            sent += 1
    return {"ok": True, "sent": sent, "targets": len(targets)}


@app.post("/api/inbox/draft")
async def inbox_draft(p: dict) -> dict:
    item = db.get_inbox(int(p.get("id") or 0))
    if not item:
        return {"error": "ไม่พบข้อความ"}
    return {"text": await _ai_reply(item.get("text") or "")}


@app.post("/api/inbox/reply")
async def inbox_reply(p: dict) -> dict:
    item = db.get_inbox(int(p.get("id") or 0))
    if not item:
        return {"error": "ไม่พบข้อความ"}
    text = (p.get("text") or "").strip()
    if not text:
        return {"error": "พิมพ์ข้อความตอบก่อน"}
    sent = False
    if item["platform"] == "LINE" and item.get("thread"):
        sent = await _line_push(item["thread"], text)
    db.set_inbox_reply(item["id"], text)
    return {"ok": True, "sent": sent}


# --- Tool Registry (G Office × n8n contract) --------------------------------
_REGISTRAR_PROMPT = """You are the Automation Registrar of the G Office system.
Convert an n8n workflow (given as JSON, a description, or a flow image caption)
into ONE valid Tool Registry entry.

Rules:
- 1 workflow = 1 tool. Map owner_department to an EXISTING department only
  (marketing, research, trader, admin, developer, ops) — never invent one.
- tool_id is a slug "{dept_prefix}_{action}", e.g. "mkt_url_to_article".
- Summarize the inputs the flow needs and outputs it returns into input_schema
  and output_schema. Read the trigger node for method/url (prefer a Webhook).
- NEVER hardcode tokens/secrets — reference them via auth.secret_ref.
- Output ONLY the JSON object (no prose, no code fences), with these keys:
  tool_id, version, status, display_name_th, display_name_en, description,
  owner_department, owner_agent, category, tags, trigger{type,method,url,
  auth{type,header,secret_ref}}, input_schema, output_schema,
  callback{expected,mode,timeout_sec}, limits, cost, metadata.
Write Thai for display_name_th and description."""


def _resolve_secret(ref: str) -> str:
    if not ref:
        return ""
    return os.getenv(ref) or db.get_settings().get(ref) or ""


@app.post("/api/tools/register-ai")
async def tools_register_ai(p: dict) -> dict:
    wf = (p.get("workflow") or "").strip()
    if not wf:
        return {"error": "วาง workflow (JSON/คำอธิบาย) ก่อน"}
    try:
        resp = await _client.messages.create(
            model=get_model(), max_tokens=2500, **thinking_kwargs(),
            system=_REGISTRAR_PROMPT, messages=[{"role": "user", "content": wf[:8000]}])
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        entry = json.loads(raw)
    except Exception:
        return {"error": "AI ไม่ได้คืน JSON ที่ถูกต้อง", "raw": text[:1500]}
    return {"ok": True, "entry": entry}


SHARED_DIR = Path(__file__).resolve().parent / "shared"


@app.post("/api/files/upload")
async def file_upload(file: UploadFile = File(...), storage: str = "vol") -> dict:
    """Shared Storage: store the file and return a file_ref (never the binary).
    Default 'vol' (self-host). gdrive/s3 need configured creds → AUTH_FAILED if missing."""
    if storage == "gdrive" and not (db.get_social("google") or {}).get("access_token"):
        return JSONResponse({"error": {"code": "AUTH_FAILED", "message": "ยังไม่ได้เชื่อม Google Drive"}}, status_code=401)
    if storage == "s3" and not _resolve_secret("S3_BUCKET"):
        return JSONResponse({"error": {"code": "AUTH_FAILED", "message": "ยังไม่ได้ตั้งค่า S3"}}, status_code=401)
    SHARED_DIR.mkdir(exist_ok=True)
    fid = secrets.token_hex(8)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "file")[:80] or "file"
    dest = SHARED_DIR / f"{fid}__{safe}"
    dest.write_bytes(await file.read())
    mime = file.content_type or "application/octet-stream"
    db.add_file(fid, file.filename or safe, mime, str(dest))
    return {"file_ref": f"vol://{fid}", "name": file.filename or safe, "mime": mime}


@app.get("/api/files/{fid}")
async def file_download(fid: str):
    f = db.get_file(fid)
    if not f or not Path(f["path"]).exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f["path"], filename=f["name"], media_type=f["mime"])


@app.get("/api/tools")
async def tools_list() -> dict:
    return {"tools": db.list_tools(), "runs": db.list_tool_runs(40)}


@app.post("/api/tools")
async def tools_save(entry: dict) -> dict:
    tid = (entry.get("tool_id") or "").strip()
    if not tid:
        return {"error": "ต้องมี tool_id"}
    if not (entry.get("trigger") or {}).get("url"):
        return {"error": "ต้องมี trigger.url (Webhook URL ของ n8n)"}
    entry.setdefault("status", "active")
    db.upsert_tool(entry)
    db.add_decision("tool", f"registered {tid}")
    return {"ok": True, "tools": db.list_tools()}


@app.post("/api/tools/delete")
async def tools_delete(p: dict) -> dict:
    db.delete_tool(p.get("tool_id", ""))
    return {"ok": True, "tools": db.list_tools()}


@app.post("/api/tools/run")
async def tools_run(p: dict, request: Request) -> dict:
    import httpx
    tool = db.get_tool(p.get("tool_id", ""))
    if not tool:
        return {"error": "ไม่พบ tool นี้"}
    trig = tool.get("trigger", {})
    url = trig.get("url")
    if not url:
        return {"error": "tool นี้ไม่มี trigger.url"}
    task_id = "tsk_" + secrets.token_hex(8)
    trace_id = "trc_" + secrets.token_hex(8)
    inputs = dict(p.get("inputs") or {})
    files = p.get("files") or []
    if files:                                  # files travel as file_ref only, never binary
        inputs["files"] = files
    callback_url = _public_base(request) + "/api/n8n/callback"
    payload = {
        "task_id": task_id, "trace_id": trace_id, "tool_id": tool["tool_id"],
        "version": tool.get("version", "1.0.0"),
        "requested_by": {"type": "user", "id": "admin", "department": tool.get("owner_department", "")},
        "inputs": inputs, "callback_url": callback_url, "idempotency_key": task_id,
        "issued_at": db._now(),
    }
    headers = {"Content-Type": "application/json"}
    auth = trig.get("auth") or {}
    if auth.get("type") == "header_token" and auth.get("header"):
        headers[auth["header"]] = _resolve_secret(auth.get("secret_ref", ""))
    db.add_tool_run(task_id, tool["tool_id"], trace_id, inputs)
    method = (trig.get("method") or "POST").upper()
    sync = (tool.get("callback") or {}).get("mode") == "sync"
    try:
        async with httpx.AsyncClient(timeout=(tool.get("callback") or {}).get("timeout_sec", 60)) as client:
            r = await client.request(method, url, json=payload, headers=headers)
        if r.status_code >= 400:
            db.finish_tool_run(task_id, "error", error={"code": "UPSTREAM_API_ERROR", "message": f"HTTP {r.status_code}: {r.text[:200]}"})
            return {"task_id": task_id, "status": "error", "message": f"n8n ตอบ {r.status_code}"}
        if sync:
            try:
                body = r.json()
                db.finish_tool_run(task_id, body.get("status", "success"), outputs=body.get("outputs"), error=body.get("error"))
            except Exception:
                db.finish_tool_run(task_id, "success", outputs={"raw": r.text[:500]})
    except Exception as exc:
        db.finish_tool_run(task_id, "error", error={"code": "INTERNAL_ERROR", "message": str(exc)})
        return {"task_id": task_id, "status": "error", "message": str(exc)}
    return {"task_id": task_id, "status": "pending" if not sync else "done"}


@app.post("/api/n8n/callback")
async def n8n_callback(p: dict) -> dict:
    task_id = p.get("task_id")
    if not task_id:
        return JSONResponse({"error": "task_id required"}, status_code=400)
    status = p.get("status", "success")
    ok = db.finish_tool_run(task_id, status, outputs=p.get("outputs"), error=p.get("error"))
    if not ok:
        return JSONResponse({"error": "unknown task_id"}, status_code=404)
    db.add_decision("tool_callback", f"{p.get('tool_id','')}: {status}")
    return {"ok": True}


@app.post("/api/inbox/inbound/{token}")
async def inbox_inbound(token: str, p: dict) -> dict:
    if token != db.get_settings().get("inbound_token"):
        return JSONResponse({"error": "invalid token"}, status_code=404)
    platform = (p.get("platform") or "webhook").strip()
    sender = (p.get("sender") or p.get("from") or "ลูกค้า").strip()
    text = (p.get("text") or p.get("message") or "").strip()
    kind = (p.get("kind") or "chat").strip()
    thread = (p.get("thread") or p.get("user_id") or "").strip()
    if not text:
        return {"error": "no text"}
    iid = db.add_inbox(platform, sender, text, kind, thread)
    # optional auto-reply (LINE only, when toggle is ON)
    if db.get_settings().get("inbox_autoreply") == "true" and platform == "LINE" and thread:
        reply = await _ai_reply(text)
        if reply and await _line_push(thread, reply):
            db.set_inbox_reply(iid, reply)
    return {"ok": True, "id": iid}


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
    kmap = db.agent_key_map()
    return {
        "agents": [
            {
                "id": a.id,
                "name": s.get(f"agent_name_{a.id}") or a.name,
                "key_id": kmap.get(a.id),
                "model": s.get(f"agent_model_{a.id}") or "",
                "title": a.title,
                "emoji": a.emoji,
                "tags": list(a.tags),
                "color": a.color,
                "custom": a.id.startswith("x_"),
                "category": a.category,
                "skills": list(a.skills),
                "desc": a.desc or a.title,
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
        return {"error": _friendly_err(exc)}
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
        "desc": a.desc or a.title,
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
        _m = model_for_agent(agent)
        resp = await client_for_agent(agent).messages.create(
            model=_m,
            max_tokens=8000,
            **thinking_kwargs(_m),
            system=_ag[agent].system,
            messages=[{"role": "user", "content": instr}],
        )
    except Exception as exc:
        return {"error": _friendly_err(exc)}
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
    return {"items": _n8n_data(), "n8n_base": N8N_BASE_URL}


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


@app.get("/api/agent/{agent_id}/mcp")
async def agent_mcp_list(agent_id: str) -> dict:
    rows = db.list_agent_mcp(agent_id)
    for r in rows:                     # mask the token
        t = r.get("token") or ""
        r["has_token"] = bool(t)
        r["token"] = (t[:4] + "…") if t else ""
    return {"mcp": rows}


@app.post("/api/agent/mcp")
async def agent_mcp_add(p: dict) -> dict:
    agent_id = (p.get("agent_id") or "").strip()
    if agent_id not in all_agents():
        return {"error": "unknown agent"}
    url = (p.get("url") or "").strip()
    if not url:
        return {"error": "ต้องมี MCP server URL"}
    db.add_agent_mcp(agent_id, (p.get("name") or "mcp").strip(), url, (p.get("token") or "").strip())
    return {"ok": True}


@app.post("/api/agent/mcp/delete")
async def agent_mcp_delete(p: dict) -> dict:
    if p.get("id"):
        db.delete_agent_mcp(int(p["id"]))
    return {"ok": True}


# --- Connector directory (connect API once → toggle onto matching agents) ----
def _matching_agents(entry) -> list:
    """[(agent_id, agent_name)] for agents whose capabilities fit this connector."""
    return [(a.id, a.name) for a in all_agents().values() if mcp_catalog.matches(a, entry)]


def _apply_connector_agents(conn_id: str, agent_ids: set) -> None:
    """Enable a connector for exactly ``agent_ids`` (restricted to matching agents)."""
    entry = mcp_catalog.CATALOG_BY_ID.get(conn_id)
    conn = db.get_mcp_connection(conn_id)
    if not entry or not conn:
        return
    matching = {aid for aid, _ in _matching_agents(entry)}
    db.clear_agent_mcp_conn(conn_id)
    for aid in (agent_ids & matching):
        db.add_agent_mcp(aid, conn["name"], conn["url"], conn.get("token") or "", conn_id=conn_id)


@app.get("/api/mcp/directory")
async def mcp_directory() -> dict:
    """The whole connector catalog with connection state + which agents each fits."""
    connected = db.connected_mcp_ids()
    conns = {c["conn_id"]: c for c in db.list_mcp_connections()}
    out = []
    for c in mcp_catalog.CATALOG:
        conn = conns.get(c["id"])
        is_conn = c["id"] in connected
        enabled = db.conn_agent_ids(c["id"]) if is_conn else set()
        matched = [{"id": aid, "name": nm, "enabled": aid in enabled} for aid, nm in _matching_agents(c)]
        out.append({
            "id": c["id"], "name": c["name"], "emoji": c["emoji"], "desc": c["desc"],
            "get": c["get"], "default_url": c["url"],
            "connected": is_conn,
            "oauth": bool(c.get("oauth")),
            "token_auth": bool(c.get("token_auth")),
            "url": (conn or {}).get("url") or c["url"],
            "agents": [m["name"] for m in matched],
            "matched": matched,
        })
    return {"connectors": out}


@app.post("/api/mcp/connect")
async def mcp_connect(p: dict) -> dict:
    cid = (p.get("conn_id") or "").strip()
    entry = mcp_catalog.CATALOG_BY_ID.get(cid)
    if not entry:
        return {"error": "unknown connector"}
    if not entry.get("token_auth"):
        return {"error": "connector นี้เชื่อมแบบวาง token (API) ไม่ได้ — ใช้ปุ่ม 🔐 เชื่อมด้วย OAuth"}
    url = (p.get("url") or "").strip() or entry["url"]
    token = (p.get("token") or "").strip()
    db.connect_mcp(cid, entry["name"], url, token)
    # auto-enable for every matching agent the moment it's connected
    _apply_connector_agents(cid, {aid for aid, _ in _matching_agents(entry)})
    return {"ok": True}


@app.post("/api/mcp/connector/agents")
async def mcp_connector_agents(p: dict) -> dict:
    """Choose exactly which matching agents a connected connector is enabled for."""
    cid = (p.get("conn_id") or "").strip()
    if not db.get_mcp_connection(cid):
        return {"error": "ยังไม่ได้เชื่อม API ของ connector นี้"}
    ids = set(p.get("agent_ids") or [])
    _apply_connector_agents(cid, ids)
    return {"ok": True}


@app.post("/api/mcp/disconnect")
async def mcp_disconnect(p: dict) -> dict:
    cid = (p.get("conn_id") or "").strip()
    if cid:
        db.disconnect_mcp(cid)
    return {"ok": True}


# --- One-click OAuth for a connector (no manual token paste) -----------------
_MCP_OAUTH_PENDING: dict = {}          # state -> {conn_id, verifier, token_endpoint, client_id, client_secret, redirect_uri}


def _mcp_redirect_uri(request: Request) -> str:
    return _public_base(request) + "/oauth/mcp-callback"


@app.get("/oauth/mcp/{conn_id}/start")
async def mcp_oauth_start(conn_id: str, request: Request):
    entry = mcp_catalog.CATALOG_BY_ID.get(conn_id)
    if not entry:
        return RedirectResponse("/?mcp_error=" + urllib.parse.quote("ไม่รู้จัก connector นี้"))
    redirect_uri = _mcp_redirect_uri(request)
    try:
        meta = mcp_oauth.discover(entry["url"])
        client = db.get_mcp_oauth_client(conn_id)
        if client and client.get("client_id"):
            client_id, client_secret = client["client_id"], client.get("client_secret") or ""
        elif meta.get("registration_endpoint"):
            client_id, client_secret = mcp_oauth.register_client(meta["registration_endpoint"], redirect_uri)
            db.save_mcp_oauth_client(conn_id, client_id or "", client_secret or "")
        else:
            return RedirectResponse("/?mcp_error=" + urllib.parse.quote(
                "เซิร์ฟเวอร์นี้ไม่รองรับการลงทะเบียนอัตโนมัติ — ใช้วิธีวาง token แทน"))
        if not client_id:
            return RedirectResponse("/?mcp_error=" + urllib.parse.quote("ลงทะเบียน client ไม่สำเร็จ"))
        verifier, challenge = mcp_oauth.pkce()
        state = secrets.token_urlsafe(24)
        _MCP_OAUTH_PENDING[state] = {
            "conn_id": conn_id, "verifier": verifier, "token_endpoint": meta["token_endpoint"],
            "client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri,
        }
        url = mcp_oauth.build_authorize_url(meta, client_id, redirect_uri, state, challenge,
                                            scope=entry.get("scope"), resource=entry["url"])
        return RedirectResponse(url)
    except Exception as exc:
        return RedirectResponse("/?mcp_error=" + urllib.parse.quote(f"เชื่อมไม่สำเร็จ: {str(exc)[:120]}"))


@app.get("/oauth/mcp-callback")
async def mcp_oauth_callback(request: Request):
    q = request.query_params
    state, code, err = q.get("state"), q.get("code"), q.get("error")
    pend = _MCP_OAUTH_PENDING.pop(state or "", None)
    if err or not code or not pend:
        return RedirectResponse("/?mcp_error=" + urllib.parse.quote("ผู้ใช้ยกเลิก หรือคำขอหมดอายุ"))
    try:
        tok = mcp_oauth.exchange_code(pend["token_endpoint"], code, pend["redirect_uri"],
                                      pend["client_id"], pend["verifier"], pend.get("client_secret") or "")
        access = tok.get("access_token")
        if not access:
            return RedirectResponse("/?mcp_error=" + urllib.parse.quote("ไม่ได้รับ access token"))
        expires_at = time.time() + int(tok.get("expires_in", 3600)) if tok.get("expires_in") else 0
        entry = mcp_catalog.CATALOG_BY_ID[pend["conn_id"]]
        db.connect_mcp_oauth(pend["conn_id"], entry["name"], entry["url"], access,
                             tok.get("refresh_token", ""), expires_at, pend["token_endpoint"],
                             pend["client_id"], pend.get("client_secret") or "")
        _apply_connector_agents(pend["conn_id"], {aid for aid, _ in _matching_agents(entry)})
        return RedirectResponse("/?mcp_connected=" + urllib.parse.quote(pend["conn_id"]))
    except Exception as exc:
        return RedirectResponse("/?mcp_error=" + urllib.parse.quote(f"แลก token ไม่สำเร็จ: {str(exc)[:120]}"))


@app.get("/api/agent/{agent_id}/connectors")
async def agent_connectors(agent_id: str) -> dict:
    """Connectors that BOTH match this agent AND are connected — each toggleable."""
    agents = all_agents()
    if agent_id not in agents:
        return {"connectors": []}
    agent = agents[agent_id]
    connected = db.connected_mcp_ids()
    enabled = db.agent_mcp_conn_ids(agent_id)
    out = []
    for c in mcp_catalog.CATALOG:
        if c["id"] not in connected or not mcp_catalog.matches(agent, c):
            continue          # only connected + matching connectors are shown
        out.append({
            "id": c["id"], "name": c["name"], "emoji": c["emoji"], "desc": c["desc"],
            "enabled": c["id"] in enabled,
        })
    return {"connectors": out}


@app.post("/api/agent/connector/toggle")
async def agent_connector_toggle(p: dict) -> dict:
    agent_id = (p.get("agent_id") or "").strip()
    cid = (p.get("conn_id") or "").strip()
    on = bool(p.get("on"))
    if agent_id not in all_agents():
        return {"error": "unknown agent"}
    entry = mcp_catalog.CATALOG_BY_ID.get(cid)
    conn = db.get_mcp_connection(cid)
    if not entry or not conn:
        return {"error": "ยังไม่ได้เชื่อม API ของ connector นี้"}
    db.delete_agent_mcp_conn(agent_id, cid)          # avoid duplicates
    if on:
        db.add_agent_mcp(agent_id, conn["name"], conn["url"], conn.get("token") or "", conn_id=cid)
    return {"ok": True, "enabled": on}


@app.post("/api/agent/model")
async def assign_agent_model(p: dict) -> dict:
    """Per-agent model override ('' = use global)."""
    agent_id = (p.get("agent_id") or "").strip()
    if agent_id not in all_agents():
        return {"error": "unknown agent"}
    db.set_settings({f"agent_model_{agent_id}": (p.get("model") or "").strip()})
    return {"ok": True}


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
    _m = model_for_agent("jarvis")
    collected: list[str] = []
    async with client.messages.stream(
        model=_m,
        max_tokens=8000,
        **thinking_kwargs(_m),
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


# ============================ Trading bots ==================================
# Default mode is paper (no real money). Live requires opt-in + trade-enabled
# keys stored in the matching connector. Safety: per-day target/loss kill-switch
# + max position size live in the strategy config.
_TRADE_CRED_CONNECTOR = {"binance": "Binance Futures", "bybit": "Bybit", "mt5": "MetaTrader 5 (Forex)"}


def _trade_creds(exchange: str) -> dict:
    """Read API creds for live trading from the matching connector config."""
    row = db.get_connector(_TRADE_CRED_CONNECTOR.get(exchange, ""))
    if not row or not row.get("config"):
        return {}
    try:
        return json.loads(row["config"])
    except Exception:
        return {}


def _tick_bot(bot: dict) -> None:
    """Advance one bot by a single tick (used by the loop and manual tick)."""
    ex, sym = bot["exchange"], bot["symbol"]
    cfg, state = bot["config"], bot.get("state") or trading.new_state()
    creds = _trade_creds(ex)
    try:
        price = trading.fetch_price(ex, sym, creds)
    except Exception as exc:
        db.add_trade_log(bot["id"], "error", f"อ่านราคาไม่ได้: {type(exc).__name__}", 0.0)
        return
    new_state, events = trading.step(cfg, state, price)
    for e in events:
        # in live mode, mirror entries/exits to the real exchange
        if bot["mode"] == "live" and e["kind"] in ("open",) and creds:
            try:
                qty = (new_state.get("pos") or {}).get("qty") or 0
                trading.place_market(ex, creds, sym, cfg.get("side", "long"), round(qty, 6))
                e["text"] += " · ส่งออเดอร์จริงแล้ว"
            except Exception as exc:
                e["text"] += f" · ⚠️ ส่งออเดอร์จริงไม่สำเร็จ: {type(exc).__name__}"
                new_state["halted"] = True
        db.add_trade_log(bot["id"], e["kind"], e["text"], e.get("pnl", 0.0))
    db.set_bot_state(bot["id"], new_state)


async def _trading_loop() -> None:
    import asyncio
    while True:
        await asyncio.sleep(15)
        try:
            for bot in db.list_bots():
                if bot.get("status") == "running":
                    await asyncio.to_thread(_tick_bot, bot)
        except Exception:
            pass


@app.get("/api/trading/bots")
async def trading_bots() -> dict:
    return {"bots": db.list_bots(), "default_config": trading.DEFAULT_CONFIG}


@app.post("/api/trading/bot")
async def trading_bot_save(p: dict) -> dict:
    name = (p.get("name") or "บอทเทรด").strip()
    ex = (p.get("exchange") or "binance").lower()
    sym = (p.get("symbol") or "BTCUSDT").upper().strip()
    mode = "live" if p.get("mode") == "live" else "paper"
    cfg = trading.normalize_config(p.get("config") or {})
    bid = p.get("id")
    if bid:
        db.update_bot(int(bid), name, ex, sym, mode, cfg)
        return {"ok": True, "id": int(bid)}
    nid = db.create_bot(name, ex, sym, mode, cfg, trading.new_state())
    return {"ok": True, "id": nid}


@app.post("/api/trading/bot/start")
async def trading_bot_start(p: dict) -> dict:
    bot = db.get_bot(int(p.get("id")))
    if not bot:
        return {"error": "ไม่พบบอท"}
    if bot["mode"] == "live" and not _trade_creds(bot["exchange"]):
        return {"error": f"โหมด live ต้องเชื่อมคีย์ {_TRADE_CRED_CONNECTOR.get(bot['exchange'],'')} ก่อน (เปิดสิทธิ์เทรด)"}
    db.set_bot_status(bot["id"], "running")
    db.add_trade_log(bot["id"], "start", f"เริ่มบอท ({bot['mode']})", 0.0)
    return {"ok": True}


@app.post("/api/trading/bot/stop")
async def trading_bot_stop(p: dict) -> dict:
    db.set_bot_status(int(p.get("id")), "stopped")
    db.add_trade_log(int(p.get("id")), "stop", "หยุดบอท", 0.0)
    return {"ok": True}


@app.post("/api/trading/bot/delete")
async def trading_bot_delete(p: dict) -> dict:
    db.delete_bot(int(p.get("id")))
    return {"ok": True}


@app.post("/api/trading/bot/tick")
async def trading_bot_tick(p: dict) -> dict:
    """Manual single tick (handy for paper testing without waiting for the loop)."""
    bot = db.get_bot(int(p.get("id")))
    if not bot:
        return {"error": "ไม่พบบอท"}
    import asyncio
    await asyncio.to_thread(_tick_bot, bot)
    return {"ok": True, "state": db.get_bot(bot["id"])["state"]}


@app.get("/api/trading/bot/{bot_id}/log")
async def trading_bot_log(bot_id: int) -> dict:
    return {"log": db.list_trade_log(bot_id, 60)}


@app.post("/api/trading/strategy/ai")
async def trading_strategy_ai(p: dict) -> dict:
    """Claude turns a Thai description into a strategy config (rules, not predictions)."""
    desc = (p.get("desc") or "").strip()
    if not desc:
        return {"error": "ใส่คำอธิบายกลยุทธ์"}
    prompt = (
        "คุณเป็นผู้ช่วยตั้งค่าบอทเทรดแบบมีกฎ (ไม่ทำนายตลาด). จากคำอธิบายของผู้ใช้ "
        "ให้ออกค่า JSON ตามสคีมานี้เท่านั้น (ตัวเลขสมเหตุผล ปลอดภัย):\n"
        f"{json.dumps(trading.DEFAULT_CONFIG, ensure_ascii=False)}\n\n"
        f"คำอธิบาย: {desc}\n\nตอบเป็น JSON อย่างเดียว."
    )
    try:
        resp = await client_for_agent("trader").messages.create(
            model=model_for_agent("trader"), max_tokens=600,
            **thinking_kwargs(model_for_agent("trader")),
            messages=[{"role": "user", "content": prompt}])
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        cfg = trading.normalize_config(json.loads(m.group(0)) if m else {})
        return {"ok": True, "config": cfg}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


# ======================= Autonomous content factory =========================
_SENDABLE = {"Slack", "Discord", "Telegram", "LINE OA", "Webhook → Make/Zapier"}


def _media_cfg(kind: str) -> dict:
    """Selected provider/model/key for a media type (image|video|voice)."""
    s = db.get_settings()
    default_prov = media.PROVIDERS[kind][0]["id"]
    prov = s.get(f"media_{kind}_provider") or default_prov
    model = s.get(f"media_{kind}_model") or ""
    if not model:
        for p in media.PROVIDERS[kind]:
            if p["id"] == prov and p["models"]:
                model = p["models"][0]
                break
    return {"provider": prov, "model": model, "key": s.get(f"media_{kind}_key", "")}


def _content_brief() -> str:
    s = db.get_settings()
    return (s.get("content_brief") or s.get("company_name") or
            "ธุรกิจของผู้ใช้ — โทนเป็นกันเอง น่าเชื่อถือ").strip()


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


async def _claude_text(agent_id: str, model: str, prompt: str, max_tokens: int = 900) -> str:
    resp = await client_for_agent(agent_id).messages.create(
        model=model, max_tokens=max_tokens, **thinking_kwargs(model),
        messages=[{"role": "user", "content": prompt}])
    return next((b.text for b in resp.content if b.type == "text"), "")


async def run_content_cycle(rules: dict) -> dict:
    """One factory cycle: scout→decide→write→art→queue. Returns the planned post."""
    rules = content_factory.normalize_rules(rules)
    model = content_factory.brain_model(rules)
    raw = await _claude_text("content", model, content_factory.CYCLE_PROMPT.format(brief=_content_brief()))
    data = _parse_json(raw)
    caption = (data.get("caption") or "").strip()
    if not caption:
        raise RuntimeError("โมเดลไม่คืนแคปชั่น")
    topic = (data.get("topic") or "").strip()
    img_prompt = (data.get("image_prompt") or topic or caption[:80]).strip()

    # safety gate
    safe = True
    try:
        sd = _parse_json(await _claude_text("content", model, content_factory.SAFETY_PROMPT.format(caption=caption), 200))
        safe = bool(sd.get("safe", True))
        safe_reason = sd.get("reason", "")
    except Exception:
        safe_reason = ""

    # art direction: real image via the chosen provider (BYO key) else SVG
    media_kind, media_ref = "none", ""
    img = _media_cfg("image")
    if rules["media_per_day"] != 0:
        if img["key"]:
            try:
                im = await __import__("asyncio").to_thread(
                    media.generate_image, img["provider"], img["model"], img["key"], img_prompt)
                media_kind, media_ref = "image", f"data:{im['mime']};base64,{im['b64']}"
            except Exception:
                media_kind = "none"
        if media_kind == "none":
            try:
                w, h = IMG_SIZES.get("1:1", (600, 600))
                svg = _extract_svg(await _claude_text("designer", model,
                        f"Create a single self-contained <svg width='{w}' height='{h}'> illustration for: {img_prompt}. Return only the SVG.", 4000))
                if svg:
                    media_kind, media_ref = "svg", svg
            except Exception:
                pass

    platforms = [n for n, st in db.list_connectors().items() if st == "connected" and n in _SENDABLE]
    if not safe:
        status = "blocked"
    elif rules["post_mode"] == "auto":
        status = "queued"
    else:
        status = "draft"
    pid = db.create_planned("social", topic, caption, media_kind, media_ref, platforms, status, db._now())
    if not safe:
        db.set_planned_status(pid, "blocked", result=f"ไม่ผ่านการกรอง: {safe_reason}")
    db.add_decision("content_factory", f"{status}: {topic[:50]}")
    return db.get_planned(pid)


async def publish_planned(post: dict) -> dict:
    """Push a queued post to every connected sendable channel + a safety re-check."""
    import asyncio
    caption = post.get("caption") or ""
    results = []
    for name in (post.get("platforms") or []):
        row = db.get_connector(name)
        if not row or row.get("status") != "connected":
            continue
        try:
            r = await _send_to_connector(name, row.get("config") or "", caption)
            results.append(f"{name}: {'✓' if r.get('ok') else '✗'}")
        except Exception as exc:
            results.append(f"{name}: ✗ {type(exc).__name__}")
    summary = " · ".join(results) or "ไม่มีช่องที่เชื่อม"
    db.set_planned_status(post["id"], "posted", posted_at=db._now(), result=summary)
    return {"ok": True, "summary": summary}


async def _content_loop() -> None:
    import asyncio
    while True:
        await asyncio.sleep(60)
        try:
            rules = content_factory.normalize_rules(db.get_content_rules())
            ok, _ = content_factory.should_produce(rules, db.count_planned_today(), db.last_planned_iso())
            if ok:
                await run_content_cycle(rules)
            for post in db.due_planned():       # publish anything queued + due
                await publish_planned(post)
        except Exception:
            pass


@app.get("/api/content/rules")
async def content_rules_get() -> dict:
    return {"rules": content_factory.normalize_rules(db.get_content_rules()),
            "defaults": content_factory.DEFAULT_RULES}


@app.post("/api/content/rules")
async def content_rules_set(p: dict) -> dict:
    rules = content_factory.normalize_rules(p.get("rules") or p)
    db.set_content_rules(rules)
    return {"ok": True, "rules": rules}


@app.get("/api/content/board")
async def content_board() -> dict:
    rules = content_factory.normalize_rules(db.get_content_rules())
    return {"posts": db.list_planned(60), "today": db.count_planned_today(),
            "rules": rules,
            "media_connected": {k: bool(_media_cfg(k)["key"]) for k in ("image", "video", "voice")}}


@app.post("/api/content/run")
async def content_run(p: dict) -> dict:
    try:
        post = await run_content_cycle(db.get_content_rules())
        return {"ok": True, "post": post}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.post("/api/content/post/approve")
async def content_approve(p: dict) -> dict:
    db.set_planned_status(int(p.get("id")), "queued")
    return {"ok": True}


@app.post("/api/content/post/publish")
async def content_publish(p: dict) -> dict:
    post = db.get_planned(int(p.get("id")))
    if not post:
        return {"error": "ไม่พบโพสต์"}
    return await publish_planned(post)


@app.post("/api/content/post/delete")
async def content_delete(p: dict) -> dict:
    db.delete_planned(int(p.get("id")))
    return {"ok": True}


@app.get("/api/content/media-keys")
async def content_media_keys() -> dict:
    sel = {}
    for kind in ("image", "video", "voice"):
        c = _media_cfg(kind)
        sel[kind] = {"provider": c["provider"], "model": c["model"],
                     "key": (c["key"][:4] + "…") if c["key"] else "", "has": bool(c["key"])}
    return {"providers": media.PROVIDERS, "selection": sel}


@app.post("/api/content/media-keys")
async def content_media_keys_set(p: dict) -> dict:
    upd = {}
    for kind in ("image", "video", "voice"):
        block = p.get(kind) or {}
        if block.get("provider"):
            upd[f"media_{kind}_provider"] = block["provider"].strip()
        if block.get("model") is not None:
            upd[f"media_{kind}_model"] = (block.get("model") or "").strip()
        if block.get("key"):                # only overwrite the key when a new one is given
            upd[f"media_{kind}_key"] = block["key"].strip()
    if upd:
        db.set_settings(upd)
    return {"ok": True}


# ============================== Projects ====================================
def _stage_client(stage: dict):
    """Per-stage engine override (a chosen API key), else None (= per-agent default)."""
    kid = stage.get("key_id")
    if kid:
        try:
            row = db.get_api_key(int(kid))
            if row:
                return _build_client(row.get("base_url"), row.get("secret"))
        except Exception:
            pass
    return None


async def _run_agent(agent_id: str, goal: str, client=None) -> str:
    """Run a single agent on a goal with its own system prompt; return text.
    Raises on API failure so the caller can stop the pipeline."""
    ag = all_agents().get(agent_id)
    if not ag:
        raise RuntimeError(f"ไม่พบเอเจนต์ {agent_id}")
    model = model_for_agent(agent_id)
    cl = client or client_for_agent(agent_id)
    resp = await cl.messages.create(
        model=model, max_tokens=2500, system=ag.system,
        **thinking_kwargs(model), messages=[{"role": "user", "content": goal}])
    return next((b.text for b in resp.content if b.type == "text"), "")


async def run_project_stage(stage: dict) -> bool:
    """Run a stage's agents in order. Stops at the first agent whose API fails,
    marks the stage 'failed' (red in the UI), and returns False. Returns True
    only when every agent finished cleanly."""
    db.set_stage_result(stage["id"], "running", stage.get("result") or "")
    client = _stage_client(stage)
    ags = all_agents()
    parts, ok = [], True
    for aid in (stage.get("agents") or []):
        name = ags[aid].name if aid in ags else aid
        emoji = ags[aid].emoji if aid in ags else "•"
        try:
            out = await _run_agent(aid, stage["goal"], client)
            parts.append(f"### {emoji} {name}\n{out}")
        except Exception as exc:
            parts.append(f"### {emoji} {name}\n⚠️ {_friendly_err(exc)}")
            ok = False
            break                      # halt at the failing agent — do not skip ahead
    db.set_stage_result(stage["id"], "done" if ok else "failed", "\n\n".join(parts))
    return ok


async def _run_project_all(pid: int) -> None:
    proj = db.get_project(pid)
    if not proj:
        return
    db.set_project_status(pid, "running")
    for st in proj["stages"]:
        if st.get("status") == "done":
            continue
        ok = await run_project_stage(st)
        if not ok:                     # a stage failed → stop the whole run here
            db.set_project_status(pid, "blocked")
            return
    db.set_project_status(pid, "done")


@app.get("/api/projects/templates")
async def projects_templates() -> dict:
    ags = all_agents()
    tpls = projects_mod.template_list()
    for t in tpls:
        for s in t["stages"]:
            s["agent_detail"] = [
                {"id": a, "name": ags[a].name if a in ags else a,
                 "emoji": ags[a].emoji if a in ags else "•",
                 "desc": (ags[a].desc or ags[a].title) if a in ags else ""}
                for a in s.get("agents", [])]
    return {"templates": tpls}


@app.get("/api/projects")
async def projects_list() -> dict:
    return {"projects": db.list_projects()}


@app.post("/api/projects")
async def projects_create(p: dict) -> dict:
    tpl_id = (p.get("template") or "").strip()
    tpl = projects_mod.TEMPLATES.get(tpl_id)
    if not tpl:
        return {"error": "unknown template"}
    brief = (p.get("brief") or "").strip() or "(ยังไม่ระบุรายละเอียด)"
    name = (p.get("name") or tpl["name"]).strip()
    stages = [{"name": s["name"], "agents": s["agents"], "goal": s["goal"].format(brief=brief)}
              for s in tpl["stages"]]
    pid = db.create_project(name, tpl_id, brief, stages)
    return {"ok": True, "id": pid}


_DESIGN_PROMPT = (
    "คุณคือ CEO ที่ออกแบบทีม AI สำหรับโปรเจกต์ จากชื่อและคำอธิบายของผู้ใช้ "
    "ออกแบบขั้นตอนการทำงาน 3-6 ขั้น ให้ครบลูป (ขั้นสุดท้ายต้องวัดผล/ปิดลูป). "
    "แต่ละขั้นมอบให้เอเจนต์ที่เหมาะ 1-2 ตัว พร้อมบทบาทหน้าที่.\n"
    "ชื่อโปรเจกต์: {name}\nคำอธิบาย/เป้าหมาย: {brief}\n\n"
    "ตอบเป็น JSON อย่างเดียว ตามรูปแบบนี้:\n"
    '{{"stages":[{{"name":"1. ชื่อขั้น","goal":"สิ่งที่ต้องทำในขั้นนี้ (ภาษาไทย กระชับ)",'
    '"agents":[{{"name":"ชื่อบทบาท","emoji":"🔬","role":"หน้าที่สั้นๆ (ภาษาไทย)",'
    '"system":"a concise one-paragraph English system prompt describing this specialist"}}]}}]}}'
)


@app.post("/api/projects/design")
async def projects_design(p: dict) -> dict:
    """AI designs a custom project pipeline + team from name + description."""
    name = (p.get("name") or "").strip() or "โปรเจกต์ใหม่"
    brief = (p.get("brief") or "").strip()
    if not brief:
        return {"error": "ใส่คำอธิบาย/เป้าหมายของโปรเจกต์ก่อน"}
    try:
        raw = await _run_agent("ceo", _DESIGN_PROMPT.format(name=name, brief=brief))
        plan = _parse_json(raw)
        if not plan.get("stages"):
            return {"error": "AI ออกแบบไม่สำเร็จ ลองใส่คำอธิบายให้ชัดขึ้น"}
        return {"ok": True, "plan": plan}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.post("/api/projects/custom")
async def projects_custom(p: dict) -> dict:
    """Create a project from an AI-designed plan, spawning custom agents per role."""
    name = (p.get("name") or "โปรเจกต์ใหม่").strip()
    brief = (p.get("brief") or "").strip()
    plan = p.get("plan") or {}
    stages_in = plan.get("stages") or []
    if not stages_in:
        return {"error": "ไม่มีแผนงาน"}
    agent_map: dict = {}                      # role-name -> created agent id
    stages = []
    for i, s in enumerate(stages_in):
        aids = []
        for a in (s.get("agents") or []):
            nm = (a.get("name") or "Agent").strip()
            if nm not in agent_map:
                slug = re.sub(r"[^a-z0-9]+", "", nm.lower())[:12] or "agent"
                aid = "x_pj_" + slug + secrets.token_hex(2)
                sysp = (a.get("system") or f"You are {nm}, {a.get('role','')}.") + _SPECIALIST_FOOTER
                db.add_custom_agent(aid, nm, (a.get("role") or "")[:60], a.get("emoji") or "🧩",
                                    "#6366f1", "project", sysp)
                agent_map[nm] = aid
            aids.append(agent_map[nm])
        stages.append({"name": s.get("name") or f"{i+1}.", "agents": aids,
                       "goal": s.get("goal") or f"{s.get('name','')} สำหรับโปรเจกต์: {brief}"})
    pid = db.create_project(name, "custom", brief, stages)
    return {"ok": True, "id": pid}


@app.get("/api/projects/{pid}")
async def projects_detail(pid: int) -> dict:
    proj = db.get_project(pid)
    if not proj:
        return {"error": "ไม่พบโปรเจกต์"}
    ags = all_agents()
    for st in proj["stages"]:
        st["agent_names"] = [(ags[a].emoji + " " + ags[a].name) if a in ags else a for a in st.get("agents", [])]
        st["has_visual"] = any(a in ("designer", "content") for a in st.get("agents", []))
    engines = [{"id": 0, "label": "ค่าเริ่มต้น (env / คีย์ของเอเจนต์)"}] + \
              [{"id": k["id"], "label": k["label"]} for k in db.list_api_keys()]
    return {"project": proj, "engines": engines, "summaries": db.list_project_summaries(pid)}


@app.post("/api/projects/stage/run")
async def projects_run_stage(p: dict) -> dict:
    sid = int(p.get("stage_id"))
    if "key_id" in p:                      # switch the API/engine for this stage first
        db.set_stage_engine(sid, int(p.get("key_id") or 0) or None)
    stage = db.get_stage(sid)
    if not stage:
        return {"error": "ไม่พบสเตจ"}
    try:
        ok = await run_project_stage(stage)
        return {"ok": ok, "failed": not ok, "stage": db.get_stage(sid)}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.post("/api/projects/{pid}/run-all")
async def projects_run_all(pid: int) -> dict:
    if not db.get_project(pid):
        return {"error": "ไม่พบโปรเจกต์"}
    import asyncio
    asyncio.create_task(_run_project_all(pid))     # run in background; UI polls detail
    return {"ok": True}


@app.post("/api/projects/{pid}/summary")
async def projects_summary(pid: int) -> dict:
    proj = db.get_project(pid)
    if not proj:
        return {"error": "ไม่พบโปรเจกต์"}
    body = "\n\n".join(f"{s['name']}:\n{(s.get('result') or '(ยังไม่ได้รัน)')[:1500]}" for s in proj["stages"])
    prompt = (f"คุณคือ CEO สรุปผลโปรเจกต์ '{proj['name']}' ให้เจ้าของธุรกิจอ่านเข้าใจง่าย.\n"
              f"เป้าหมาย: {proj['brief']}\n\nผลแต่ละขั้น:\n{body}\n\n"
              "สรุปเป็นภาษาไทย: ภาพรวม, ผลลัพธ์สำคัญแต่ละขั้น, และ next steps ที่ควรทำต่อ")
    try:
        text = await _run_agent("ceo", prompt)
        db.set_project_summary(pid, text)
        return {"ok": True, "summary": text}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.post("/api/projects/{pid}/delete")
async def projects_delete(pid: int) -> dict:
    db.delete_project(pid)
    return {"ok": True}


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
    orch = Orchestrator(lambda aid: client_for_agent(aid, default=conn["default"]), model_for_agent)
    orch.mcp_for = mcp_for_agent

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

            if action == "clear_tasks":
                db.clear_tasks(current_id)
                await send({"type": "history", **db.get_history(current_id)})
                continue

            if action == "delete_task":
                tid = msg.get("id")
                if tid:
                    db.delete_task(int(tid))
                    await send({"type": "history", **db.get_history(current_id)})
                continue

            if action == "translate_tasks":
                import re as _re
                for t in db.list_session_tasks(current_id)[:30]:
                    txt = t.get("task") or ""
                    # only translate ones that are mostly English (no Thai chars but has Latin)
                    if _re.search(r"[฀-๿]", txt) or not _re.search(r"[A-Za-z]", txt):
                        continue
                    try:
                        r = await _client.messages.create(
                            model=get_model(), max_tokens=600, **thinking_kwargs(),
                            system="แปลข้อความงานต่อไปนี้เป็นภาษาไทยที่กระชับ เข้าใจง่าย คงชื่อเฉพาะ/โค้ด/ตัวย่อไว้ ตอบเฉพาะข้อความที่แปลแล้วเท่านั้น",
                            messages=[{"role": "user", "content": txt}])
                        th = next((b.text for b in r.content if b.type == "text"), "").strip()
                        if th:
                            db.set_task_text(t["id"], th)
                    except Exception:
                        pass
                await send({"type": "history", **db.get_history(current_id)})
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
                    await send({"type": "error", "message": _friendly_err(exc)})
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
                    await send({"type": "error", "message": _friendly_err(exc)})
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
                await send({"type": "error", "message": _friendly_err(exc)})
                await send({"type": "done"})
            # Refresh the session list and usage totals after a run.
            await send({"type": "sessions", "sessions": db.list_sessions()})
            await send(usage_payload())
    except WebSocketDisconnect:
        return
