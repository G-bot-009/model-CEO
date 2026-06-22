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
from . import ads
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
    asyncio.create_task(_ads_autopilot_loop())  # daily ads autopilot + kill-switch
    asyncio.create_task(_fb_autoreply_loop())   # auto-reply page comments
    asyncio.create_task(_telegram_loop())       # mirror + AI-assistant over Telegram


LOGIN_USER = os.getenv("LOGIN_USER", "admin")
LOGIN_PASS = os.getenv("LOGIN_PASS", "admin")


def _hash_pw(p: str) -> str:
    return hashlib.sha256(((p or "") + "|goffice-login").encode()).hexdigest()


def _check_login(user: str, pwd: str) -> bool:
    """DB-stored credentials (changed in-app) override the env defaults."""
    s = db.get_settings()
    du, dh = s.get("login_user"), s.get("login_pass_hash")
    if du and dh:
        return (user or "").strip() == du and _hash_pw(pwd) == dh
    return (user or "").strip() == LOGIN_USER and (pwd or "") == LOGIN_PASS

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
    if _check_login(p.get("username") or "", p.get("password") or ""):
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


@app.get("/api/account")
async def account_info() -> dict:
    s = db.get_settings()
    return {"username": s.get("login_user") or LOGIN_USER}


@app.post("/api/account/password")
async def account_password(p: dict, request: Request) -> dict:
    if request.cookies.get("mf_auth") != _auth_token():
        return {"error": "กรุณาเข้าสู่ระบบก่อน"}
    s = db.get_settings()
    cur_user = s.get("login_user") or LOGIN_USER
    if not _check_login(cur_user, p.get("current") or ""):
        return {"error": "รหัสผ่านปัจจุบันไม่ถูกต้อง"}
    new_user = (p.get("username") or cur_user).strip()
    new_pass = p.get("password") or ""
    if len(new_pass) < 4:
        return {"error": "รหัสใหม่สั้นเกินไป (อย่างน้อย 4 ตัวอักษร)"}
    db.set_settings({"login_user": new_user, "login_pass_hash": _hash_pw(new_pass)})
    return {"ok": True, "username": new_user}


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
    # Meta คุมสิทธิ์เพจเข้มงวด (pages_*) — ต้องผ่าน App Review + ยืนยันธุรกิจก่อนถึงจะขอได้
    # ตอนนี้ขอแค่ public_profile เพื่อให้เชื่อม OAuth ผ่าน; เปิดสิทธิ์เต็ม (scope_full) เมื่อผ่านรีวิวแล้ว
    "instagram": {"label": "Instagram", "auth": "https://www.facebook.com/v19.0/dialog/oauth", "token": "https://graph.facebook.com/v19.0/oauth/access_token", "scope": "public_profile", "scope_full": "instagram_basic,instagram_content_publish,pages_show_list"},
    "facebook":  {"label": "Facebook", "auth": "https://www.facebook.com/v19.0/dialog/oauth", "token": "https://graph.facebook.com/v19.0/oauth/access_token", "scope": "public_profile", "scope_full": "public_profile,pages_show_list,pages_manage_posts"},
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
    # หลังผ่าน App Review ตั้ง META_FULL_SCOPE=1 เพื่อขอสิทธิ์เพจเต็ม (scope_full) อัตโนมัติ
    scope = cfg["scope"]
    if cfg.get("scope_full") and os.getenv("META_FULL_SCOPE", "").strip() in ("1", "true", "True"):
        scope = cfg["scope_full"]
    params = {
        "response_type": "code",
        cfg.get("client_param", "client_id"): creds["client_id"],
        "redirect_uri": _redirect_uri(request, platform),
        "scope": scope,
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
            tok, chat = data.get("bot_token", ""), str(data.get("chat_id", "")).strip()
            if not tok or not chat:
                return {"error": "ต้องมี Bot Token และ Chat ID — เชื่อมใหม่"}
            r = await client.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id": chat, "text": text})
            if r.status_code >= 400:
                low = r.text.lower()
                if "can't send messages to the bot" in low or "chat not found" in low:
                    return {"error": ("Chat ID ไม่ถูกต้อง — คุณกรอก ID ของ \"บอท\" เอง บอทส่งหาตัวเองไม่ได้\n"
                                      "วิธีแก้: 1) เปิดแชตกับบอทของคุณใน Telegram แล้วกด Start (พิมพ์ /start)\n"
                                      "2) เอา Chat ID ของ \"คุณ\" (ไม่ใช่ของบอท) — ทักไปที่ @userinfobot จะได้ตัวเลข ID\n"
                                      "3) ถ้าจะส่งเข้ากลุ่ม: เพิ่มบอทเข้ากลุ่มก่อน แล้วใช้ Chat ID ของกลุ่ม (ขึ้นต้นด้วย -)")}
                if "bot was blocked" in low:
                    return {"error": "คุณบล็อกบอทอยู่ — เปิดแชตกับบอทแล้วกดปลดบล็อก/Start ใหม่"}
                if "unauthorized" in low:
                    return {"error": "Bot Token ไม่ถูกต้อง — คัดลอกใหม่จาก @BotFather"}
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


@app.post("/api/mcp/test")
async def mcp_test(p: dict) -> dict:
    """Ping a connected MCP server with a real `initialize` handshake to confirm it works."""
    import httpx
    cid = (p.get("conn_id") or "").strip()
    conn = db.get_mcp_connection(cid)
    if not conn:
        return {"ok": False, "error": "ยังไม่ได้เชื่อม connector นี้"}
    url = (conn.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "ไม่มี URL ของ MCP server"}
    tok = _resolve_secret(_live_mcp_token(conn)) or _live_mcp_token(conn)
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                          "clientInfo": {"name": "G Office", "version": "1.0"}}}
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as cx:
            r = await cx.post(url, json=payload, headers=headers)
    except Exception as exc:
        return {"ok": False, "error": "ต่อ MCP server ไม่ได้: " + _friendly_err(exc)}
    if r.status_code in (200, 202):
        return {"ok": True, "message": "เชื่อมต่อสำเร็จ ✅ MCP server ตอบกลับปกติ"}
    if r.status_code in (401, 403):
        return {"ok": False, "error": f"โทเค็นหมดอายุ/ไม่มีสิทธิ์ (HTTP {r.status_code}) — เชื่อม OAuth หรือวาง token ใหม่"}
    if r.status_code == 404:
        return {"ok": False, "error": "ไม่พบ endpoint (HTTP 404) — เช็ก URL ของ MCP server"}
    return {"ok": False, "error": f"server ตอบ HTTP {r.status_code}"}


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
        # reuse a cached client only if it was registered with the SAME redirect_uri
        if client and client.get("client_id") and (client.get("redirect_uri") or "") == redirect_uri:
            client_id, client_secret = client["client_id"], client.get("client_secret") or ""
        elif meta.get("registration_endpoint"):
            client_id, client_secret = mcp_oauth.register_client(meta["registration_endpoint"], redirect_uri)
            db.save_mcp_oauth_client(conn_id, client_id or "", client_secret or "", redirect_uri)
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
        db.add_trade_log(bot["id"], e["kind"], e["text"], e.get("pnl", 0.0),
                         price=e.get("price"), epoch=int(time.time()))
    # paper mode: realized P&L flows into the demo wallet
    if bot["mode"] == "paper":
        realized = sum(e.get("pnl", 0.0) for e in events)
        if realized:
            db.paper_balance_adjust(realized)
    new_state["last_price"] = price          # for live unrealized P&L / equity
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
    return {"bots": db.list_bots(), "default_config": trading.DEFAULT_CONFIG,
            "paper_balance": db.paper_balance_get()}


def _bot_unrealized(b: dict) -> float:
    """Floating P&L of a bot's open paper position at its last seen price."""
    st = b.get("state") or {}
    pos = st.get("pos")
    lp = st.get("last_price")
    if not pos or not lp:
        return 0.0
    sign = 1 if (pos.get("side") == "long") else -1
    rem_qty = sum(l["qty"] for l in pos.get("legs", []) if not l.get("filled"))
    return rem_qty * (float(lp) - pos["entry"]) * sign


@app.get("/api/trading/wallet")
async def trading_wallet() -> dict:
    today = trading._today()
    pnl_today = 0.0
    unrealized = 0.0
    holding = False
    for b in db.list_bots():
        if b["mode"] != "paper":
            continue
        st = b.get("state") or {}
        if st.get("day") == today:
            pnl_today += float(st.get("daily_pnl") or 0.0)
        u = _bot_unrealized(b)
        unrealized += u
        if st.get("pos"):
            holding = True
    bal = db.paper_balance_get()
    return {"balance": bal, "pnl_today": round(pnl_today, 2),
            "unrealized": round(unrealized, 2), "equity": round(bal + unrealized, 2),
            "holding": holding}


@app.post("/api/trading/wallet")
async def trading_wallet_set(p: dict) -> dict:
    action = (p.get("action") or "set").lower()
    try:
        amount = float(p.get("amount") or 0)
    except Exception:
        return {"error": "จำนวนเงินไม่ถูกต้อง"}
    if action == "reset":
        bal = db.paper_balance_set(db.PAPER_DEFAULT)
    elif action == "add":
        bal = db.paper_balance_adjust(abs(amount))
    elif action == "sub":
        bal = db.paper_balance_adjust(-abs(amount))
    else:  # set
        bal = db.paper_balance_set(amount)
    return {"ok": True, "balance": bal}


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


@app.get("/api/trading/klines")
async def trading_klines(exchange: str = "binance", symbol: str = "BTCUSDT",
                         interval: str = "5") -> dict:
    import asyncio
    try:
        candles = await asyncio.to_thread(trading.fetch_klines, exchange, symbol, interval, 200)
    except Exception as exc:
        return {"error": _friendly_err(exc), "candles": []}
    return {"candles": candles}


@app.get("/api/trading/bot/{bot_id}/markers")
async def trading_bot_markers(bot_id: int) -> dict:
    """Entry/exit markers (time, price, kind, reason) for the chart + current position lines."""
    bot = db.get_bot(bot_id)
    if not bot:
        return {"error": "ไม่พบบอท", "markers": []}
    marks = db.trade_markers(bot_id)
    st = bot.get("state") or {}
    pos = st.get("pos")
    lines = []
    position = None
    if pos:
        lines.append({"price": pos["entry"], "color": "#3b82f6", "title": "จุดเข้า"})
        lines.append({"price": pos["sl"], "color": "#ef4444", "title": "SL"})
        for leg in pos.get("legs", []):
            lines.append({"price": leg["tp"], "color": "#22c55e",
                          "title": f"TP{leg['i']}" + (" ✓" if leg.get("filled") else "")})
        unreal = _bot_unrealized(bot)
        position = {"side": pos.get("side"), "entry": pos["entry"], "sl": pos["sl"],
                    "size_usd": bot["config"].get("size_usd"),
                    "last_price": st.get("last_price"),
                    "unrealized": round(unreal, 2)}
    # global demo wallet (so the figure matches the wallet panel)
    bal = db.paper_balance_get()
    total_unreal = sum(_bot_unrealized(b) for b in db.list_bots() if b["mode"] == "paper")
    wallet = {"balance": bal, "equity": round(bal + total_unreal, 2),
              "unrealized": round(total_unreal, 2)} if bot["mode"] == "paper" else None
    return {"markers": marks, "lines": lines, "position": position, "wallet": wallet,
            "side": bot["config"].get("side"), "mode": bot["mode"],
            "exchange": bot["exchange"], "symbol": bot["symbol"]}


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


@app.post("/api/trading/bot/{bot_id}/chat")
async def trading_bot_chat(bot_id: int, p: dict) -> dict:
    """Chat about a bot and optionally adjust its settings in natural Thai."""
    bot = db.get_bot(bot_id)
    if not bot:
        return {"error": "ไม่พบบอท"}
    msg = (p.get("message") or "").strip()
    if not msg:
        return {"error": "พิมพ์ข้อความก่อน"}
    st = bot.get("state") or {}
    pos = st.get("pos")
    context = {
        "ชื่อ": bot["name"], "ตลาด": bot["exchange"], "คู่เทรด": bot["symbol"],
        "โหมด": bot["mode"], "config": bot["config"],
        "กำลังถือไม้": bool(pos),
        "PnL_วันนี้": round(float(st.get("daily_pnl") or 0), 2),
        "ราคาล่าสุด": st.get("last_price"),
    }
    prompt = (
        "คุณเป็นโค้ชบอทเทรดที่พูดไทย เป็นกันเอง ตรงไปตรงมา ไม่การันตีกำไร และไม่ทำนายตลาด.\n"
        "ผู้ใช้จะคุย/ถาม หรือสั่งปรับค่าบอท. ปรับได้เฉพาะคีย์เหล่านี้: "
        "size_usd, tp_legs_pct(ลิสต์%), sl_pct, move_sl_to_be_after_leg, "
        "daily_target_usd, daily_loss_limit_usd, side(long/short).\n"
        f"ข้อมูลบอทตอนนี้: {json.dumps(context, ensure_ascii=False)}\n"
        f"ผู้ใช้พูดว่า: {msg}\n\n"
        "ตอบ JSON อย่างเดียว: {\"reply\":\"คำตอบ/คำแนะนำสั้นๆ ภาษาไทย\","
        "\"config\":{เฉพาะคีย์ที่ผู้ใช้ขอให้เปลี่ยน หรือ {} ถ้าไม่ต้องเปลี่ยน}}"
    )
    try:
        resp = await client_for_agent("trader").messages.create(
            model=model_for_agent("trader"), max_tokens=700,
            **thinking_kwargs(model_for_agent("trader")),
            messages=[{"role": "user", "content": prompt}])
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    reply = (data.get("reply") or "").strip() or "รับทราบครับ"
    patch = data.get("config") or {}
    applied = {}
    if isinstance(patch, dict) and patch:
        merged = dict(bot["config"]); merged.update(patch)
        newcfg = trading.normalize_config(merged)
        # only report keys that actually changed
        for k, v in newcfg.items():
            if bot["config"].get(k) != v and k in patch:
                applied[k] = v
        if applied:
            db.update_bot(bot["id"], bot["name"], bot["exchange"], bot["symbol"],
                          bot["mode"], newcfg)
    return {"ok": True, "reply": reply, "applied": applied}


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

    # art direction: real image via the chosen provider (BYO key) else SVG fallback.
    # media_wanted = ผู้ใช้ตั้งให้มีรูป; media_err = สาเหตุถ้าสร้างไม่สำเร็จ (ห้ามแสดงว่าสำเร็จถ้ารูปพัง)
    media_kind, media_ref, media_err = "none", "", ""
    media_wanted = rules["media_per_day"] != 0
    img = _media_cfg("image")
    if media_wanted:
        if img["key"]:
            try:
                im = await __import__("asyncio").to_thread(
                    media.generate_image, img["provider"], img["model"], img["key"], img_prompt)
                b64 = (im.get("b64") or "").strip()
                if len(b64) < 100:                      # คืนมาว่าง/ไม่สมบูรณ์
                    raise RuntimeError("ภาพที่ได้ว่างเปล่า")
                media_kind, media_ref = "image", f"data:{im['mime']};base64,{b64}"
            except Exception as exc:
                media_err = f"สร้างรูปด้วย {img['provider']} ไม่สำเร็จ: {_friendly_err(exc)}"
                media_kind = "none"
        if media_kind == "none":                        # SVG สำรอง
            try:
                w, h = IMG_SIZES.get("1:1", (600, 600))
                svg = _extract_svg(await _claude_text("designer", model,
                        f"Create a single self-contained <svg width='{w}' height='{h}'> illustration for: {img_prompt}. Return only the SVG.", 4000))
                if svg and "<svg" in svg.lower() and len(svg) > 120:
                    media_kind, media_ref, media_err = "svg", svg, ""
                else:
                    media_err = media_err or "สร้างภาพสำรอง (SVG) ไม่สำเร็จ"
            except Exception as exc:
                media_err = media_err or f"สร้างภาพสำรองไม่สำเร็จ: {_friendly_err(exc)}"

    platforms = [n for n, st in db.list_connectors().items() if st == "connected" and n in _SENDABLE]
    incomplete = media_wanted and media_kind == "none"      # อยากได้รูปแต่ไม่มีรูปใช้ได้
    if not safe:
        status = "blocked"
    elif incomplete:
        status = "draft"                                    # ไม่ auto-queue งานที่รูปพัง
    elif rules["post_mode"] == "auto":
        status = "queued"
    else:
        status = "draft"
    pid = db.create_planned("social", topic, caption, media_kind, media_ref, platforms, status, db._now())
    if not safe:
        db.set_planned_status(pid, "blocked", result=f"ไม่ผ่านการกรอง: {safe_reason}")
    elif incomplete:
        db.set_planned_status(pid, "draft", result=f"⚠️ รูปไม่สมบูรณ์ — {media_err or 'สร้างรูปไม่สำเร็จ'} (ยังไม่ควรโพสต์)")
    db.add_decision("content_factory", f"{status}: {topic[:50]}")
    post = db.get_planned(pid)
    if isinstance(post, dict):
        post["media_error"] = bool(incomplete)
    return post


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


@app.get("/api/tts")
async def tts(text: str = "", lang: str = "th"):
    """Server-side text-to-speech (returns MP3). Reliable audio playback on mobile.
    Uses the public Google Translate TTS endpoint; chunks long text to stay within limits."""
    import httpx
    import urllib.parse
    from fastapi import Response
    txt = (text or "").strip()[:900]
    if not txt:
        return Response(b"", media_type="audio/mpeg")
    # split into <=180-char chunks on word boundaries
    chunks, cur = [], ""
    for w in txt.split(" "):
        if len(cur) + len(w) + 1 > 180:
            if cur:
                chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur)
    audio = b""
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36"}
    try:
        async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
            for ch in chunks[:6]:
                q = urllib.parse.urlencode({"ie": "UTF-8", "tl": lang, "client": "tw-ob", "q": ch})
                r = await client.get("https://translate.google.com/translate_tts?" + q)
                if r.status_code < 400 and r.content:
                    audio += r.content
    except Exception:
        pass
    if not audio:
        # signal the client to fall back to the browser's built-in voice
        return Response(b"", media_type="audio/mpeg", status_code=204)
    return Response(audio, media_type="audio/mpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


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


@app.post("/api/content/media-key/clear")
async def content_media_key_clear(p: dict) -> dict:
    kind = p.get("kind")
    if kind in ("image", "video", "voice"):
        db.set_settings({f"media_{kind}_key": ""})
    return {"ok": True}


# ============================== Music AI ====================================
@app.get("/api/music/board")
async def music_board() -> dict:
    c = _media_cfg("music")
    return {
        "providers": media.PROVIDERS["music"],
        "selection": {"provider": c["provider"], "model": c["model"],
                      "key": (c["key"][:4] + "…") if c["key"] else "", "has": bool(c["key"])},
        "tracks": db.music_list(),
    }


@app.post("/api/music/key")
async def music_key_set(p: dict) -> dict:
    upd = {}
    if p.get("provider"):
        upd["media_music_provider"] = p["provider"].strip()
    if p.get("model") is not None:
        upd["media_music_model"] = (p.get("model") or "").strip()
    if p.get("key"):
        upd["media_music_key"] = p["key"].strip()
    if upd:
        db.set_settings(upd)
    return {"ok": True}


@app.post("/api/music/key/clear")
async def music_key_clear(p: dict) -> dict:
    db.set_settings({"media_music_key": ""})
    return {"ok": True}


@app.post("/api/music/generate")
async def music_generate(p: dict) -> dict:
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        return {"error": "ใส่คำอธิบายเพลง/อารมณ์ก่อน (เช่น 'ดนตรีลูกทุ่งสนุก จังหวะเร็ว')"}
    cfg = _media_cfg("music")
    if not cfg["key"]:
        return {"error": "ยังไม่ได้ใส่คีย์ Music AI — กด 🔑 ใส่คีย์ ก่อน"}
    try:
        dur = int(p.get("duration") or 20)
    except (TypeError, ValueError):
        dur = 20
    try:
        m = await __import__("asyncio").to_thread(
            media.generate_music, cfg["provider"], cfg["model"], cfg["key"], prompt, dur)
        b64 = (m.get("b64") or "").strip()
        if len(b64) < 100:
            return {"error": "ไฟล์เสียงที่ได้ว่างเปล่า — ลองใหม่หรือเปลี่ยน provider"}
        ref = f"data:{m['mime']};base64,{b64}"
        tid = db.music_add(prompt, cfg["provider"], cfg["model"], m["mime"], ref)
        return {"ok": True, "track": next((t for t in db.music_list() if t["id"] == tid), None)}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.post("/api/music/delete")
async def music_delete(p: dict) -> dict:
    db.music_delete(int(p.get("id")))
    return {"ok": True}


# ============================== Image AI ====================================
@app.get("/api/imageai/board")
async def imageai_board() -> dict:
    c = _media_cfg("image")
    return {
        "providers": media.PROVIDERS["image"],
        "selection": {"provider": c["provider"], "model": c["model"],
                      "key": (c["key"][:4] + "…") if c["key"] else "", "has": bool(c["key"])},
        "images": db.image_list(),
    }


@app.post("/api/imageai/key")
async def imageai_key_set(p: dict) -> dict:
    upd = {}
    if p.get("provider"):
        upd["media_image_provider"] = p["provider"].strip()
    if p.get("model") is not None:
        upd["media_image_model"] = (p.get("model") or "").strip()
    if p.get("key"):
        upd["media_image_key"] = p["key"].strip()
    if upd:
        db.set_settings(upd)
    return {"ok": True}


@app.post("/api/imageai/key/clear")
async def imageai_key_clear(p: dict) -> dict:
    db.set_settings({"media_image_key": ""})
    return {"ok": True}


@app.post("/api/imageai/generate")
async def imageai_generate(p: dict) -> dict:
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        return {"error": "ใส่คำอธิบายภาพก่อน (เช่น 'โลโก้ร้านกาแฟ มินิมอล โทนน้ำตาล')"}
    cfg = _media_cfg("image")
    if not cfg["key"]:
        return {"error": "ยังไม่ได้ใส่คีย์ Image AI — กด 🔑 ใส่คีย์ ก่อน"}
    try:
        im = await __import__("asyncio").to_thread(
            media.generate_image, cfg["provider"], cfg["model"], cfg["key"], prompt)
        b64 = (im.get("b64") or "").strip()
        if len(b64) < 100:
            return {"error": "ภาพที่ได้ว่างเปล่า — ลองใหม่หรือเปลี่ยน provider"}
        ref = f"data:{im['mime']};base64,{b64}"
        tid = db.image_add(prompt, cfg["provider"], cfg["model"], im["mime"], ref)
        return {"ok": True, "image": next((t for t in db.image_list() if t["id"] == tid), None)}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.post("/api/imageai/delete")
async def imageai_delete(p: dict) -> dict:
    db.image_delete(int(p.get("id")))
    return {"ok": True}


# ================================ VEO AI (video) ============================
@app.get("/api/veo/board")
async def veo_board() -> dict:
    c = _media_cfg("video")
    return {
        "providers": media.PROVIDERS["video"],
        "selection": {"provider": c["provider"], "model": c["model"],
                      "key": (c["key"][:4] + "…") if c["key"] else "", "has": bool(c["key"])},
        "videos": db.video_list(),
    }


@app.post("/api/veo/key")
async def veo_key_set(p: dict) -> dict:
    upd = {}
    if p.get("provider"):
        upd["media_video_provider"] = p["provider"].strip()
    if p.get("model") is not None:
        upd["media_video_model"] = (p.get("model") or "").strip()
    if p.get("key"):
        upd["media_video_key"] = p["key"].strip()
    if upd:
        db.set_settings(upd)
    return {"ok": True}


@app.post("/api/veo/key/clear")
async def veo_key_clear(p: dict) -> dict:
    db.set_settings({"media_video_key": ""})
    return {"ok": True}


@app.post("/api/veo/generate")
async def veo_generate(p: dict) -> dict:
    import asyncio
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        return {"error": "ใส่คำอธิบายวิดีโอก่อน (เช่น 'โดรนบินเหนือทะเลตอนพระอาทิตย์ตก ภาพยนตร์')"}
    cfg = _media_cfg("video")
    if not cfg["key"]:
        return {"error": "ยังไม่ได้ใส่คีย์ VEO AI — กด 🔑 ใส่คีย์ ก่อน"}
    try:
        job = await asyncio.to_thread(media.generate_video, cfg["provider"], cfg["model"], cfg["key"], prompt)
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    tid = db.video_add(prompt, cfg["provider"], cfg["model"], job.get("task_id", ""))
    return {"ok": True, "video": db.video_get(tid)}


@app.post("/api/veo/poll")
async def veo_poll(p: dict) -> dict:
    import asyncio
    v = db.video_get(int(p.get("id")))
    if not v:
        return {"error": "ไม่พบงานวิดีโอ"}
    if v["status"] != "pending":
        return {"ok": True, "video": v}
    cfg = _media_cfg("video")
    try:
        res = await asyncio.to_thread(media.poll_video, v["provider"], v["model"], cfg["key"], v["task_id"])
    except Exception as exc:
        return {"status": "pending", "warn": _friendly_err(exc), "video": v}
    if res.get("status") == "done":
        b64 = (res.get("b64") or "").strip()
        if len(b64) < 100:
            db.video_set_error(v["id"]); return {"ok": True, "video": db.video_get(v["id"])}
        ref = f"data:{res.get('mime','video/mp4')};base64,{b64}"
        db.video_set_done(v["id"], res.get("mime", "video/mp4"), ref)
    elif res.get("status") == "error":
        db.video_set_error(v["id"])
    return {"ok": True, "video": db.video_get(v["id"])}


@app.post("/api/veo/delete")
async def veo_delete(p: dict) -> dict:
    db.video_delete(int(p.get("id")))
    return {"ok": True}


# ============================== WordPress writer ============================
def _wp_cfg() -> dict:
    row = db.get_connector("WordPress")
    if not row or row.get("status") != "connected":
        return {}
    try:
        return json.loads(row.get("config") or "{}")
    except Exception:
        return {}


_WP_ARTICLE_PROMPT = (
    "เขียนบทความ SEO ภาษาไทยที่อ่านลื่น น่าเชื่อถือ สำหรับเผยแพร่บนเว็บ WordPress "
    "หัวข้อ/บริบท: {topic}\n\n"
    "ตอบกลับเป็น JSON อย่างเดียว: {{\"title\":\"พาดหัวที่ดึงดูด\","
    "\"html\":\"เนื้อหาเต็มเป็น HTML ใช้ <h2>,<h3>,<p>,<ul>,<li>,<strong> "
    "ความยาว 500-900 คำ มีบทนำ หัวข้อย่อย และสรุปปิดท้าย\"}}"
)


@app.post("/api/wordpress/write")
async def wordpress_write(p: dict) -> dict:
    """Claude เขียนบทความเต็มแล้วโพสต์ขึ้น WordPress จริง (เลือกร่าง/เผยแพร่)."""
    import httpx
    cfg = _wp_cfg()
    if not (cfg.get("url") and cfg.get("user") and cfg.get("pass")):
        return {"error": "ยังไม่ได้เชื่อม WordPress — กรอก Site URL + ชื่อผู้ใช้ + Application Password ก่อน"}
    topic = (p.get("topic") or "").strip()
    if not topic:
        return {"error": "ใส่หัวข้อ/เรื่องที่อยากให้เขียนก่อน"}
    status = "publish" if p.get("publish") else "draft"
    model = model_for_agent("content")
    try:
        raw = await _claude_text("content", model, _WP_ARTICLE_PROMPT.format(topic=topic), 3500)
    except Exception as exc:
        return {"error": "เขียนบทความไม่สำเร็จ: " + _friendly_err(exc)}
    data = _parse_json(raw)
    title = (data.get("title") or topic).strip()
    html = (data.get("html") or raw).strip()
    url = cfg["url"].rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{url}/wp-json/wp/v2/posts",
                             auth=(cfg["user"], cfg["pass"]),
                             json={"title": title, "content": html, "status": status})
    except Exception as exc:
        return {"error": "ต่อ WordPress ไม่ได้: " + _friendly_err(exc)}
    if r.status_code in (200, 201):
        d = r.json()
        db.add_decision("content", f"WordPress {status}: {title[:50]}")
        return {"ok": True, "title": title, "status": status,
                "link": d.get("link", ""), "edit": f"{url}/wp-admin/post.php?post={d.get('id')}&action=edit"}
    if r.status_code in (401, 403):
        return {"error": "สิทธิ์ไม่ผ่าน (401/403) — เช็กชื่อผู้ใช้ + Application Password"}
    return {"error": f"WordPress ตอบ {r.status_code}: {r.text[:200]}"}


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


_PROJECT_STYLE = (
    "\n\n---\nสำคัญมาก: ตอบเป็น**ภาษาไทยที่อ่านเข้าใจง่าย**สำหรับเจ้าของธุรกิจที่ไม่ใช่สายเทคนิค "
    "เขียนเป็นคำอธิบายเหมือนคนคุยกัน ใช้หัวข้อสั้น ๆ และบูลเล็ต (-) "
    "**ห้ามตอบเป็น JSON, โค้ด หรือ code block เด็ดขาด** "
    "เน้นบอกว่า ‘ทำอะไร’ ‘ได้ผลอะไร’ และ ‘ขั้นต่อไปควรทำอะไร’ ให้กระชับ ชัดเจน นำไปใช้ได้จริง"
)


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
            out = await _run_agent(aid, stage["goal"] + _PROJECT_STYLE, client)
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


# ============================== Bio Link Page ===============================
_BIO_RESERVED = {"", "api", "oauth", "login", "logout", "ws", "l", "p", "health",
                 "favicon.ico", "robots.txt", "static", "assets", "privacy", "terms"}


def _ip_hash(request: Request) -> str:
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "") or "")
    return hashlib.sha256((ip + "|goffice-bio").encode()).hexdigest()[:16]


def _esc(s) -> str:
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _norm_slug(s: str) -> str:
    return (s or "").strip().strip("/").strip()


@app.get("/api/bio/pages")
async def bio_pages() -> dict:
    pages = db.bio_list_pages()
    for p in pages:
        a = db.bio_analytics(p["id"], 90)
        p["views"] = a["views"]; p["clicks"] = a["clicks"]
        p["link_count"] = len(db.bio_list_links(p["id"]))
    return {"pages": pages, "base": PUBLIC_BASE_URL}


@app.post("/api/bio/page")
async def bio_page_save(p: dict) -> dict:
    slug = _norm_slug(p.get("slug") or "")
    if not slug or slug.lower() in _BIO_RESERVED or "/" in slug:
        return {"error": "slug ไม่ถูกต้อง (ห้ามว่าง/ซ้ำคำสงวน/มี /)"}
    pid = p.get("id")
    if db.bio_slug_taken(slug, int(pid) if pid else None):
        return {"error": "slug นี้ถูกใช้แล้ว ลองตั้งใหม่"}
    name = (p.get("display_name") or slug).strip()
    avatar = (p.get("avatar_url") or "").strip()
    bio = (p.get("bio") or "").strip()
    theme = "dark" if p.get("theme") == "dark" else "light"
    if pid:
        db.bio_update_page(int(pid), slug, name, avatar, bio, theme)
        return {"ok": True, "id": int(pid)}
    nid = db.bio_create_page(slug, name, avatar, bio, theme)
    return {"ok": True, "id": nid}


@app.get("/api/bio/page/{pid}")
async def bio_page_detail(pid: int, range: str = "90d") -> dict:
    page = db.bio_get_page(pid)
    if not page:
        return {"error": "ไม่พบเพจ"}
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 90)
    return {"page": page, "links": db.bio_list_links(pid), "analytics": db.bio_analytics(pid, days)}


@app.post("/api/bio/page/delete")
async def bio_page_delete(p: dict) -> dict:
    db.bio_delete_page(int(p.get("id")))
    return {"ok": True}


@app.post("/api/bio/link")
async def bio_link_save(p: dict) -> dict:
    label = (p.get("label") or "").strip()
    url = (p.get("url") or "").strip()
    if not label or not url:
        return {"error": "ใส่ชื่อปุ่มและ URL"}
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    ord_ = int(p.get("ord") or 0)
    active = 0 if p.get("is_active") is False else 1
    lid = p.get("id")
    if lid:
        db.bio_update_link(int(lid), label, url, ord_, active)
        return {"ok": True, "id": int(lid)}
    nid = db.bio_add_link(int(p.get("page_id")), label, url, ord_, active)
    return {"ok": True, "id": nid}


@app.post("/api/bio/link/delete")
async def bio_link_delete(p: dict) -> dict:
    db.bio_delete_link(int(p.get("id")))
    return {"ok": True}


@app.get("/api/bio/analytics")
async def bio_analytics_api(page_id: int, range: str = "90d") -> dict:
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 90)
    return db.bio_analytics(page_id, days)


# ============================== Ads Manager =================================
def _ads_cfg(name: str) -> dict:
    row = db.get_connector(name)
    if not row or row.get("status") != "connected":
        return {}
    try:
        return json.loads(row.get("config") or "{}")
    except Exception:
        return {}


def _ads_mode() -> str:
    return "auto" if db.get_settings().get("ads_mode") == "auto" else "review"


def _ads_meta_accounts() -> list:
    """Connected Meta ad accounts (up to 10). Falls back to the single legacy
    'Meta Ads' connector if no multi-accounts have been added yet."""
    accts = db.ads_list_accounts("meta")
    if accts:
        return accts
    cfg = _ads_cfg("Meta Ads")
    if cfg.get("token"):
        return [{"id": None, "label": "บัญชีหลัก", "token": cfg["token"], "ad_account_id": cfg.get("ad_account_id", "")}]
    return []


_ADS_PROMPT = (
    "คุณคือผู้จัดการโฆษณา วิเคราะห์ผลแคมเปญแล้วแนะนำการตัดสินใจรายแคมเปญ "
    "(ไม่การันตีผล ใช้ดุลพินิจจากตัวเลข).\n"
    "เกณฑ์: ROAS ต่ำ/ใช้งบแต่ไม่มีคอนเวอร์ชัน → pause · ROAS ดี/คุ้ม → scale (เพิ่มงบ ~20-50%) · "
    "ยังเก็บข้อมูล/ก้ำกึ่ง → keep.\n\n"
    "ข้อมูลแคมเปญ (JSON):\n{data}\n\n"
    "ตอบ JSON อย่างเดียว: {{\"recos\":[{{\"campaign_id\":\"...\",\"action\":\"pause|scale|keep\","
    "\"reason\":\"เหตุผลสั้นๆ ภาษาไทย\",\"suggested_budget\":ตัวเลขงบ/วันที่แนะนำ(บาท)}}]}}"
)


async def run_ads_review(only_account_id=None) -> list:
    """Pull campaign metrics from connected Meta accounts (all, or one when
    only_account_id is given), let Claude recommend per campaign, store as
    pending recos tagged by account."""
    import asyncio
    campaigns, errors = [], []
    metas = _ads_meta_accounts()
    if only_account_id is not None:
        metas = [a for a in metas if str(a.get("id")) == str(only_account_id)]
    for acc in metas:
        try:
            rows = await asyncio.to_thread(ads.meta_campaigns, acc["token"], acc.get("ad_account_id", ""))
        except Exception as exc:
            errors.append(f"{acc['label']}: {type(exc).__name__}")
            continue
        for r in rows:
            r["_platform"] = "meta"; r["_acc_id"] = acc["id"]; r["_acc_label"] = acc["label"]
            campaigns.append(r)
    gcfg = _ads_cfg("Google Ads")
    if only_account_id is None and gcfg.get("developer_token") and gcfg.get("customer_id"):
        try:
            for r in await asyncio.to_thread(ads.google_campaigns, gcfg):
                r["_platform"] = "google"; r["_acc_id"] = None; r["_acc_label"] = "Google Ads"
                campaigns.append(r)
        except Exception as exc:
            errors.append(f"Google Ads: {type(exc).__name__}")
    if not campaigns:
        raise RuntimeError("ยังไม่มีข้อมูลแคมเปญ — เพิ่มบัญชี Facebook (token + Ad Account ID) ก่อน"
                           + (f" · ปัญหา: {', '.join(errors)}" if errors else ""))
    model = model_for_agent("analyst")
    data = [dict(account=c["_acc_label"], **{k: v for k, v in c.items() if not k.startswith("_")}) for c in campaigns]
    raw = await _claude_text("analyst", model, _ADS_PROMPT.format(data=json.dumps(data, ensure_ascii=False)), 2000)
    recos = {r.get("campaign_id"): r for r in _parse_json(raw).get("recos", [])}
    db.ads_clear_pending(only_account_id)
    for c in campaigns:
        rc = recos.get(c["id"], {})
        action = rc.get("action") if rc.get("action") in ("pause", "scale", "keep") else "keep"
        clean = {k: v for k, v in c.items() if not k.startswith("_")}
        db.ads_add_reco("meta", c["id"], c["name"], clean, action,
                        rc.get("reason", ""), float(rc.get("suggested_budget") or c.get("daily_budget") or 0),
                        account_id=c["_acc_id"], account_label=c["_acc_label"])
    return db.ads_list_reco()


def _apply_reco(reco: dict) -> str:
    if reco["platform"] == "meta":
        token = ""
        if reco.get("account_id"):
            acc = db.ads_get_account(reco["account_id"])
            token = acc["token"] if acc else ""
        if not token:
            token = _ads_cfg("Meta Ads").get("token", "")
        meta = {"token": token}
        if not meta.get("token"):
            raise RuntimeError("ไม่พบ token ของบัญชี Facebook นี้")
        if reco["action"] == "pause":
            ads.meta_pause(meta["token"], reco["campaign_id"]); return "หยุดแคมเปญแล้ว"
        if reco["action"] == "scale":
            ads.meta_set_budget(meta["token"], reco["campaign_id"], reco["suggested_budget"]); return f"เพิ่มงบเป็น {reco['suggested_budget']}/วัน"
        return "คงไว้ (ไม่เปลี่ยน)"
    if reco["platform"] == "google":
        gcfg = _ads_cfg("Google Ads")
        if not (gcfg.get("developer_token") and gcfg.get("customer_id")):
            raise RuntimeError("ยังไม่ได้ตั้งค่า Google Ads")
        if reco["action"] == "pause":
            ads.google_pause(gcfg, reco["campaign_id"]); return "หยุดแคมเปญ Google แล้ว"
        if reco["action"] == "scale":
            res = (reco.get("metrics") or {}).get("budget_resource", "")
            ads.google_set_budget(gcfg, res, reco["suggested_budget"]); return f"เพิ่มงบเป็น {reco['suggested_budget']}/วัน"
        return "คงไว้ (ไม่เปลี่ยน)"
    raise RuntimeError("แพลตฟอร์มนี้ยังใช้กับการสั่งจริงไม่ได้")


@app.get("/api/ads/status")
async def ads_status() -> dict:
    _g = _ads_cfg("Google Ads")
    return {"meta": len(_ads_meta_accounts()) > 0, "meta_count": len(_ads_meta_accounts()),
            "google": bool(_g.get("developer_token") and _g.get("customer_id")),
            "mode": _ads_mode()}


# --- One-click Facebook Ads OAuth (no manual token / no System User) ---------
def _fb_ads_creds() -> tuple:
    """App ID/Secret for the ads OAuth — from settings (UI) else env (operator)."""
    s = db.get_settings()
    app_id = (s.get("fb_ads_app_id") or os.getenv("FB_ADS_APP_ID", "")).strip()
    app_secret = (s.get("fb_ads_app_secret") or os.getenv("FB_ADS_APP_SECRET", "")).strip()
    return app_id, app_secret


@app.post("/api/ads/fb-oauth-creds")
async def ads_fb_oauth_creds(p: dict) -> dict:
    upd = {}
    if p.get("app_id"):
        upd["fb_ads_app_id"] = p["app_id"].strip()
    if p.get("app_secret"):
        upd["fb_ads_app_secret"] = p["app_secret"].strip()
    if upd:
        db.set_settings(upd)
    return {"ok": True}


@app.get("/api/ads/fb-appid")
async def ads_fb_appid() -> dict:
    """App ID for the JS SDK login button (App ID is not secret)."""
    app_id, _ = _fb_ads_creds()
    return {"app_id": app_id}


@app.post("/api/ads/fb-sdk-connect")
async def ads_fb_sdk_connect(p: dict) -> dict:
    """Receive a token from the in-page FB JS SDK login → fetch ad accounts → add/pick."""
    import httpx
    token = (p.get("token") or "").strip()
    slot = str(p.get("slot") or "")
    if not token:
        return {"error": "ไม่ได้รับ token จาก Facebook"}
    app_id, app_secret = _fb_ads_creds()
    long_tok = token
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            if app_id and app_secret:        # exchange to long-lived if we have the secret
                r = await c.get("https://graph.facebook.com/v19.0/oauth/access_token",
                                params={"grant_type": "fb_exchange_token", "client_id": app_id,
                                        "client_secret": app_secret, "fb_exchange_token": token})
                long_tok = r.json().get("access_token") or token
            r3 = await c.get("https://graph.facebook.com/v19.0/me/adaccounts",
                             params={"fields": "name,account_id", "access_token": long_tok, "limit": 50})
            accts = r3.json().get("data", [])
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    if not accts:
        return {"error": "บัญชี Facebook นี้ไม่มีสิทธิ์ ads / ไม่มีบัญชีโฆษณา"}
    existing = {a["ad_account_id"] for a in db.ads_list_accounts("meta")}
    fresh = [a for a in accts if a.get("account_id") and a.get("account_id") not in existing]
    if not fresh:
        return {"error": "บัญชีโฆษณาทั้งหมดของบัญชีนี้เชื่อมไว้แล้ว"}
    if len(fresh) == 1:
        err = _fb_ads_assign(slot, long_tok, fresh[0])
        return {"error": err} if err else {"ok": True, "added": fresh[0].get("name")}
    pid = secrets.token_urlsafe(12)
    _FB_ADS_PICK[pid] = {"slot": slot, "token": long_tok, "accounts": fresh, "ts": time.time()}
    return {"ok": True, "pick_id": pid,
            "accounts": [{"name": a.get("name"), "account_id": a.get("account_id")} for a in fresh]}


@app.post("/api/ads/fb-sdk-pick")
async def ads_fb_sdk_pick(p: dict) -> dict:
    d = _FB_ADS_PICK.pop(p.get("pick_id") or "", None)
    if not d:
        return {"error": "หมดเวลาเลือก — ล็อกอินใหม่"}
    a = next((x for x in d["accounts"] if str(x.get("account_id")) == str(p.get("account_id"))), None)
    if not a:
        return {"error": "ไม่พบบัญชีที่เลือก"}
    err = _fb_ads_assign(d["slot"], d["token"], a)
    return {"error": err} if err else {"ok": True, "added": a.get("name")}


@app.get("/api/ads/fb-oauth-status")
async def ads_fb_oauth_status(request: Request) -> dict:
    app_id, app_secret = _fb_ads_creds()
    return {"has": bool(app_id and app_secret),
            "redirect_uri": _public_base(request) + "/oauth-ads/facebook/callback"}


@app.get("/oauth-ads/facebook/start")
async def fb_ads_oauth_start(request: Request, slot: str = "", kind: str = "ads"):
    """Implicit flow (response_type=token): token returns in the URL fragment so
    NO App Secret is needed — only App ID. kind=ads → ad accounts · kind=pages → page management."""
    from urllib.parse import urlencode
    app_id, _ = _fb_ads_creds()
    if not app_id:
        return HTMLResponse("<h3>ยังไม่ได้ใส่ App ID — ปิดหน้าต่างแล้วกรอกก่อน</h3>", status_code=400)
    if kind == "pages":
        scope = "pages_show_list,pages_manage_posts,pages_read_engagement,pages_manage_engagement,pages_manage_metadata"
        state = "pages|x"
    else:
        scope = "ads_management,ads_read"
        state = (slot or "") + "|x"
    params = {"response_type": "token", "client_id": app_id,
              "redirect_uri": _public_base(request) + "/oauth-ads/facebook/callback",
              "scope": scope, "state": state, "auth_type": "reauthenticate"}
    return RedirectResponse("https://www.facebook.com/v19.0/dialog/oauth?" + urlencode(params))


_FB_ADS_PICK: dict = {}   # pick_id -> {slot, token, accounts, ts}
_ADS_PAGE = ("<!doctype html><meta charset=utf-8><body style='font-family:sans-serif;background:#0f172a;"
             "color:#e2e8f0;text-align:center;padding:40px 20px'>")


def _fb_ads_assign(slot: str, token: str, acc: dict) -> str:
    acc_id = acc.get("account_id")
    if not acc_id:
        return "ไม่พบบัญชีโฆษณา"
    if acc_id in {a["ad_account_id"] for a in db.ads_list_accounts("meta")}:
        return "บัญชีนี้เชื่อมไว้แล้ว"
    if len(db.ads_list_accounts("meta")) >= 10:
        return "ครบ 10 บัญชีแล้ว"
    # เก็บชื่อบัญชีโฆษณาจริงเป็น label เพื่อแสดงให้เห็นว่าเชื่อมบัญชีไหน
    name = (acc.get("name") or "").strip()
    label = (f"ตัวที่ {slot} · {name}" if slot and name else name) or (f"Ads Facebook ตัวที่ {slot}" if slot else "Ads Facebook")
    db.ads_add_account("meta", label, token, acc_id)
    return ""


@app.get("/oauth-ads/facebook/callback")
async def fb_ads_oauth_callback():
    """Implicit-flow callback: the token is in the URL fragment (#access_token=…),
    which the server can't read — so return a page whose JS extracts it, sends it to
    the server to fetch ad accounts (no App Secret needed), and handles picking."""
    return HTMLResponse(_ADS_PAGE + """
<h2 id="m">⏳ กำลังเชื่อม…</h2><div id="pick" style="margin-top:14px"></div>
<script>
(function(){
  var h = new URLSearchParams((location.hash||'').replace(/^#/,''));
  var token = h.get('access_token'); var state = h.get('state')||''; var slot = (state.split('|')[0]||'');
  var m = document.getElementById('m'), pick = document.getElementById('pick');
  function done(t){ m.textContent=t; try{ if(window.opener) window.opener.postMessage('goffice-ads-connected','*'); }catch(e){} setTimeout(function(){ try{window.close();}catch(e){} },1800); }
  if(!token){ m.textContent='❌ ไม่ได้รับสิทธิ์ / ยกเลิก — ปิดหน้าต่างแล้วลองใหม่'; return; }
  if(state.indexOf('pages')===0){
    fetch('/api/fb/pages-connect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:token})})
     .then(function(r){return r.json();}).then(function(d){
       done(d.error?('❌ '+d.error):('✅ เชื่อมเพจ '+(d.count||0)+' เพจแล้ว — ปิดหน้าต่างนี้'));
     }).catch(function(){ m.textContent='❌ เชื่อมเพจไม่สำเร็จ'; });
    return;
  }
  fetch('/api/ads/fb-sdk-connect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:token,slot:slot})})
   .then(function(r){return r.json();}).then(function(d){
     if(d.error){ m.textContent='❌ '+d.error; return; }
     if(d.added){ done('✅ เชื่อม '+d.added+' แล้ว — ปิดหน้าต่างนี้ กลับไปรีเฟรช G Office'); return; }
     m.textContent='เลือกบัญชีโฆษณา 1 บัญชี สำหรับตัวที่ '+slot;
     (d.accounts||[]).forEach(function(a){
       var b=document.createElement('button');
       b.textContent='📘 '+(a.name||'Ad Account')+' · act_'+a.account_id;
       b.style.cssText='display:block;width:100%;max-width:420px;margin:8px auto;padding:12px;border-radius:10px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;cursor:pointer;font-weight:700';
       b.onclick=function(){ fetch('/api/ads/fb-sdk-pick',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pick_id:d.pick_id,account_id:a.account_id})}).then(function(r){return r.json();}).then(function(p){ pick.innerHTML=''; done(p.error?('❌ '+p.error):('✅ เชื่อม '+(p.added||'')+' แล้ว — ปิดหน้าต่างนี้')); }); };
       pick.appendChild(b);
     });
   }).catch(function(){ m.textContent='❌ เชื่อมไม่สำเร็จ ลองใหม่'; });
})();
</script>""")


@app.get("/oauth-ads/facebook/pick")
async def fb_ads_pick(pid: str = "", acc: str = ""):
    d = _FB_ADS_PICK.pop(pid or "", None)
    if not d:
        return HTMLResponse(_ADS_PAGE + "<h3>หมดเวลาเลือก — ปิดแล้วล็อกอินใหม่</h3>"
                            "<script>setTimeout(function(){window.close()},2500)</script>")
    a = next((x for x in d["accounts"] if str(x.get("account_id")) == str(acc)), None)
    if not a:
        return HTMLResponse(_ADS_PAGE + "<h3>ไม่พบบัญชีที่เลือก</h3>")
    err = _fb_ads_assign(d["slot"], d["token"], a)
    slot_txt = f"ตัวที่ {_esc(d['slot'])}" if d["slot"] else "ช่องนี้"
    msg = (f"❌ {err}" if err else f"✅ ผูก {slot_txt} กับ {_esc(a.get('name') or '')} แล้ว")
    return HTMLResponse(_ADS_PAGE + f"<h2>{msg}</h2><p>ปิดหน้าต่างนี้แล้วกลับไป G Office (รีเฟรช)</p>"
                        "<script>setTimeout(function(){window.close()},2200)</script>")


def _ads_shared_token() -> str:
    """Reuse the token from an already-connected Meta account (same business/user)."""
    accts = db.ads_list_accounts("meta")
    return accts[-1]["token"] if accts else ""


@app.get("/api/ads/available")
async def ads_available() -> dict:
    """List ad accounts reachable by the existing (shared) token — pick to add, no new token."""
    import httpx
    token = _ads_shared_token()
    if not token:
        return {"error": "ยังไม่มีบัญชีที่เชื่อมไว้ — เชื่อมตัวแรกด้วย token/ล็อกอินก่อน 1 ครั้ง", "accounts": []}
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://graph.facebook.com/v19.0/me/adaccounts",
                            params={"fields": "name,account_id", "access_token": token, "limit": 50})
        data = r.json().get("data", [])
    except Exception as exc:
        return {"error": _friendly_err(exc), "accounts": []}
    existing = {a["ad_account_id"] for a in db.ads_list_accounts("meta")}
    return {"accounts": [{"name": d.get("name") or "Ad Account", "account_id": d.get("account_id"),
                          "added": d.get("account_id") in existing} for d in data if d.get("account_id")]}


@app.post("/api/ads/account-quick")
async def ads_account_quick(p: dict) -> dict:
    """Add an ad account by ID only, reusing the existing shared token."""
    token = _ads_shared_token()
    if not token:
        return {"error": "ยังไม่มี token เดิม — เชื่อมตัวแรกด้วย token/ล็อกอินก่อน 1 ครั้ง"}
    acc_id = "".join(ch for ch in (p.get("ad_account_id") or "") if ch.isdigit())
    if not acc_id:
        return {"error": "ใส่ Ad Account ID ก่อน"}
    if acc_id in {a["ad_account_id"] for a in db.ads_list_accounts("meta")}:
        return {"error": "บัญชีนี้เชื่อมแล้ว"}
    if len(db.ads_list_accounts("meta")) >= 10:
        return {"error": "ครบ 10 บัญชีแล้ว"}
    db.ads_add_account("meta", (p.get("label") or "Ads Facebook").strip(), token, acc_id)
    return {"ok": True}


def _autopilot_cfg(a: dict) -> dict:
    try:
        return json.loads(a.get("autopilot") or "{}")
    except Exception:
        return {}


@app.get("/api/ads/accounts")
async def ads_accounts() -> dict:
    out = []
    for a in db.ads_list_accounts("meta"):
        t = a.get("token") or ""
        out.append({"id": a["id"], "label": a["label"], "ad_account_id": a["ad_account_id"],
                    "token": (t[:6] + "…") if t else "", "autopilot": _autopilot_cfg(a)})
    return {"accounts": out, "max": 10}


@app.post("/api/ads/account")
async def ads_account_add(p: dict) -> dict:
    if db.ads_count_accounts("meta") >= 10:
        return {"error": "เชื่อม Facebook ได้สูงสุด 10 บัญชี"}
    token = (p.get("token") or "").strip()
    acct = (p.get("ad_account_id") or "").strip()
    if not token or not acct:
        return {"error": "ใส่ Access token และ Ad Account ID"}
    label = (p.get("label") or ("บัญชี " + str(db.ads_count_accounts("meta") + 1))).strip()
    nid = db.ads_add_account("meta", label, token, acct)
    return {"ok": True, "id": nid}


@app.post("/api/ads/account/delete")
async def ads_account_delete(p: dict) -> dict:
    db.ads_delete_account(int(p.get("id")))
    return {"ok": True}


@app.post("/api/ads/account/refresh")
async def ads_account_refresh(p: dict) -> dict:
    """Fetch the real ad-account name from Meta and store it as the label."""
    acc = db.ads_get_account(int(p.get("id")))
    if not acc:
        return {"error": "ไม่พบบัญชี"}
    try:
        info = await __import__("asyncio").to_thread(ads.meta_account_info, acc["token"], acc["ad_account_id"])
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    name = (info.get("name") or "").strip()
    if not name:
        return {"error": "ดึงชื่อไม่ได้ (token หมดอายุ/ไม่มีสิทธิ์)"}
    label = f"{name} · {info['business']}" if info.get("business") else name
    db.ads_set_account_label(acc["id"], label)
    return {"ok": True, "label": label, "info": info}


# --- Ads Autopilot: A = manage budget within cap · B = auto-create campaign ---
_AUTOPILOT_MANAGE_PROMPT = (
    "คุณคือผู้จัดการโฆษณาอัตโนมัติ จัดสรรงบรายวันใหม่ภายใต้ 'เพดานงบรวม/วัน' ที่กำหนด "
    "(ห้ามรวมเกินเพดานเด็ดขาด) เป้าหมาย: {goal}. เพดานงบรวม/วัน: {cap} บาท.\n"
    "เกณฑ์: ตัวไม่มีคอนเวอร์ชัน/ROAS ต่ำ → pause · ตัว ROAS ดี → เพิ่มงบ · ก้ำกึ่ง → คงไว้ "
    "ผลรวม daily_budget ของตัวที่ active ต้อง ≤ เพดาน.\n\n"
    "ข้อมูลแคมเปญ (JSON):\n{data}\n\n"
    "ตอบ JSON อย่างเดียว: {{\"recos\":[{{\"campaign_id\":\"...\",\"action\":\"pause|scale|keep\","
    "\"reason\":\"สั้นๆ ไทย\",\"suggested_budget\":งบ/วัน(บาท)}}]}}"
)

_AUTOPILOT_CREATE_PROMPT = (
    "เขียนแผนโฆษณา Facebook 1 แคมเปญ (ภาษาไทย) จากโจทย์ลูกค้า ให้ครีเอทีฟดึงดูด ปลอดภัยตามนโยบาย Meta "
    "และเขียนคำโฆษณา 2 แบบ (A/B) + แนะนำกลุ่มเป้าหมาย.\n"
    "ธุรกิจ/โจทย์: {brief}\nเป้าหมาย: {goal}\n\n"
    "ตอบ JSON อย่างเดียว: {{\"campaign_name\":\"...\",\"headline\":\"พาดหัว A\","
    "\"primary_text\":\"ข้อความ A 1-3 บรรทัด มี CTA\",\"headline_b\":\"พาดหัว B\","
    "\"primary_text_b\":\"ข้อความ B\",\"image_prompt\":\"คำอธิบายภาพโฆษณา\","
    "\"age_min\":18,\"age_max\":65,\"gender\":\"all|male|female\"}}"
)


@app.post("/api/ads/autopilot")
async def ads_autopilot_save(p: dict) -> dict:
    acc = db.ads_get_account(int(p.get("account_id")))
    if not acc:
        return {"error": "ไม่พบบัญชี"}
    def _num(v):
        try: return float(v)
        except (TypeError, ValueError): return 0
    cfg = {
        "mode": p.get("mode") if p.get("mode") in ("off", "manage", "create") else "off",
        "daily_cap": _num(p.get("daily_cap")),
        "goal": (p.get("goal") or "traffic").strip(),
        "page_id": (p.get("page_id") or "").strip(),
        "link": (p.get("link") or "").strip(),
        "brief": (p.get("brief") or "").strip(),
        "auto_daily": bool(p.get("auto_daily")),
        "kill_spend": _num(p.get("kill_spend")),
        "report_to": (p.get("report_to") or "").strip(),
        "ab_test": bool(p.get("ab_test")),
        "use_image": bool(p.get("use_image")),
    }
    db.ads_set_autopilot(acc["id"], cfg)
    return {"ok": True, "autopilot": cfg}


def _gender_codes(g: str):
    return [1] if g == "male" else ([2] if g == "female" else None)


async def _ads_report(report_to: str, text: str):
    """Send an autopilot summary to a connected channel (LINE/Telegram/Slack/Discord/Email)."""
    if not report_to:
        return
    row = db.get_connector(report_to)
    if not row or row.get("status") != "connected":
        return
    try:
        await _send_to_connector(report_to, row.get("config") or "", text)
    except Exception:
        pass


async def _run_autopilot(acc: dict) -> dict:
    """Run the account's chosen Autopilot mode. Returns {ok/error, message, actions}."""
    import asyncio
    cfg = _autopilot_cfg(acc)
    mode = cfg.get("mode")
    model = model_for_agent("analyst")
    tok, acct_id = acc["token"], acc.get("ad_account_id", "")
    if mode == "manage":
        cap = cfg.get("daily_cap") or 0
        if cap <= 0:
            return {"error": "ตั้งเพดานงบ/วัน ก่อน (โหมดจัดการงบ)"}
        try:
            rows = await asyncio.to_thread(ads.meta_campaigns, tok, acct_id)
        except Exception as exc:
            return {"error": "ดึงแคมเปญไม่สำเร็จ: " + _friendly_err(exc)}
        if not rows:
            return {"ok": True, "message": "ยังไม่มีแคมเปญให้จัดการ", "actions": []}
        done = []
        # Kill-switch: ถ้าใช้งบวันนี้เกินที่ตั้ง → หยุดทุกแคมเปญ
        kill = cfg.get("kill_spend") or 0
        if kill > 0:
            try:
                today = await asyncio.to_thread(ads.meta_campaigns, tok, acct_id, "today")
                spent = sum(float(r.get("spend") or 0) for r in today)
                if spent >= kill:
                    for r in today:
                        if (r.get("status") or "").upper() == "ACTIVE":
                            try: await asyncio.to_thread(ads.meta_pause, tok, r["id"])
                            except Exception: pass
                    msg = f"🛑 Kill-switch: ใช้งบวันนี้ ฿{spent:.0f} ≥ ฿{kill:.0f} — หยุดทุกแคมเปญแล้ว"
                    await _ads_report(cfg.get("report_to"), f"[{acc['label']}] {msg}")
                    return {"ok": True, "message": msg, "actions": []}
            except Exception:
                pass
        data = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        raw = await _claude_text("analyst", model,
                                 _AUTOPILOT_MANAGE_PROMPT.format(goal=cfg.get("goal"), cap=cap,
                                 data=json.dumps(data, ensure_ascii=False)), 2000)
        recos = {r.get("campaign_id"): r for r in _parse_json(raw).get("recos", [])}
        for r in rows:
            rc = recos.get(r["id"], {})
            act = rc.get("action")
            try:
                if act == "pause":
                    await asyncio.to_thread(ads.meta_pause, tok, r["id"]); done.append(f"⏸️ {r['name']}")
                elif act == "scale":
                    b = float(rc.get("suggested_budget") or r.get("daily_budget") or 0)
                    if b > 0:
                        await asyncio.to_thread(ads.meta_set_budget, tok, r["id"], b); done.append(f"🔼 {r['name']} → ฿{b}/วัน")
            except Exception as exc:
                done.append(f"⚠️ {r['name']}: {type(exc).__name__}")
        msg = "จัดการงบอัตโนมัติแล้ว (เพดาน ฿%s/วัน)" % cap
        await _ads_report(cfg.get("report_to"), f"[{acc['label']}] {msg}\n" + ("\n".join(done) or "ไม่มีการเปลี่ยนแปลง"))
        return {"ok": True, "message": msg, "actions": done}
    if mode == "create":
        if not cfg.get("page_id") or not cfg.get("link"):
            return {"error": "โหมดสร้างแคมเปญ ต้องใส่ Page ID + ลิงก์ปลายทาง ก่อน"}
        raw = await _claude_text("marketing", model,
                                 _AUTOPILOT_CREATE_PROMPT.format(brief=cfg.get("brief") or acc["label"],
                                 goal=cfg.get("goal")), 1500)
        plan = _parse_json(raw)
        image_hash = ""
        if cfg.get("use_image"):
            img = _media_cfg("image")
            if img["key"]:
                try:
                    im = await asyncio.to_thread(media.generate_image, img["provider"], img["model"],
                                                 img["key"], plan.get("image_prompt") or cfg.get("brief") or "ad creative")
                    if im.get("b64"):
                        image_hash = await asyncio.to_thread(ads.meta_upload_image, tok, acct_id, im["b64"])
                except Exception:
                    image_hash = ""
        targeting = {"age_min": plan.get("age_min") or 18, "age_max": plan.get("age_max") or 65,
                     "genders": _gender_codes(plan.get("gender") or "all"), "countries": ["TH"]}
        variant_b = {"message": plan.get("primary_text_b"), "headline": plan.get("headline_b")} if cfg.get("ab_test") else None
        try:
            res = await asyncio.to_thread(
                ads.meta_create_campaign, tok, acct_id,
                plan.get("campaign_name") or "G Office Autopilot", cfg.get("goal"),
                cfg.get("daily_cap") or 100, cfg.get("page_id"), cfg.get("link"),
                plan.get("primary_text") or "", plan.get("headline") or "",
                targeting, image_hash, variant_b)
        except Exception as exc:
            return {"error": "สร้างแคมเปญไม่สำเร็จ: " + _friendly_err(exc)}
        extras = []
        if image_hash: extras.append("แนบรูป")
        if res.get("ad_b_id"): extras.append("A/B 2 แบบ")
        msg = "สร้างแคมเปญใหม่ (PAUSED — ตรวจก่อนเปิด)" + (" · " + ", ".join(extras) if extras else "")
        await _ads_report(cfg.get("report_to"), f"[{acc['label']}] {msg}\nแคมเปญ: {plan.get('campaign_name','')}")
        return {"ok": True, "message": msg, "plan": plan, "result": res}
    return {"error": "ยังไม่ได้เลือกโหมด Autopilot ของบัญชีนี้"}


@app.post("/api/ads/autopilot/run")
async def ads_autopilot_run(p: dict) -> dict:
    acc = db.ads_get_account(int(p.get("account_id")))
    if not acc:
        return {"error": "ไม่พบบัญชี"}
    return await _run_autopilot(acc)


@app.post("/api/ads/duplicate")
async def ads_duplicate(p: dict) -> dict:
    """Duplicate a winning campaign (created PAUSED)."""
    import asyncio
    reco = db.ads_get_reco(int(p.get("id"))) if p.get("id") else None
    cid = (p.get("campaign_id") or (reco or {}).get("campaign_id") or "").strip()
    acc_id = (reco or {}).get("account_id") or p.get("account_id")
    acc = db.ads_get_account(int(acc_id)) if acc_id else None
    if not acc or not cid:
        return {"error": "ไม่พบบัญชี/แคมเปญ"}
    try:
        res = await asyncio.to_thread(ads.meta_duplicate_campaign, acc["token"], cid)
    except Exception as exc:
        return {"error": "ก๊อปแคมเปญไม่สำเร็จ: " + _friendly_err(exc)}
    return {"ok": True, "message": "ก๊อปแคมเปญแล้ว (PAUSED — ตรวจก่อนเปิด)", "result": res}


@app.post("/api/ads/library/search")
async def ads_library_search(p: dict) -> dict:
    """Search Facebook Ad Library for competitor ads."""
    import asyncio
    token = _ads_shared_token()      # use a connected account's user token to query the library
    if not token:
        return {"error": "ต้องเชื่อมบัญชี Facebook อย่างน้อย 1 บัญชีก่อน (ใช้ token ของคุณค้นคลังโฆษณา)", "ads": []}
    q = (p.get("query") or "").strip()
    if not q:
        return {"error": "ใส่คำค้น (ชื่อแบรนด์/สินค้า/คู่แข่ง)", "ads": []}
    country = (p.get("country") or "TH").strip()
    ad_type = p.get("ad_type") if p.get("ad_type") in ("ALL", "POLITICAL_AND_ISSUE_ADS") else "ALL"
    try:
        rows = await asyncio.to_thread(ads.meta_ad_library, token, q, [country], ad_type, 24)
    except Exception as exc:
        return {"error": "ดึงคลังโฆษณาไม่สำเร็จ: " + _friendly_err(exc) +
                " (โฆษณาทั่วไปบางประเทศ Meta จำกัดการเข้าถึง API)", "ads": []}
    out = []
    for r in rows:
        bodies = r.get("ad_creative_bodies") or []
        titles = r.get("ad_creative_link_titles") or []
        out.append({"id": r.get("id"), "page": r.get("page_name", ""),
                    "body": bodies[0] if bodies else "", "title": titles[0] if titles else "",
                    "snapshot": r.get("ad_snapshot_url", ""),
                    "since": (r.get("ad_delivery_start_time") or "")[:10],
                    "platforms": r.get("publisher_platforms") or []})
    return {"ads": out, "count": len(out)}


_AD_IMITATE_PROMPT = (
    "นี่คือโฆษณา Facebook ของคู่แข่ง (ข้อความ/คำอธิบายที่ผู้ใช้เห็น):\n«{ad}»\n\n"
    "ธุรกิจของผู้ใช้: {brief}\n\n"
    "งาน: วิเคราะห์กลยุทธ์/จุดขาย/อารมณ์ของโฆษณานี้ แล้วสร้างของ \"เลียนแบบสไตล์\" ให้ผู้ใช้ "
    "(ห้ามลอกคำต่อคำ ดัดแปลงเป็นของเราเอง ปลอดภัยตามนโยบาย Meta)\n"
    "ตอบ JSON อย่างเดียว:\n"
    "{{\"analysis\":\"วิเคราะห์สั้นๆ 2-3 ข้อ (ภาษาไทย)\","
    "\"captions\":[\"โฆษณาใหม่แบบ A: พาดหัว + เนื้อหา + CTA\",\"โฆษณาใหม่แบบ B\"],"
    "\"image_prompt\":\"prompt ภาษาอังกฤษสำหรับสร้าง 'ภาพโฆษณา' สไตล์เดียวกัน (องค์ประกอบ/มุมภาพ/โทนสี/อารมณ์) แต่เป็นสินค้า/บริการของผู้ใช้\","
    "\"video_prompt\":\"prompt ภาษาอังกฤษสำหรับสร้าง 'วิดีโอโฆษณาสั้น' สไตล์เดียวกัน (ฉาก/การเคลื่อนไหว/โทน) ของผู้ใช้\"}}"
)


@app.post("/api/ads/library/imitate")
async def ads_library_imitate(p: dict) -> dict:
    ad = (p.get("ad") or "").strip()
    brief = (p.get("brief") or "").strip()
    if not ad or not brief:
        return {"error": "ต้องมีข้อความ/คำอธิบายโฆษณา + โจทย์ธุรกิจของคุณ"}
    try:
        txt = await _claude_text("marketing", model_for_agent("marketing"),
                                 _AD_IMITATE_PROMPT.format(ad=ad[:1500], brief=brief), 1500)
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    data = _parse_json(txt) or {}
    if not data:
        return {"ok": True, "result": txt}     # fallback: raw text
    return {"ok": True,
            "analysis": data.get("analysis", ""),
            "captions": data.get("captions", []),
            "image_prompt": data.get("image_prompt", ""),
            "video_prompt": data.get("video_prompt", "")}


@app.post("/api/ads/library/imitate-image")
async def ads_library_imitate_image(p: dict) -> dict:
    """Look at an uploaded ad screenshot (AI vision) → analysis + caption + image/video prompts."""
    import base64
    img = (p.get("image") or "").strip()
    brief = (p.get("brief") or "").strip()
    if not img:
        return {"error": "อัปโหลดสกรีนช็อตโฆษณาก่อน"}
    if not brief:
        return {"error": "ใส่โจทย์ธุรกิจของคุณก่อน"}
    # parse data URI
    media_type, b64 = "image/png", img
    if img.startswith("data:"):
        try:
            head, b64 = img.split(",", 1)
            media_type = head.split(":", 1)[1].split(";", 1)[0] or "image/png"
        except Exception:
            return {"error": "รูปไม่ถูกต้อง"}
    # cap ~4.5MB of base64
    if len(b64) > 6_000_000:
        return {"error": "รูปใหญ่เกินไป — ย่อขนาดก่อน (ไม่เกิน ~4MB)"}
    instruction = (
        "ดูภาพโฆษณา Facebook ของคู่แข่งนี้ แล้ววิเคราะห์: องค์ประกอบภาพ/มุมกล้อง/โทนสี/อารมณ์/จุดขาย/ข้อความบนภาพ.\n"
        f"ธุรกิจของผู้ใช้: {brief}\n"
        "สร้างของ 'เลียนแบบสไตล์' ให้ผู้ใช้ (ห้ามลอกตรงๆ ปลอดภัยตามนโยบาย Meta).\n"
        "ตอบ JSON อย่างเดียว: {\"analysis\":\"วิเคราะห์ภาพ 2-3 ข้อ (ไทย)\","
        "\"captions\":[\"แคปชั่นใหม่ A (พาดหัว+เนื้อหา+CTA)\",\"แคปชั่นใหม่ B\"],"
        "\"image_prompt\":\"prompt อังกฤษสร้างภาพสไตล์เดียวกันแต่เป็นสินค้า/บริการของผู้ใช้\","
        "\"video_prompt\":\"prompt อังกฤษสร้างวิดีโอโฆษณาสั้นสไตล์เดียวกัน\"}"
    )
    try:
        resp = await client_for_agent("marketing").messages.create(
            model=model_for_agent("marketing"), max_tokens=1500,
            **thinking_kwargs(model_for_agent("marketing")),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": instruction},
            ]}])
        txt = next((b.text for b in resp.content if b.type == "text"), "{}")
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    data = _parse_json(txt) or {}
    if not data:
        return {"ok": True, "result": txt}
    return {"ok": True,
            "analysis": data.get("analysis", ""),
            "captions": data.get("captions", []),
            "image_prompt": data.get("image_prompt", ""),
            "video_prompt": data.get("video_prompt", "")}


# ============================== Auto Post Facebook (Pages) ===================
@app.post("/api/fb/pages-connect")
async def fb_pages_connect(p: dict) -> dict:
    import asyncio
    token = (p.get("token") or "").strip()
    if not token:
        return {"error": "ไม่ได้รับ token"}
    try:
        pages = await asyncio.to_thread(ads.meta_get_pages, token)
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    if not pages:
        return {"error": "ไม่พบเพจที่คุณจัดการ (หรือยังไม่ได้ให้สิทธิ์เพจ)"}
    for pg in pages:
        if pg.get("id") and pg.get("access_token"):
            db.fb_page_upsert(pg["id"], pg.get("name", ""), pg["access_token"])
    return {"ok": True, "count": len(pages)}


@app.get("/api/fb/pages")
async def fb_pages() -> dict:
    return {"pages": [{"page_id": p["page_id"], "name": p["name"],
                       "autoreply": bool(p.get("autoreply"))} for p in db.fb_pages_list()]}


@app.post("/api/fb/page/delete")
async def fb_page_delete(p: dict) -> dict:
    db.fb_page_delete(p.get("page_id"))
    return {"ok": True}


def _decode_image_arg(p: dict):
    """Resolve an image from the request into (bytes, mime) or (None, None).
    Accepts: image_id (from Image AI board), image (data URI / raw base64), image_url."""
    import base64
    # 1) pick from Image AI board
    iid = p.get("image_id")
    ref = None
    if iid:
        try:
            ref = next((t.get("ref") for t in db.image_list() if t["id"] == int(iid)), None)
        except Exception:
            ref = None
    # 2) explicit data URI / base64
    ref = ref or p.get("image")
    if ref and ref.startswith("data:"):
        try:
            head, b64 = ref.split(",", 1)
            mime = head.split(":", 1)[1].split(";", 1)[0] or "image/png"
            return base64.b64decode(b64), mime
        except Exception:
            return None, None
    if ref:  # raw base64 with no header
        try:
            return base64.b64decode(ref), "image/png"
        except Exception:
            return None, None
    return None, None


@app.post("/api/fb/post")
async def fb_post(p: dict) -> dict:
    import asyncio
    pg = db.fb_page_get(p.get("page_id"))
    if not pg:
        return {"error": "ไม่พบเพจ"}
    msg = (p.get("message") or "").strip()
    img_bytes, mime = _decode_image_arg(p)
    img_url = (p.get("image_url") or "").strip()
    if not msg and not img_bytes and not img_url:
        return {"error": "ใส่ข้อความโพสต์ หรือแนบรูปก่อน"}
    try:
        if img_bytes:
            ext = "jpg" if "jpeg" in (mime or "") else "png"
            res = await asyncio.to_thread(ads.meta_page_photo, pg["token"], pg["page_id"],
                                          img_bytes, msg, f"image.{ext}")
        elif img_url:
            res = await asyncio.to_thread(ads.meta_page_photo_url, pg["token"], pg["page_id"],
                                          img_url, msg)
        else:
            res = await asyncio.to_thread(ads.meta_page_post, pg["token"], pg["page_id"],
                                          msg, (p.get("link") or "").strip())
    except Exception as exc:
        return {"error": "โพสต์ไม่สำเร็จ: " + _friendly_err(exc)}
    return {"ok": True, "message": "โพสต์ขึ้นเพจแล้ว ✅",
            "id": res.get("id") or res.get("post_id")}


@app.post("/api/fb/write-post")
async def fb_write_post(p: dict) -> dict:
    """AI เขียนแคปชั่นโพสต์จากหัวข้อ."""
    topic = (p.get("topic") or "").strip()
    if not topic:
        return {"error": "ใส่หัวข้อก่อน"}
    try:
        txt = await _claude_text("content", model_for_agent("content"),
                                 f"เขียนแคปชั่นโพสต์ Facebook ภาษาไทย น่าสนใจ มี emoji + แฮชแท็ก สำหรับ: {topic}", 600)
    except Exception as exc:
        return {"error": _friendly_err(exc)}
    return {"ok": True, "caption": txt.strip()}


@app.post("/api/fb/comments")
async def fb_comments(p: dict) -> dict:
    import asyncio
    pg = db.fb_page_get(p.get("page_id"))
    if not pg:
        return {"error": "ไม่พบเพจ", "comments": []}
    try:
        cs = await asyncio.to_thread(ads.meta_page_recent_comments, pg["token"], pg["page_id"])
    except Exception as exc:
        return {"error": _friendly_err(exc), "comments": []}
    return {"comments": cs}


@app.post("/api/fb/comment-reply")
async def fb_comment_reply(p: dict) -> dict:
    import asyncio
    pg = db.fb_page_get(p.get("page_id"))
    if not pg:
        return {"error": "ไม่พบเพจ"}
    cid = (p.get("comment_id") or "").strip()
    reply = (p.get("reply") or "").strip()
    if not reply:                            # ให้ AI ร่างถ้าไม่ส่งมา
        try:
            reply = (await _ai_reply(p.get("text") or "")).strip() or "ขอบคุณที่สนใจค่ะ 🙏"
        except Exception:
            reply = "ขอบคุณที่สนใจค่ะ 🙏"
    try:
        await asyncio.to_thread(ads.meta_reply_comment, pg["token"], cid, reply)
    except Exception as exc:
        return {"error": "ตอบไม่สำเร็จ: " + _friendly_err(exc)}
    return {"ok": True, "reply": reply}


@app.post("/api/fb/page/autoreply")
async def fb_page_autoreply(p: dict) -> dict:
    db.fb_page_set_autoreply(p.get("page_id"), bool(p.get("on")))
    return {"ok": True}


async def _fb_autoreply_loop() -> None:
    """Auto-reply new comments on pages that turned autoreply on."""
    import asyncio
    await asyncio.sleep(90)
    seen = set()
    while True:
        try:
            for pg in db.fb_pages_list():
                if not pg.get("autoreply"):
                    continue
                try:
                    cs = await asyncio.to_thread(ads.meta_page_recent_comments, pg["token"], pg["page_id"])
                except Exception:
                    continue
                for c in cs:
                    cid = c.get("comment_id")
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    try:
                        reply = (await _ai_reply(c.get("text") or "")).strip() or "ขอบคุณที่สนใจค่ะ 🙏"
                        await asyncio.to_thread(ads.meta_reply_comment, pg["token"], cid, reply)
                    except Exception:
                        pass
        except Exception:
            pass
        await asyncio.sleep(300)     # check every 5 min


# --- Chat Bot (Telegram conversation viewer) --------------------------------
def _telegram_creds():
    """(bot_token, chat_id) from the connected Telegram connector, or (None, None)."""
    row = db.get_connector("Telegram")
    if not row or row.get("status") != "connected":
        return None, None
    try:
        data = json.loads(row.get("config") or "{}")
    except Exception:
        data = {}
    return data.get("bot_token", ""), str(data.get("chat_id", "") or "").strip()


def _tg_now() -> str:
    from datetime import datetime, timezone, timedelta
    return datetime.now(tz=timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M")


async def _telegram_pull(tok: str, long_poll: bool = False) -> list:
    """Fetch new updates via getUpdates and store incoming. Returns the new incoming messages."""
    import httpx
    from datetime import datetime, timezone, timedelta
    offset = db.chat_last_update_id("telegram")
    params = {"limit": 100, "timeout": 25 if long_poll else 0}
    if offset:
        params["offset"] = offset + 1
    new_msgs = []
    async with httpx.AsyncClient(timeout=(35 if long_poll else 25)) as client:
        r = await client.get(f"https://api.telegram.org/bot{tok}/getUpdates", params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram {r.status_code}: {r.text[:160]}")
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("description") or "getUpdates ล้มเหลว")
    for u in data.get("result", []):
        uid = u.get("update_id")
        msg = u.get("message") or u.get("edited_message") or u.get("channel_post") or {}
        if not msg:
            # still advance the offset so we don't loop on non-message updates
            db.chat_msg_add("telegram", None, "sys", "", "", _tg_now(), update_id=uid)
            continue
        frm = msg.get("from") or {}
        name = (frm.get("first_name", "") + " " + frm.get("last_name", "")).strip() \
            or frm.get("username") or "ผู้ใช้"
        text = msg.get("text") or msg.get("caption") or ("[" + (
            "รูปภาพ" if msg.get("photo") else "ไฟล์/สื่อ") + "]")
        chat = (msg.get("chat") or {}).get("id")
        ts = datetime.fromtimestamp(msg.get("date", 0), tz=timezone(timedelta(hours=7))) \
            .strftime("%Y-%m-%d %H:%M") if msg.get("date") else _tg_now()
        if db.chat_msg_add("telegram", chat, "in", name, text, ts, update_id=uid):
            new_msgs.append({"chat_id": chat, "name": name, "text": text})
    return new_msgs


async def _telegram_send(tok: str, chat_id, text: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat_id, "text": text})


# live status of the Telegram poller so the UI can explain "why nothing happens"
_TG_STATUS = {"error": "", "ok_at": 0, "polls": 0, "bot": "", "webhook_cleared": False}


async def _telegram_assistant_reply(incoming: str) -> str:
    """G Office personal assistant reply (concise Thai summary) using recent context."""
    history = [m for m in db.chat_msgs_list("telegram", 14) if m.get("direction") in ("in", "out")]
    msgs = []
    for m in history[-12:]:
        role = "user" if m["direction"] == "in" else "assistant"
        if m.get("text"):
            msgs.append({"role": role, "content": m["text"]})
    if not msgs or msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": incoming})
    try:
        resp = await _client.messages.create(
            model=get_model(), max_tokens=700, **thinking_kwargs(),
            system=("คุณคือผู้ช่วย AI ของ G Office (ออฟฟิศ AI ของผู้ใช้) คุยกับเจ้าของผ่าน Telegram "
                    "ตอบเป็นภาษาไทย กระชับ ตรงประเด็น สรุปเป็นข้อ ๆ เมื่อเหมาะสม ใช้ emoji พอประมาณ "
                    "ถ้าถูกถามเรื่องงาน/การตลาด/เทรด/คอนเทนต์ ให้คำแนะนำที่ทำได้จริง ไม่เกิน 6 บรรทัด"),
            messages=msgs)
        return next((b.text for b in resp.content if b.type == "text"), "").strip()
    except Exception:
        return ""


_TG_HELP = (
    "🤖 สั่งงาน G Office ผ่าน Telegram ได้เลย:\n"
    "• สถานะ — สรุปภาพรวม (บอท + กระเป๋า)\n"
    "• กระเป๋า — ดูยอดเดโม่ (จริง/ลอยตัว/รวม)\n"
    "• เติม 100 / ถอน 50 — ปรับเครดิตเดโม่\n"
    "• บอท — ดูรายชื่อบอทเทรด + สถานะ\n"
    "• เริ่มบอท <ชื่อ> / หยุดบอท <ชื่อ>\n"
    "• รันคอนเทนต์ — สั่งทีมทำคอนเทนต์ 1 รอบ\n"
    "หรือพิมพ์ถามอะไรก็ได้ เดี๋ยว AI ตอบให้ 💬"
)


async def _telegram_command(text: str):
    """Run a real G Office action from a Telegram message. Returns a reply, or None to fall through to AI."""
    import re as _re
    t = (text or "").strip()
    low = t.lower()
    def has(*ks): return any(k in low for k in ks)

    if low in ("/start", "/help") or has("ช่วยอะไรได้", "คำสั่ง", "เมนูคำสั่ง", "สั่งอะไรได้"):
        return _TG_HELP

    # adjust demo wallet: "เติม 100" / "ถอน 50"
    m = _re.search(r"(เติม|ถอน)\s*\$?\s*([0-9][0-9,\.]*)", t)
    if m:
        amt = float(m.group(2).replace(",", ""))
        bal = db.paper_balance_adjust(amt if m.group(1) == "เติม" else -amt)
        verb = "เติม" if m.group(1) == "เติม" else "ถอน"
        return f"✅ {verb}กระเป๋าเดโม่ ${amt:,.2f} แล้ว\n💰 ยอดเงินจริงตอนนี้ ${bal:,.2f}"

    # wallet view
    if low in ("กระเป๋า", "เงิน", "ยอด", "wallet") or has("ยอดเงิน", "ยอดคงเหลือ", "กระเป๋าเดโม"):
        bal = db.paper_balance_get()
        unreal = sum(_bot_unrealized(b) for b in db.list_bots() if b["mode"] == "paper")
        return (f"💰 กระเป๋าเดโม่ (เครดิตปลอม)\n"
                f"ยอดเงินจริง ${bal:,.2f}\n"
                f"ลอยตัว {unreal:+,.2f}\n"
                f"มูลค่ารวมสด ${bal + unreal:,.2f}")

    # overall status
    if has("สถานะ", "status", "สรุปภาพรวม", "ภาพรวม") or low == "สรุป":
        bots = db.list_bots()
        running = [b for b in bots if b["status"] == "running"]
        bal = db.paper_balance_get()
        unreal = sum(_bot_unrealized(b) for b in bots if b["mode"] == "paper")
        asst = db.get_settings().get("telegram_assistant") == "true"
        return ("📊 สรุป G Office\n"
                f"🤖 บอทเทรด: {len(running)}/{len(bots)} กำลังรัน\n"
                f"💰 กระเป๋าเดโม่: ${bal + unreal:,.2f} (จริง ${bal:,.2f} · ลอยตัว {unreal:+,.2f})\n"
                f"💬 ผู้ช่วย Telegram: {'เปิด' if asst else 'ปิด'}")

    # start/stop a bot by name
    m = _re.search(r"(เริ่ม|เปิด|หยุด|ปิด|start|stop)\s*บอท\s*(.+)", t, _re.I)
    if m:
        want = m.group(2).strip().lower()
        start = m.group(1).lower() in ("เริ่ม", "เปิด", "start")
        bots = db.list_bots()
        hit = next((b for b in bots if want in b["name"].lower() or want in b["symbol"].lower()), None)
        if not hit:
            return f"❓ ไม่เจอบอทชื่อ “{m.group(2).strip()}” — พิมพ์ “บอท” เพื่อดูรายชื่อ"
        db.set_bot_status(hit["id"], "running" if start else "stopped")
        db.add_trade_log(hit["id"], "start" if start else "stop",
                         f"{'เริ่ม' if start else 'หยุด'}บอทผ่าน Telegram", 0.0)
        return f"{'▶️ เริ่ม' if start else '■ หยุด'}บอท “{hit['name']}” ({hit['symbol']}) แล้ว"

    # list bots
    if low in ("บอท", "bots", "บอทเทรด", "รายชื่อบอท"):
        bots = db.list_bots()
        if not bots:
            return "ยังไม่มีบอทเทรด — สร้างในเว็บก่อน"
        out = ["🤖 บอทเทรด:"]
        for b in bots:
            st = b.get("state") or {}
            out.append(f"• {b['name']} [{b['symbol']}] "
                       f"{'▶รัน' if b['status'] == 'running' else '■หยุด'} · "
                       f"PnL วันนี้ {float(st.get('daily_pnl') or 0):+.2f}")
        return "\n".join(out)

    # run a content cycle
    if has("รันคอนเทนต์", "ทำคอนเทนต์", "คอนเทนต์", "content", "โพสต์อัตโนมัติ"):
        try:
            post = await run_content_cycle(db.get_content_rules())
        except Exception as exc:
            return "❌ รันคอนเทนต์ไม่สำเร็จ: " + _friendly_err(exc)
        if post and post.get("caption"):
            return "✅ ทีมทำคอนเทนต์ 1 รอบแล้ว\n📝 " + post["caption"][:300]
        return "✅ สั่งทีมทำคอนเทนต์ 1 รอบแล้ว (ดูผลในเว็บ เมนูโรงงานคอนเทนต์)"

    return None


async def _telegram_loop() -> None:
    """Single poller: mirror Telegram chat into DB; auto-reply via AI when assistant is on."""
    import asyncio
    import httpx
    import time as _t
    await asyncio.sleep(20)
    while True:
        tok, _ = _telegram_creds()
        if not tok:
            await asyncio.sleep(30)
            continue
        # a webhook blocks getUpdates (409); clear it once so polling works
        if not _TG_STATUS["webhook_cleared"]:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    await client.get(f"https://api.telegram.org/bot{tok}/deleteWebhook")
                _TG_STATUS["webhook_cleared"] = True
            except Exception:
                pass
        try:
            new_msgs = await _telegram_pull(tok, long_poll=True)
            _TG_STATUS["error"] = ""
            _TG_STATUS["ok_at"] = int(_t.time())
            _TG_STATUS["polls"] += 1
            assistant_on = db.get_settings().get("telegram_assistant") == "true"
            _, owner = _telegram_creds()
            if assistant_on:
                for nm in new_msgs:
                    # commands only from the owner's chat (if a Chat ID is configured)
                    is_owner = (not owner) or (str(nm["chat_id"]) == str(owner))
                    reply, who = "", "ผู้ช่วย AI"
                    if is_owner:
                        cmd = await _telegram_command(nm["text"])
                        if cmd:
                            reply, who = cmd, "G Office"
                    if not reply:
                        reply = (await _telegram_assistant_reply(nm["text"])).strip()
                    if not reply:
                        continue
                    await _telegram_send(tok, nm["chat_id"], reply)
                    db.chat_msg_add("telegram", nm["chat_id"], "out", who, reply, _tg_now())
        except Exception as exc:
            _TG_STATUS["error"] = _friendly_err(exc)
            await asyncio.sleep(15)


@app.get("/api/chatbot/diag")
async def chatbot_diag() -> dict:
    """Diagnose why Telegram chat may not be working (token, webhook, assistant, poller)."""
    import httpx
    import time as _t
    tok, chat = _telegram_creds()
    out = {"connected": bool(tok), "assistant": db.get_settings().get("telegram_assistant") == "true",
           "chat_id": chat, "poller_polls": _TG_STATUS["polls"],
           "poller_error": _TG_STATUS["error"],
           "poller_age_sec": (int(_t.time()) - _TG_STATUS["ok_at"]) if _TG_STATUS["ok_at"] else None,
           "checks": []}
    if not tok:
        out["checks"].append(["❌", "ยังไม่ได้เชื่อม Telegram — เชื่อมในเมนูการเชื่อมต่อก่อน"])
        return out
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            me = (await client.get(f"https://api.telegram.org/bot{tok}/getMe")).json()
            wh = (await client.get(f"https://api.telegram.org/bot{tok}/getWebhookInfo")).json()
            if me.get("ok"):
                bot = me["result"].get("username", "")
                _TG_STATUS["bot"] = bot
                out["bot"] = bot
                out["checks"].append(["✅", f"Bot Token ใช้ได้ — บอท @{bot}"])
            else:
                out["checks"].append(["❌", "Bot Token ไม่ถูกต้อง — คัดลอกใหม่จาก @BotFather"])
                return out
            whurl = (wh.get("result") or {}).get("url") or ""
            if whurl:
                # a webhook is set → getUpdates is blocked; clear it
                await client.get(f"https://api.telegram.org/bot{tok}/deleteWebhook")
                _TG_STATUS["webhook_cleared"] = True
                out["checks"].append(["🔧", "พบ webhook ค้างอยู่ (บล็อกการอ่านข้อความ) — ลบให้แล้ว ลองพิมพ์ใหม่"])
            else:
                out["checks"].append(["✅", "ไม่มี webhook บล็อก — อ่านข้อความได้"])
    except Exception as exc:
        out["checks"].append(["❌", "ต่อ Telegram ไม่ได้: " + _friendly_err(exc)])
        return out
    out["checks"].append(["✅" if out["assistant"] else "⚠️",
                          "ผู้ช่วย AI เปิดอยู่ — บอทจะตอบให้อัตโนมัติ" if out["assistant"]
                          else "ผู้ช่วย AI ปิดอยู่ — เปิดสวิตช์ “ให้บอทคุยกับฉัน” ก่อน บอทถึงจะตอบ"])
    inc = len([m for m in db.chat_msgs_list("telegram") if m.get("direction") == "in"])
    out["checks"].append(["✅" if inc else "⚠️",
                          f"ได้รับข้อความเข้ามาแล้ว {inc} ข้อความ" if inc
                          else "ยังไม่เคยได้รับข้อความ — กด Start กับบอท แล้วพิมพ์ทักไปหาบอท 1 ครั้ง"])
    return out


@app.get("/api/chatbot/messages")
async def chatbot_messages() -> dict:
    tok, chat = _telegram_creds()
    if not tok:
        return {"error": "ยังไม่ได้เชื่อม Telegram — ไปที่เมนู “การเชื่อมต่อ” แล้วเชื่อม Telegram ก่อน",
                "messages": [], "connected": False}
    msgs = [m for m in db.chat_msgs_list("telegram") if m.get("direction") != "sys"]
    return {"messages": msgs, "connected": True, "chat_id": chat,
            "assistant": db.get_settings().get("telegram_assistant") == "true",
            "warn": _TG_STATUS["error"]}


@app.post("/api/chatbot/assistant")
async def chatbot_assistant(p: dict) -> dict:
    db.set_settings({"telegram_assistant": "true" if p.get("on") else "false"})
    return {"ok": True, "on": bool(p.get("on"))}


@app.post("/api/chatbot/send")
async def chatbot_send(p: dict) -> dict:
    import httpx
    tok, default_chat = _telegram_creds()
    if not tok:
        return {"error": "ยังไม่ได้เชื่อม Telegram"}
    text = (p.get("text") or "").strip()
    chat = str(p.get("chat_id") or default_chat or "").strip()
    if not text:
        return {"error": "พิมพ์ข้อความก่อน"}
    if not chat:
        return {"error": "ยังไม่มี Chat ID — ตั้งค่าตอนเชื่อม Telegram"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": text})
    if r.status_code >= 400:
        low = r.text.lower()
        if "can't send messages to the bot" in low or "chat not found" in low:
            return {"error": "ส่งไม่ได้ — Chat ID ไม่ถูกต้อง (ดูวิธีหา Chat ID ที่เมนูการเชื่อมต่อ)"}
        return {"error": f"Telegram {r.status_code}: {r.text[:160]}"}
    from datetime import datetime, timezone, timedelta
    ts = datetime.now(tz=timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M")
    db.chat_msg_add("telegram", chat, "out", "บอท (คุณ)", text, ts)
    return {"ok": True}


@app.post("/api/chatbot/ai-reply")
async def chatbot_ai_reply(p: dict) -> dict:
    """Let AI draft + send a reply to the latest incoming Telegram message."""
    import httpx
    tok, default_chat = _telegram_creds()
    if not tok:
        return {"error": "ยังไม่ได้เชื่อม Telegram"}
    msgs = [m for m in db.chat_msgs_list("telegram") if m.get("direction") == "in"]
    if not msgs:
        return {"error": "ยังไม่มีข้อความเข้ามาให้ตอบ"}
    last = msgs[-1]
    chat = str(last.get("chat_id") or default_chat or "").strip()
    reply = (await _ai_reply(last.get("text") or "")).strip() or "ขอบคุณที่ทักมาค่ะ 🙏"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": reply})
    if r.status_code >= 400:
        return {"error": f"Telegram {r.status_code}: {r.text[:160]}"}
    from datetime import datetime, timezone, timedelta
    ts = datetime.now(tz=timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M")
    db.chat_msg_add("telegram", chat, "out", "บอท (AI)", reply, ts)
    return {"ok": True, "reply": reply}


@app.post("/api/chatbot/clear")
async def chatbot_clear() -> dict:
    db.chat_clear("telegram")
    return {"ok": True}


@app.get("/api/ads/audiences")
async def ads_audiences(account_id: int) -> dict:
    import asyncio
    acc = db.ads_get_account(int(account_id))
    if not acc:
        return {"error": "ไม่พบบัญชี", "audiences": []}
    try:
        auds = await asyncio.to_thread(ads.meta_list_audiences, acc["token"], acc["ad_account_id"])
    except Exception as exc:
        return {"error": _friendly_err(exc), "audiences": []}
    return {"audiences": [{"id": a.get("id"), "name": a.get("name"), "subtype": a.get("subtype"),
                           "count": a.get("approximate_count_lower_bound")} for a in auds]}


@app.post("/api/ads/audience/custom")
async def ads_audience_custom(p: dict) -> dict:
    """Create a Custom Audience from a pasted customer list (emails/phones are
    SHA256-hashed server-side — raw data is never stored)."""
    import asyncio
    acc = db.ads_get_account(int(p.get("account_id")))
    if not acc:
        return {"error": "ไม่พบบัญชี"}
    emails, phones = [], []
    for line in (p.get("data") or "").replace(",", "\n").splitlines():
        v = line.strip()
        if not v:
            continue
        if "@" in v:
            emails.append(v.lower())
        else:
            dig = "".join(c for c in v if c.isdigit())
            if dig.startswith("0"):
                dig = "66" + dig[1:]          # TH local → country code
            if dig:
                phones.append(dig)
    if not emails and not phones:
        return {"error": "ใส่อีเมลหรือเบอร์โทรอย่างน้อย 1 รายการ"}
    name = (p.get("name") or "G Office Custom").strip()
    try:
        aid = await asyncio.to_thread(ads.meta_create_custom_audience, acc["token"], acc["ad_account_id"], name)
        n = await asyncio.to_thread(ads.meta_add_audience_users, acc["token"], aid, emails, phones)
    except Exception as exc:
        return {"error": "สร้างไม่สำเร็จ: " + _friendly_err(exc)}
    return {"ok": True, "id": aid, "message": f"สร้าง Custom Audience “{name}” + เพิ่ม {n} รายชื่อแล้ว"}


@app.post("/api/ads/audience/lookalike")
async def ads_audience_lookalike(p: dict) -> dict:
    import asyncio
    acc = db.ads_get_account(int(p.get("account_id")))
    if not acc:
        return {"error": "ไม่พบบัญชี"}
    source = (p.get("source_id") or "").strip()
    if not source:
        return {"error": "เลือกกลุ่มต้นทาง (Custom Audience) ก่อน"}
    try:
        ratio = float(p.get("ratio") or 0.01)
    except (TypeError, ValueError):
        ratio = 0.01
    name = (p.get("name") or "G Office Lookalike").strip()
    try:
        aid = await asyncio.to_thread(ads.meta_create_lookalike, acc["token"], acc["ad_account_id"],
                                      name, source, (p.get("country") or "TH").strip(), ratio)
    except Exception as exc:
        return {"error": "สร้าง Lookalike ไม่สำเร็จ: " + _friendly_err(exc)}
    return {"ok": True, "id": aid, "message": f"สร้าง Lookalike “{name}” แล้ว (Meta กำลังประมวลผล ~ไม่กี่ชม.)"}


async def _ads_autopilot_loop() -> None:
    """Once a day, run Autopilot for accounts that turned on auto-run (and kill-switch)."""
    import asyncio
    await asyncio.sleep(60)
    while True:
        try:
            today = db._now()[:10]
            for acc in db.ads_list_accounts("meta"):
                cfg = _autopilot_cfg(acc)
                if cfg.get("mode") not in ("manage", "create"):
                    continue
                if not cfg.get("auto_daily"):
                    continue
                if cfg.get("last_run") == today:
                    continue
                try:
                    await _run_autopilot(acc)
                except Exception:
                    pass
                cfg["last_run"] = today
                db.ads_set_autopilot(acc["id"], cfg)
        except Exception:
            pass
        await asyncio.sleep(3600)     # check hourly; runs each account once/day


@app.post("/api/ads/settings")
async def ads_settings(p: dict) -> dict:
    db.set_settings({"ads_mode": "auto" if p.get("mode") == "auto" else "review"})
    return {"ok": True, "mode": _ads_mode()}


@app.post("/api/ads/review")
async def ads_review(p: dict = None) -> dict:
    acc = (p or {}).get("account_id")
    try:
        return {"ok": True, "recommendations": await run_ads_review(acc if acc not in (None, "", "all") else None)}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.get("/api/ads/recommendations")
async def ads_recos() -> dict:
    return {"recommendations": db.ads_list_reco(), "mode": _ads_mode()}


@app.post("/api/ads/apply")
async def ads_apply(p: dict) -> dict:
    reco = db.ads_get_reco(int(p.get("id")))
    if not reco:
        return {"error": "ไม่พบรายการ"}
    try:
        msg = await __import__("asyncio").to_thread(_apply_reco, reco)
        db.ads_set_reco_status(reco["id"], "applied")
        return {"ok": True, "message": msg}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


@app.post("/api/ads/dismiss")
async def ads_dismiss(p: dict) -> dict:
    db.ads_set_reco_status(int(p.get("id")), "dismissed")
    return {"ok": True}


_ADS_IMG_SYS = (
    "คุณเป็นผู้เชี่ยวชาญนโยบายโฆษณาของ Meta (Facebook/Instagram Advertising Policies) "
    "และนักการตลาดสายภาพ หน้าที่คุณคือตรวจรูปครีเอทีฟ 'ก่อนยิงแอด' ว่ามีโอกาสผ่านรีวิวของ Meta ไหม "
    "ดูความเสี่ยงเข้าข่ายเนื้อหาต้องห้าม เช่น การพนัน/คาสิโน/เดิมพัน, ยา/อาหารเสริมเกินจริง, "
    "อาวุธ, เนื้อหาผู้ใหญ่, คำกล่าวอ้างเกินจริง, before/after, ภาพร่างกายเชิงลบ, เนื้อหาหลอกลวง ฯลฯ "
    "และวิเคราะห์ว่าคนเห็นภาพนี้จะรู้สึกอย่างไร (อารมณ์/ความประทับใจแรก) "
    "ตอบเป็นภาษาไทยทั้งหมด และตอบกลับเป็น JSON อย่างเดียว ห้ามมีข้อความอื่นนอก JSON "
    'รูปแบบ: {"verdict":"ผ่าน|เสี่ยง|ไม่ผ่าน","pass_score":0-100,'
    '"gambling_risk":"ต่ำ|กลาง|สูง","policy_flags":["ข้อที่สุ่มเสี่ยง..."],'
    '"feeling":"คนเห็นภาพนี้จะรู้สึก...","suggestions":["ปรับแบบนี้เพื่อให้ผ่าน..."]}'
)


@app.post("/api/ads/analyze-image")
async def ads_analyze_image(p: dict) -> dict:
    """วิเคราะห์รูปครีเอทีฟก่อนยิงแอดด้วย Claude vision — เช็คความเสี่ยงนโยบาย Meta + อารมณ์ผู้ชม."""
    data = (p.get("image") or "").strip()
    media_type = (p.get("media_type") or "image/jpeg").strip()
    note = (p.get("note") or "").strip()
    if "," in data and data.startswith("data:"):      # เผื่อส่ง data URL มาเต็ม
        head, data = data.split(",", 1)
        if "image/" in head:
            media_type = head.split(":", 1)[1].split(";", 1)[0]
    if not data:
        return {"error": "ยังไม่ได้แนบรูป — อัปโหลดรูปก่อน"}
    prompt = "ตรวจรูปครีเอทีฟนี้ก่อนยิงแอด Facebook/Instagram"
    if note:
        prompt += f"\nบริบท/แคปชั่นที่จะใช้: {note}"
    try:
        resp = await _client.messages.create(
            model=get_model(), max_tokens=1100, **thinking_kwargs(),
            system=_ADS_IMG_SYS,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": prompt},
            ]}])
        txt = next((b.text for b in resp.content if b.type == "text"), "").strip()
        if txt.startswith("```"):
            txt = txt.split("```", 2)[1].lstrip("json").strip() if txt.count("```") >= 2 else txt.strip("`")
        import json as _json
        try:
            result = _json.loads(txt)
        except Exception:
            s, e = txt.find("{"), txt.rfind("}")
            result = _json.loads(txt[s:e + 1]) if s >= 0 and e > s else {"verdict": "เสี่ยง", "feeling": txt, "suggestions": []}
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"error": _friendly_err(exc)}


# ---- Public: click redirect + the public bio page (catch-all, keep last) ----
@app.get("/l/{link_id}")
async def bio_click(link_id: int, request: Request):
    link = db.bio_get_link(link_id)
    if not link:
        return RedirectResponse("/")
    db.bio_record_click(link_id, link["page_id"], _ip_hash(request), request.headers.get("referer", ""))
    return RedirectResponse(link["url"])


def _render_bio(page: dict, links: list) -> str:
    dark = page.get("theme") == "dark"
    bg = "#0b1020" if dark else "#f4f1ea"
    fg = "#e5e7eb" if dark else "#1f2937"
    sub = "#94a3b8" if dark else "#6b7280"
    card = "#161c2e" if dark else "#ffffff"
    avatar = page.get("avatar_url") or ""
    av = (f'<img src="{_esc(avatar)}" style="width:96px;height:96px;border-radius:50%;object-fit:cover;margin:0 auto 12px;display:block">'
          if avatar else
          f'<div style="width:96px;height:96px;border-radius:50%;margin:0 auto 12px;background:#6366f1;color:#fff;font-size:42px;display:flex;align-items:center;justify-content:center">{_esc((page.get("display_name") or "?")[:1])}</div>')
    btns = "".join(
        f'<a href="/l/{l["id"]}" style="display:block;background:{card};color:{fg};text-decoration:none;'
        f'padding:16px;border-radius:14px;margin:10px 0;font-weight:700;text-align:center;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.08)">{_esc(l["label"])}</a>'
        for l in links if l.get("is_active"))
    bio_html = f'<p style="color:{sub};margin:4px 0 18px">{_esc(page.get("bio"))}</p>' if page.get("bio") else ""
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(page.get("display_name"))}</title></head>
<body style="margin:0;background:{bg};font-family:-apple-system,Segoe UI,Roboto,'Noto Sans Thai',sans-serif;min-height:100vh">
<div style="max-width:520px;margin:0 auto;padding:48px 20px">
  {av}
  <h1 style="color:{fg};text-align:center;margin:0 0 4px;font-size:22px">{_esc(page.get("display_name"))}</h1>
  <div style="text-align:center">{bio_html}</div>
  {btns or f'<p style="color:{sub};text-align:center">ยังไม่มีลิงก์</p>'}
  <p style="text-align:center;color:{sub};font-size:12px;margin-top:28px">⚡ by G Office</p>
</div></body></html>"""


def _legal_page(title: str, body_html: str) -> HTMLResponse:
    company = db.get_settings().get("company_name") or "G Office"
    today = __import__("datetime").date.today().isoformat()
    base = _public_base_or_default()
    return HTMLResponse(f"""<!doctype html><html lang="th"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} · {_esc(company)}</title>
<style>
 body{{font-family:-apple-system,'Segoe UI',Roboto,'Noto Sans Thai',sans-serif;background:#0f172a;color:#e2e8f0;margin:0;line-height:1.8}}
 .wrap{{max-width:780px;margin:0 auto;padding:40px 22px 80px}}
 h1{{color:#fff;font-size:26px;margin:0 0 6px}} h2{{color:#a5b4fc;font-size:18px;margin:30px 0 8px}}
 .muted{{color:#64748b;font-size:13px}} a{{color:#818cf8}} ul{{padding-left:22px}} li{{margin:5px 0}}
 .top{{display:flex;gap:14px;font-size:14px;margin-bottom:26px}}
</style></head><body><div class="wrap">
 <div class="top"><a href="/privacy">นโยบายความเป็นส่วนตัว</a><a href="/terms">ข้อกำหนดการใช้งาน</a></div>
 <h1>{_esc(title)}</h1>
 <p class="muted">{_esc(company)} · ปรับปรุงล่าสุด {today} · {_esc(base)}</p>
 {body_html}
</div></body></html>""")


_PRIVACY_BODY = """
<p>นโยบายนี้อธิบายว่า {C} ("เรา") เก็บ ใช้ และปกป้องข้อมูลของผู้ใช้บริการแพลตฟอร์มของเราอย่างไร
รวมถึงเมื่อคุณเชื่อมต่อบัญชีภายนอก เช่น Facebook, Instagram, Google และบริการอื่น ๆ</p>
<h2>1. ข้อมูลที่เราเก็บ</h2>
<ul>
 <li><b>ข้อมูลบัญชีที่เชื่อมต่อ:</b> เมื่อคุณยินยอมเชื่อมบริการภายนอก เราเก็บโทเค็นการเข้าถึง (access token) และข้อมูลโปรไฟล์สาธารณะที่จำเป็นต่อการให้บริการเท่านั้น</li>
 <li><b>ข้อมูลเพจ/โฆษณา:</b> หากคุณให้สิทธิ์ เราเข้าถึงชื่อเพจ โพสต์ และข้อมูลแคมเปญโฆษณา เพื่อช่วยจัดการและวิเคราะห์ตามคำสั่งของคุณ</li>
 <li><b>เนื้อหาที่คุณอัปโหลด:</b> เช่น รูปภาพที่ส่งให้ AI วิเคราะห์ ใช้เพื่อประมวลผลตามที่คุณร้องขอเท่านั้น</li>
 <li><b>ข้อมูลการใช้งาน:</b> บันทึกการใช้งานพื้นฐานเพื่อความปลอดภัยและปรับปรุงบริการ</li>
</ul>
<h2>2. การใช้ข้อมูล</h2>
<p>เราใช้ข้อมูลเพื่อให้บริการตามที่คุณสั่งเท่านั้น เช่น โพสต์/ตอบข้อความ จัดการโฆษณา วิเคราะห์เนื้อหา
เราไม่ขายข้อมูลของคุณให้บุคคลที่สาม</p>
<h2>3. การแชร์ข้อมูลกับบุคคลที่สาม</h2>
<p>เราส่งข้อมูลไปยังผู้ให้บริการที่จำเป็นต่อการทำงาน เช่น Meta (Facebook/Instagram) ผ่าน API ทางการ
และผู้ให้บริการ AI เพื่อประมวลผลคำสั่งของคุณ ภายใต้ข้อกำหนดของผู้ให้บริการเหล่านั้น</p>
<h2>4. การเก็บรักษาและการลบข้อมูล</h2>
<p>คุณสามารถยกเลิกการเชื่อมต่อบัญชีได้ตลอดเวลาจากในแอป ซึ่งจะลบโทเค็นการเข้าถึงที่เก็บไว้
หากต้องการให้ลบข้อมูลทั้งหมด ติดต่อเราผ่านช่องทางในแอป เราจะดำเนินการภายใน 30 วัน</p>
<h2>5. ความปลอดภัย</h2>
<p>เราจัดเก็บโทเค็นและข้อมูลสำคัญอย่างปลอดภัย และเข้าถึงเฉพาะเท่าที่จำเป็นต่อการให้บริการ</p>
<h2>6. สิทธิ์ของคุณ</h2>
<p>คุณมีสิทธิ์เข้าถึง แก้ไข หรือขอลบข้อมูลส่วนบุคคลของคุณ และเพิกถอนความยินยอมได้ทุกเมื่อ</p>
<h2>7. การเปลี่ยนแปลงนโยบาย</h2>
<p>เราอาจปรับปรุงนโยบายนี้เป็นครั้งคราว โดยจะแสดงวันที่ปรับปรุงล่าสุดไว้ด้านบน</p>
"""

_TERMS_BODY = """
<p>ข้อกำหนดนี้ใช้กับการใช้บริการแพลตฟอร์มของ {C} ("บริการ") การใช้บริการถือว่าคุณยอมรับข้อกำหนดนี้</p>
<h2>1. การใช้งาน</h2>
<p>คุณต้องใช้บริการอย่างถูกกฎหมาย และปฏิบัติตามนโยบายของแพลตฟอร์มที่เชื่อมต่อ เช่น นโยบายโฆษณาและเนื้อหาของ Meta</p>
<h2>2. บัญชีที่เชื่อมต่อ</h2>
<p>คุณรับผิดชอบบัญชีภายนอกที่นำมาเชื่อม และการให้สิทธิ์ต่าง ๆ คุณยืนยันว่ามีสิทธิ์จัดการบัญชี/เพจ/บัญชีโฆษณาที่นำมาเชื่อม</p>
<h2>3. เนื้อหา</h2>
<p>เนื้อหาที่คุณสร้างหรือเผยแพร่ผ่านบริการเป็นความรับผิดชอบของคุณ
ฟีเจอร์ AI (เช่น การวิเคราะห์ภาพหรือคำแนะนำโฆษณา) เป็นเพียงตัวช่วย ไม่รับประกันผลลัพธ์หรือการอนุมัติจากแพลตฟอร์มภายนอก</p>
<h2>4. ข้อจำกัดความรับผิด</h2>
<p>บริการให้ "ตามสภาพ" เราไม่รับประกันว่าจะไม่มีข้อผิดพลาด และไม่รับผิดต่อความเสียหายที่เกิดจากการใช้บริการหรือบริการภายนอก</p>
<h2>5. การเปลี่ยนแปลง</h2>
<p>เราอาจปรับปรุงหรือยุติบริการ และอาจแก้ไขข้อกำหนดนี้ได้ โดยจะแจ้งวันที่ปรับปรุงล่าสุดไว้ด้านบน</p>
"""


def _public_base_or_default() -> str:
    return PUBLIC_BASE_URL or "https://iamceo.ai"


@app.get("/privacy")
async def privacy_page():
    c = db.get_settings().get("company_name") or "G Office"
    return _legal_page("นโยบายความเป็นส่วนตัว", _PRIVACY_BODY.replace("{C}", _esc(c)))


@app.get("/terms")
async def terms_page():
    c = db.get_settings().get("company_name") or "G Office"
    return _legal_page("ข้อกำหนดการใช้งาน", _TERMS_BODY.replace("{C}", _esc(c)))


@app.get("/{slug}")
async def bio_public(slug: str, request: Request):
    s = _norm_slug(slug)
    if s.lower() in _BIO_RESERVED:
        return RedirectResponse("/")
    page = db.bio_get_page_by_slug(s)
    if not page:
        return HTMLResponse("<!doctype html><meta charset='utf-8'><div style='font-family:sans-serif;text-align:center;padding:60px'>404 — ไม่พบหน้านี้</div>", status_code=404)
    db.bio_record_view(page["id"], _ip_hash(request), request.headers.get("user-agent", ""))
    return HTMLResponse(_render_bio(page, db.bio_list_links(page["id"], active_only=True)))


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
