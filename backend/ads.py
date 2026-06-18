"""Ads Manager — read campaign metrics and apply actions (Meta + Google).

Meta (Facebook/Instagram) is implemented against the Graph API. Google Ads
needs a developer token + customer id (and Google review for production); its
fetch is left as an honest "not configured yet" until those are supplied —
no fake data.

All calls use stdlib urllib. Live calls can't be exercised in the sandbox
(outbound blocked); the request shapes follow each platform's documented API.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

GRAPH = "https://graph.facebook.com/v19.0"
_UA = "GOffice-ads/1.0"


def _get(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url: str, data: dict, timeout: int = 20) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers={"User-Agent": _UA}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _acct(ad_account_id: str) -> str:
    a = (ad_account_id or "").strip()
    return a if a.startswith("act_") else "act_" + a


def _actions_value(actions, want) -> float:
    for a in (actions or []):
        if a.get("action_type") == want:
            try:
                return float(a.get("value", 0))
            except Exception:
                return 0.0
    return 0.0


def meta_campaigns(token: str, ad_account_id: str, date_preset: str = "last_7d") -> list:
    """Return [{id,name,status,daily_budget,spend,impressions,clicks,conversions,roas}]."""
    if not token or not ad_account_id:
        raise RuntimeError("ยังไม่ได้เชื่อม Meta Ads (token + Ad Account ID)")
    acct = _acct(ad_account_id)
    camps = _get(f"{GRAPH}/{acct}/campaigns?fields=name,status,daily_budget,objective&limit=100&access_token={token}")
    by_id = {c["id"]: c for c in camps.get("data", [])}
    ins = _get(f"{GRAPH}/{acct}/insights?level=campaign&date_preset={date_preset}"
               f"&fields=campaign_id,campaign_name,spend,impressions,clicks,actions,purchase_roas"
               f"&limit=200&access_token={token}")
    out = []
    seen = set()
    for r in ins.get("data", []):
        cid = r.get("campaign_id")
        seen.add(cid)
        base = by_id.get(cid, {})
        roas = 0.0
        if r.get("purchase_roas"):
            try:
                roas = float(r["purchase_roas"][0]["value"])
            except Exception:
                roas = 0.0
        out.append({
            "id": cid, "name": r.get("campaign_name") or base.get("name", ""),
            "status": base.get("status", ""),
            "daily_budget": (float(base.get("daily_budget", 0)) / 100.0) if base.get("daily_budget") else 0.0,
            "spend": float(r.get("spend", 0) or 0),
            "impressions": int(r.get("impressions", 0) or 0),
            "clicks": int(r.get("clicks", 0) or 0),
            "conversions": _actions_value(r.get("actions"), "purchase") or _actions_value(r.get("actions"), "lead"),
            "roas": roas,
        })
    # campaigns with no insight rows (no spend yet)
    for cid, base in by_id.items():
        if cid not in seen:
            out.append({"id": cid, "name": base.get("name", ""), "status": base.get("status", ""),
                        "daily_budget": (float(base.get("daily_budget", 0)) / 100.0) if base.get("daily_budget") else 0.0,
                        "spend": 0.0, "impressions": 0, "clicks": 0, "conversions": 0.0, "roas": 0.0})
    return out


def meta_pause(token: str, campaign_id: str) -> dict:
    return _post(f"{GRAPH}/{campaign_id}", {"status": "PAUSED", "access_token": token})


def meta_set_budget(token: str, campaign_id: str, daily_budget_thb: float) -> dict:
    # Graph API expects the smallest currency unit (satang for THB)
    cents = int(round(float(daily_budget_thb) * 100))
    return _post(f"{GRAPH}/{campaign_id}", {"daily_budget": cents, "access_token": token})


# ---------------------------------------------------------------------------
# Google Ads (REST API v17). Needs: developer_token, customer_id, and OAuth
# (either a raw access_token, or refresh_token + client_id + client_secret to
# auto-refresh). login_customer_id is optional (for manager/MCC accounts).
# ---------------------------------------------------------------------------
GADS = "https://googleads.googleapis.com/v17"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"


def _digits(s: str) -> str:
    return "".join(ch for ch in str(s or "") if ch.isdigit())


def _post_json(url: str, body: dict, headers: dict, timeout: int = 25) -> dict:
    data = json.dumps(body).encode()
    h = {"User-Agent": _UA, "Content-Type": "application/json", "Accept": "application/json"}
    h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _google_access_token(cfg: dict) -> str:
    tok = (cfg.get("access_token") or "").strip()
    if tok:
        return tok
    rt = (cfg.get("refresh_token") or "").strip()
    cid = (cfg.get("client_id") or "").strip()
    sec = (cfg.get("client_secret") or "").strip()
    if not (rt and cid and sec):
        raise RuntimeError("Google Ads ยังขาด OAuth — ใส่ access token หรือ (refresh token + client id + client secret)")
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token", "refresh_token": rt,
        "client_id": cid, "client_secret": sec}).encode()
    req = urllib.request.Request(GOOGLE_TOKEN, data=body, headers={"User-Agent": _UA}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        out = json.loads(resp.read().decode())
    at = out.get("access_token")
    if not at:
        raise RuntimeError("รีเฟรช Google access token ไม่สำเร็จ")
    return at


def _google_headers(cfg: dict, token: str) -> dict:
    h = {"Authorization": f"Bearer {token}",
         "developer-token": (cfg.get("developer_token") or "").strip()}
    login = _digits(cfg.get("login_customer_id") or "")
    if login:
        h["login-customer-id"] = login
    return h


def google_campaigns(cfg: dict) -> list:
    """Real Google Ads fetch via GAQL searchStream → same shape as meta_campaigns."""
    cid = _digits(cfg.get("customer_id"))
    if not cfg.get("developer_token") or not cid:
        raise RuntimeError("ยังไม่ได้ตั้งค่า Google Ads (developer token + customer id)")
    token = _google_access_token(cfg)
    query = ("SELECT campaign.id, campaign.name, campaign.status, "
             "campaign_budget.amount_micros, campaign_budget.resource_name, "
             "metrics.cost_micros, metrics.impressions, metrics.clicks, "
             "metrics.conversions, metrics.conversions_value "
             "FROM campaign WHERE segments.date DURING LAST_7_DAYS")
    res = _post_json(f"{GADS}/customers/{cid}/googleAds:searchStream",
                     {"query": query}, _google_headers(cfg, token))
    batches = res if isinstance(res, list) else [res]
    out = []
    for batch in batches:
        for row in (batch.get("results") or []):
            c = row.get("campaign", {})
            b = row.get("campaignBudget", {})
            m = row.get("metrics", {})
            spend = int(m.get("costMicros", 0) or 0) / 1e6
            conv_val = float(m.get("conversionsValue", 0) or 0)
            out.append({
                "id": str(c.get("id", "")), "name": c.get("name", ""),
                "status": c.get("status", ""),
                "daily_budget": int(b.get("amountMicros", 0) or 0) / 1e6,
                "budget_resource": b.get("resourceName", ""),
                "spend": spend,
                "impressions": int(m.get("impressions", 0) or 0),
                "clicks": int(m.get("clicks", 0) or 0),
                "conversions": float(m.get("conversions", 0) or 0),
                "roas": (conv_val / spend) if spend else 0.0,
            })
    return out


def google_pause(cfg: dict, campaign_id: str) -> dict:
    cid = _digits(cfg.get("customer_id"))
    token = _google_access_token(cfg)
    op = {"operations": [{"update": {
        "resourceName": f"customers/{cid}/campaigns/{_digits(campaign_id)}",
        "status": "PAUSED"}, "updateMask": "status"}]}
    return _post_json(f"{GADS}/customers/{cid}/campaigns:mutate", op, _google_headers(cfg, token))


def google_set_budget(cfg: dict, budget_resource: str, daily_budget_thb: float) -> dict:
    if not budget_resource:
        raise RuntimeError("ไม่พบ budget resource ของแคมเปญนี้")
    cid = _digits(cfg.get("customer_id"))
    token = _google_access_token(cfg)
    micros = int(round(float(daily_budget_thb) * 1e6))
    op = {"operations": [{"update": {
        "resourceName": budget_resource, "amountMicros": micros},
        "updateMask": "amount_micros"}]}
    return _post_json(f"{GADS}/customers/{cid}/campaignBudgets:mutate", op, _google_headers(cfg, token))
