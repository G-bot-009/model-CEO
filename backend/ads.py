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


def google_campaigns(cfg: dict) -> list:
    """Google Ads needs a developer token + customer id + OAuth. Honest stub."""
    raise RuntimeError("Google Ads ยังต้องตั้งค่าเพิ่ม (developer token + customer id) — เชื่อมไว้ก่อน เร็วๆ นี้รองรับดึงข้อมูลอัตโนมัติ")
