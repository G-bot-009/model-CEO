"""Ads Manager — read campaign metrics and apply actions (Meta + Google).

Meta (Facebook/Instagram) is implemented against the Graph API. Google Ads
needs a developer token + customer id (and Google review for production); its
fetch is left as an honest "not configured yet" until those are supplied —
no fake data.

All calls use stdlib urllib. Live calls can't be exercised in the sandbox
(outbound blocked); the request shapes follow each platform's documented API.
"""

from __future__ import annotations

import hashlib
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


def meta_account_info(token: str, ad_account_id: str) -> dict:
    """Real ad account name + business + status (for displaying which account is connected)."""
    acct = _acct(ad_account_id)
    d = _get(f"{GRAPH}/{acct}?fields=name,account_status,currency,business{{name}}&access_token={token}")
    return {"name": d.get("name", ""), "business": (d.get("business") or {}).get("name", ""),
            "currency": d.get("currency", ""), "status": d.get("account_status", "")}


def meta_pause(token: str, campaign_id: str) -> dict:
    return _post(f"{GRAPH}/{campaign_id}", {"status": "PAUSED", "access_token": token})


def meta_set_budget(token: str, campaign_id: str, daily_budget_thb: float) -> dict:
    # Graph API expects the smallest currency unit (satang for THB)
    cents = int(round(float(daily_budget_thb) * 100))
    return _post(f"{GRAPH}/{campaign_id}", {"daily_budget": cents, "access_token": token})


# Map a plain goal → Meta campaign objective (new ODAX objectives)
_GOAL_OBJECTIVE = {
    "sales": "OUTCOME_SALES", "messages": "OUTCOME_ENGAGEMENT",
    "leads": "OUTCOME_LEADS", "traffic": "OUTCOME_TRAFFIC", "website": "OUTCOME_TRAFFIC",
}


def meta_upload_image(token: str, ad_account_id: str, image_b64: str) -> str:
    """Upload an image (base64) to the ad account → return its image_hash."""
    acct = _acct(ad_account_id)
    d = _post(f"{GRAPH}/{acct}/adimages", {"bytes": image_b64, "access_token": token})
    imgs = d.get("images") or {}
    for v in imgs.values():
        if v.get("hash"):
            return v["hash"]
    raise RuntimeError("อัปโหลดรูปไม่สำเร็จ")


def meta_duplicate_campaign(token: str, campaign_id: str) -> dict:
    """Duplicate an existing (winning) campaign — created PAUSED."""
    return _post(f"{GRAPH}/{campaign_id}/copies",
                 {"deep_copy": "true", "status_option": "PAUSED", "access_token": token})


def _sha(s: str) -> str:
    return hashlib.sha256((s or "").strip().lower().encode()).hexdigest()


def meta_create_custom_audience(token: str, ad_account_id: str, name: str) -> str:
    """Create an empty Custom Audience (customer list). Returns its id."""
    acct = _acct(ad_account_id)
    d = _post(f"{GRAPH}/{acct}/customaudiences", {
        "name": name or "G Office Custom", "subtype": "CUSTOM",
        "customer_file_source": "USER_PROVIDED_ONLY",
        "description": "Created by G Office", "access_token": token})
    if not d.get("id"):
        raise RuntimeError(f"สร้าง Custom Audience ไม่สำเร็จ: {str(d)[:160]}")
    return d["id"]


def meta_add_audience_users(token: str, audience_id: str, emails: list, phones: list) -> int:
    """Add hashed customer data (SHA256) to a Custom Audience. Returns count added."""
    added = 0
    for schema, vals in (("EMAIL_SHA256", emails), ("PHONE_SHA256", phones)):
        vals = [v for v in (vals or []) if v]
        if not vals:
            continue
        payload = json.dumps({"schema": schema, "data": [_sha(v) for v in vals]})
        _post(f"{GRAPH}/{audience_id}/users", {"payload": payload, "access_token": token})
        added += len(vals)
    return added


def meta_create_lookalike(token: str, ad_account_id: str, name: str,
                          source_id: str, country: str = "TH", ratio: float = 0.01) -> str:
    """Create a Lookalike Audience from a source Custom Audience. Returns its id."""
    acct = _acct(ad_account_id)
    spec = json.dumps({"type": "similarity", "country": country or "TH", "ratio": ratio or 0.01})
    d = _post(f"{GRAPH}/{acct}/customaudiences", {
        "name": name or "G Office Lookalike", "subtype": "LOOKALIKE",
        "origin_audience_id": source_id, "lookalike_spec": spec, "access_token": token})
    if not d.get("id"):
        raise RuntimeError(f"สร้าง Lookalike ไม่สำเร็จ: {str(d)[:160]}")
    return d["id"]


def meta_list_audiences(token: str, ad_account_id: str) -> list:
    acct = _acct(ad_account_id)
    d = _get(f"{GRAPH}/{acct}/customaudiences?fields=name,subtype,approximate_count_lower_bound"
             f"&limit=200&access_token={token}")
    return d.get("data", [])


# --- Facebook Page management (Auto Post) -----------------------------------
def meta_get_pages(user_token: str) -> list:
    """Pages the user manages, each with its own Page Access Token."""
    d = _get(f"{GRAPH}/me/accounts?fields=name,access_token,id&limit=100&access_token={user_token}")
    return d.get("data", [])


def meta_page_post(page_token: str, page_id: str, message: str, link: str = "") -> dict:
    data = {"message": message or "", "access_token": page_token}
    if link:
        data["link"] = link
    return _post(f"{GRAPH}/{page_id}/feed", data)


def meta_page_photo(page_token: str, page_id: str, image_b64: str, message: str) -> dict:
    # post a photo with caption (image as base64 bytes)
    return _post(f"{GRAPH}/{page_id}/photos",
                 {"caption": message or "", "source_bytes": image_b64, "access_token": page_token})


def meta_page_recent_comments(page_token: str, page_id: str, limit: int = 5) -> list:
    """Recent posts + their unanswered comments (for AI auto-reply)."""
    posts = _get(f"{GRAPH}/{page_id}/posts?fields=message,created_time,"
                 f"comments.limit(10){{id,message,from,created_time}}&limit={limit}&access_token={page_token}")
    out = []
    for p in posts.get("data", []):
        for c in (p.get("comments", {}).get("data", []) or []):
            out.append({"post_id": p.get("id"), "post_msg": (p.get("message") or "")[:80],
                        "comment_id": c.get("id"), "text": c.get("message", ""),
                        "from": (c.get("from") or {}).get("name", "")})
    return out


def meta_reply_comment(page_token: str, comment_id: str, message: str) -> dict:
    return _post(f"{GRAPH}/{comment_id}/comments", {"message": message, "access_token": page_token})


def meta_ad_library(token: str, search_terms: str, countries=None,
                    ad_type: str = "ALL", limit: int = 24) -> list:
    """Search the public Ad Library (ads_archive). Note: Meta returns full data
    for political/issue ads; general commercial ads may be limited per region."""
    params = urllib.parse.urlencode({
        "search_terms": search_terms or "",
        "ad_reached_countries": json.dumps(countries or ["TH"]),
        "ad_type": ad_type or "ALL",
        "ad_active_status": "ALL",
        "fields": "id,page_name,ad_creative_bodies,ad_creative_link_titles,"
                  "ad_snapshot_url,ad_delivery_start_time,publisher_platforms",
        "limit": int(limit), "access_token": token,
    })
    d = _get(f"{GRAPH}/ads_archive?{params}", timeout=25)
    return d.get("data", [])


def _make_creative(acct, token, name, page_id, link, message, headline, image_hash):
    link_data = {"link": link, "message": message or "", "name": headline or ""}
    if image_hash:
        link_data["image_hash"] = image_hash
    story = json.dumps({"page_id": page_id, "link_data": link_data})
    cr = _post(f"{GRAPH}/{acct}/adcreatives",
               {"name": name, "object_story_spec": story, "access_token": token})
    if not cr.get("id"):
        raise RuntimeError(f"สร้างชิ้นงานไม่สำเร็จ: {str(cr)[:160]}")
    return cr["id"]


def meta_create_campaign(token: str, ad_account_id: str, name: str, goal: str,
                         daily_budget_thb: float, page_id: str, link: str,
                         message: str, headline: str, targeting: dict = None,
                         image_hash: str = "", variant_b: dict = None) -> dict:
    """Create a PAUSED campaign (campaign→ad set→creative→ad) for review.
    targeting: {age_min,age_max,genders:[1|2],countries:[..]}. variant_b: {message,headline}
    creates a 2nd ad for A/B testing. image_hash attaches an uploaded image."""
    if not (token and ad_account_id):
        raise RuntimeError("ต้องมี token + Ad Account ID")
    if not page_id:
        raise RuntimeError("ต้องมี Facebook Page ID สำหรับสร้างชิ้นงานโฆษณา")
    if not link:
        raise RuntimeError("ต้องมีลิงก์ปลายทาง (เว็บ/เพจ)")
    acct = _acct(ad_account_id)
    objective = _GOAL_OBJECTIVE.get((goal or "").lower(), "OUTCOME_TRAFFIC")
    cents = int(round(float(daily_budget_thb or 100) * 100))

    camp = _post(f"{GRAPH}/{acct}/campaigns", {
        "name": name or "G Office Autopilot", "objective": objective,
        "status": "PAUSED", "special_ad_categories": "[]", "access_token": token})
    cid = camp.get("id")
    if not cid:
        raise RuntimeError(f"สร้างแคมเปญไม่สำเร็จ: {str(camp)[:160]}")

    t = targeting or {}
    tgt = {"geo_locations": {"countries": t.get("countries") or ["TH"]},
           "age_min": int(t.get("age_min") or 18), "age_max": int(t.get("age_max") or 65)}
    if t.get("genders"):
        tgt["genders"] = t["genders"]
    adset = _post(f"{GRAPH}/{acct}/adsets", {
        "name": (name or "Autopilot") + " — Ad set", "campaign_id": cid,
        "daily_budget": cents, "billing_event": "IMPRESSIONS",
        "optimization_goal": "LINK_CLICKS", "bid_strategy": "LOWEST_COST_WITHOUT_CAP",
        "targeting": json.dumps(tgt), "status": "PAUSED", "access_token": token})
    asid = adset.get("id")
    if not asid:
        raise RuntimeError(f"สร้าง ad set ไม่สำเร็จ: {str(adset)[:160]}")

    crid = _make_creative(acct, token, (name or "Autopilot") + " — A", page_id, link, message, headline, image_hash)
    ad = _post(f"{GRAPH}/{acct}/ads", {
        "name": (name or "Autopilot") + " — Ad A", "adset_id": asid,
        "creative": json.dumps({"creative_id": crid}), "status": "PAUSED", "access_token": token})
    out = {"campaign_id": cid, "adset_id": asid, "creative_id": crid, "ad_id": ad.get("id"),
           "objective": objective, "status": "PAUSED"}
    if variant_b and (variant_b.get("message") or variant_b.get("headline")):   # A/B: 2nd ad
        cr2 = _make_creative(acct, token, (name or "Autopilot") + " — B", page_id, link,
                             variant_b.get("message"), variant_b.get("headline"), image_hash)
        ad2 = _post(f"{GRAPH}/{acct}/ads", {
            "name": (name or "Autopilot") + " — Ad B", "adset_id": asid,
            "creative": json.dumps({"creative_id": cr2}), "status": "PAUSED", "access_token": token})
        out["ad_b_id"] = ad2.get("id")
    return out


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
