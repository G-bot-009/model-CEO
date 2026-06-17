"""MCP OAuth 2.0 helper — discovery + dynamic client registration + PKCE.

Implements the Model Context Protocol authorization flow (the same one Claude
Desktop runs) so the user can click "เชื่อม" and authorize a connector in the
browser instead of pasting a token by hand. Uses only the Python standard
library (urllib) so there is no extra dependency.

All network calls go to the connector's own OAuth server — outbound HTTP must be
allowed for the live handshake to complete.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.parse
import urllib.request

_UA = "GOffice/1.0 (+mcp-oauth)"


def _get_json(url: str, timeout: int = 8) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post_form(url: str, data: dict, headers: dict | None = None, timeout: int = 10) -> dict:
    body = urllib.parse.urlencode(data).encode()
    h = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post_json(url: str, obj: dict, timeout: int = 10) -> dict:
    body = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Accept": "application/json", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def discover(mcp_url: str) -> dict:
    """Find the connector's OAuth server metadata via well-known discovery.

    Returns a dict with at least authorization_endpoint + token_endpoint
    (and registration_endpoint / scopes_supported when advertised).
    """
    p = urllib.parse.urlparse(mcp_url)
    origin = f"{p.scheme}://{p.netloc}"
    candidates = [
        origin + "/.well-known/oauth-authorization-server",
        origin + "/.well-known/openid-configuration",
    ]
    # The protected-resource document may point at a separate auth server.
    try:
        pr = _get_json(origin + "/.well-known/oauth-protected-resource")
        for a in (pr.get("authorization_servers") or []):
            candidates.insert(0, a.rstrip("/") + "/.well-known/oauth-authorization-server")
    except Exception:
        pass
    last = None
    for c in candidates:
        try:
            m = _get_json(c)
            if m.get("authorization_endpoint") and m.get("token_endpoint"):
                return m
        except Exception as e:
            last = e
    raise RuntimeError(f"ไม่พบ OAuth metadata ของเซิร์ฟเวอร์นี้ ({type(last).__name__ if last else 'no metadata'})")


def register_client(registration_endpoint: str, redirect_uri: str, client_name: str = "G Office") -> tuple:
    """Dynamic client registration (RFC 7591). Returns (client_id, client_secret)."""
    obj = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    r = _post_json(registration_endpoint, obj)
    return r.get("client_id"), r.get("client_secret")


def pkce() -> tuple:
    """Return (code_verifier, code_challenge) for S256 PKCE."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def build_authorize_url(meta: dict, client_id: str, redirect_uri: str, state: str,
                        challenge: str, scope: str | None = None, resource: str | None = None) -> str:
    params = {
        "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
        "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
    }
    if scope:
        params["scope"] = scope
    elif meta.get("scopes_supported"):
        params["scope"] = " ".join(meta["scopes_supported"])
    if resource:
        params["resource"] = resource
    sep = "&" if "?" in meta["authorization_endpoint"] else "?"
    return meta["authorization_endpoint"] + sep + urllib.parse.urlencode(params)


def _basic(client_id: str, client_secret: str) -> dict:
    raw = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return {"Authorization": "Basic " + raw}


def exchange_code(token_endpoint: str, code: str, redirect_uri: str, client_id: str,
                  verifier: str, client_secret: str = "") -> dict:
    data = {
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
        "client_id": client_id, "code_verifier": verifier,
    }
    headers = _basic(client_id, client_secret) if client_secret else None
    return _post_form(token_endpoint, data, headers)


def refresh(token_endpoint: str, refresh_token: str, client_id: str, client_secret: str = "") -> dict:
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": client_id}
    headers = _basic(client_id, client_secret) if client_secret else None
    return _post_form(token_endpoint, data, headers)
