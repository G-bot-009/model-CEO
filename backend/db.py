"""SQLite persistence for the Multi-Agent Agentic OS.

Uses only the standard-library ``sqlite3`` — no extra install. A fresh connection
is opened per call (cheap, and avoids cross-thread/async sharing issues). The DB
file lives next to the project as ``agents.db``.

Schema:
    sessions      (id, created_at, name)
    messages      (id, session_id, agent_name, role, content, timestamp)
    tasks         (id, session_id, agent_name, task, status, result,
                   created_at, updated_at)
    agent_status  (id, agent_name, status, timestamp)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

# DB path is overridable (e.g. a mounted volume in Docker) via AGENTS_DB.
DB_PATH = Path(os.getenv("AGENTS_DB") or (Path(__file__).resolve().parent.parent / "agents.db"))


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                name       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                agent_name TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                agent_name TEXT NOT NULL,
                task       TEXT NOT NULL,
                status     TEXT NOT NULL,
                result     TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS agent_status (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                status     TEXT NOT NULL,
                timestamp  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS token_usage (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    INTEGER NOT NULL,
                agent_name    TEXT NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                timestamp     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS routines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT NOT NULL, cadence TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER, agent_name TEXT, content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, decided_at TEXT
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS connectors (
                name TEXT PRIMARY KEY, status TEXT NOT NULL, config TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL, prompt TEXT NOT NULL, size TEXT NOT NULL,
                svg TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS response_cache (
                hash TEXT PRIMARY KEY, agent TEXT, prompt TEXT, answer TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS custom_agents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, title TEXT, emoji TEXT,
                color TEXT, tags TEXT, system TEXT, created_at TEXT NOT NULL,
                category TEXT, skills TEXT
            );
            CREATE TABLE IF NOT EXISTS social_oauth (
                platform TEXT PRIMARY KEY, client_id TEXT, client_secret TEXT,
                access_token TEXT, refresh_token TEXT, extra TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS social_inbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT, sender TEXT, text TEXT, kind TEXT,
                status TEXT, thread TEXT, reply TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tools (
                tool_id TEXT PRIMARY KEY, entry TEXT, owner_department TEXT,
                owner_agent TEXT, status TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS tool_runs (
                task_id TEXT PRIMARY KEY, tool_id TEXT, trace_id TEXT, status TEXT,
                inputs TEXT, outputs TEXT, error TEXT, created_at TEXT, finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY, name TEXT, mime TEXT, path TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_mcp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT, name TEXT, url TEXT, token TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS mcp_connections (
                conn_id TEXT PRIMARY KEY, name TEXT, url TEXT, token TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS mcp_oauth_clients (
                conn_id TEXT PRIMARY KEY, client_id TEXT, client_secret TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS trade_bots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, exchange TEXT, symbol TEXT, mode TEXT, status TEXT,
                config TEXT, state TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER, ts TEXT, kind TEXT, text TEXT, pnl REAL
            );
            CREATE TABLE IF NOT EXISTS planned_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT, topic TEXT, caption TEXT, media_kind TEXT, media_ref TEXT,
                platforms TEXT, status TEXT, scheduled_at TEXT, created_at TEXT,
                posted_at TEXT, result TEXT
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, template TEXT, brief TEXT, status TEXT,
                summary TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS project_stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER, idx INTEGER, name TEXT, agents TEXT,
                goal TEXT, status TEXT, result TEXT, updated_at TEXT, key_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS project_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER, summary TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS bio_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE, display_name TEXT, avatar_url TEXT,
                bio TEXT, theme TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS bio_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER, label TEXT, url TEXT, ord INTEGER,
                is_active INTEGER DEFAULT 1, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS bio_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER, visited_at TEXT, ip_hash TEXT, ua TEXT
            );
            CREATE TABLE IF NOT EXISTS bio_clicks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                link_id INTEGER, page_id INTEGER, clicked_at TEXT, ip_hash TEXT, referrer TEXT
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL, provider TEXT, base_url TEXT, secret TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_api_map (
                agent_id TEXT PRIMARY KEY, key_id INTEGER NOT NULL
            );
            """
        )
        # migrate older DBs that predate the skills columns
        cols = {r[1] for r in c.execute("PRAGMA table_info(custom_agents)").fetchall()}
        if "category" not in cols:
            c.execute("ALTER TABLE custom_agents ADD COLUMN category TEXT")
        if "skills" not in cols:
            c.execute("ALTER TABLE custom_agents ADD COLUMN skills TEXT")
        # per-key token attribution
        ucols = {r[1] for r in c.execute("PRAGMA table_info(token_usage)").fetchall()}
        if "key_id" not in ucols:
            c.execute("ALTER TABLE token_usage ADD COLUMN key_id INTEGER")
        # link agent_mcp rows back to a directory connector (NULL = manual entry)
        mcols = {r[1] for r in c.execute("PRAGMA table_info(agent_mcp)").fetchall()}
        if "conn_id" not in mcols:
            c.execute("ALTER TABLE agent_mcp ADD COLUMN conn_id TEXT")
        # per-stage API/engine override
        pscols = {r[1] for r in c.execute("PRAGMA table_info(project_stages)").fetchall()}
        if "key_id" not in pscols:
            c.execute("ALTER TABLE project_stages ADD COLUMN key_id INTEGER")
        # OAuth fields on directory connections (refresh, expiry, client creds)
        ccols = {r[1] for r in c.execute("PRAGMA table_info(mcp_connections)").fetchall()}
        for col in ("refresh_token", "token_url", "client_id", "client_secret"):
            if col not in ccols:
                c.execute(f"ALTER TABLE mcp_connections ADD COLUMN {col} TEXT")
        if "expires_at" not in ccols:
            c.execute("ALTER TABLE mcp_connections ADD COLUMN expires_at REAL")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- Sessions ---------------------------------------------------------------
def create_session(name: Optional[str] = None) -> dict:
    ts = _now()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO sessions (created_at, name) VALUES (?, ?)",
            (ts, name or ""),
        )
        sid = cur.lastrowid
        if not name:
            name = f"Session {sid}"
            c.execute("UPDATE sessions SET name = ? WHERE id = ?", (name, sid))
    return {"id": sid, "name": name, "created_at": ts}


def list_sessions() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, name, created_at FROM sessions ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: int) -> None:
    """Delete a session and everything attached to it."""
    with _conn() as c:
        for tbl in ("messages", "tasks", "token_usage"):
            c.execute(f"DELETE FROM {tbl} WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))


def latest_session() -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT id, name, created_at FROM sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_or_create_current() -> dict:
    """Continue the most recent session, or start the first one."""
    return latest_session() or create_session()


# --- Messages ---------------------------------------------------------------
def save_message(session_id: int, agent_name: str, role: str, content: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO messages (session_id, agent_name, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, agent_name, role, content, _now()),
        )
        return cur.lastrowid


# --- Tasks ------------------------------------------------------------------
def create_task(session_id: int, agent_name: str, task: str) -> int:
    ts = _now()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO tasks (session_id, agent_name, task, status, result, "
            "created_at, updated_at) VALUES (?, ?, ?, 'pending', NULL, ?, ?)",
            (session_id, agent_name, task, ts, ts),
        )
        return cur.lastrowid


def clear_tasks(session_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM tasks WHERE session_id=?", (session_id,))


def set_task_text(task_id: int, task: str) -> None:
    with _conn() as c:
        c.execute("UPDATE tasks SET task=? WHERE id=?", (task, task_id))


def list_session_tasks(session_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, agent_name, task, status FROM tasks WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    return [dict(r) for r in rows]


def update_task(task_id: int, status: Optional[str] = None, result: Optional[str] = None) -> None:
    with _conn() as c:
        if status is not None:
            c.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), task_id),
            )
        if result is not None:
            c.execute(
                "UPDATE tasks SET result = ?, updated_at = ? WHERE id = ?",
                (result, _now(), task_id),
            )


# --- Agent status -----------------------------------------------------------
def save_status(agent_name: str, status: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO agent_status (agent_name, status, timestamp) VALUES (?, ?, ?)",
            (agent_name, status, _now()),
        )


# --- Token usage ------------------------------------------------------------
def save_token_usage(session_id: int, agent_name: str, input_tokens: int, output_tokens: int) -> None:
    with _conn() as c:
        row = c.execute("SELECT key_id FROM agent_api_map WHERE agent_id=?", (agent_name,)).fetchone()
        key_id = row["key_id"] if row else None
        c.execute(
            "INSERT INTO token_usage (session_id, agent_name, input_tokens, "
            "output_tokens, timestamp, key_id) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, agent_name, int(input_tokens or 0), int(output_tokens or 0), _now(), key_id),
        )


def usage_today() -> dict:
    """Per-agent and total token usage for the current calendar day."""
    today = _now()[:10]
    with _conn() as c:
        rows = c.execute(
            "SELECT agent_name, SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens FROM token_usage "
            "WHERE substr(timestamp,1,10) = ? GROUP BY agent_name",
            (today,),
        ).fetchall()
    per_agent = {r["agent_name"]: {"input": r["input_tokens"], "output": r["output_tokens"]} for r in rows}
    total_in = sum(v["input"] for v in per_agent.values())
    total_out = sum(v["output"] for v in per_agent.values())
    return {"per_agent": per_agent, "input": total_in, "output": total_out}


def usage_series(days: int = 7) -> list[dict]:
    """Daily input/output totals, oldest→newest, for the last `days` days."""
    with _conn() as c:
        rows = c.execute(
            "SELECT substr(timestamp,1,10) AS day, SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens FROM token_usage "
            "GROUP BY day ORDER BY day DESC LIMIT ?",
            (days,),
        ).fetchall()
    series = [
        {"day": r["day"], "input": r["input_tokens"], "output": r["output_tokens"]}
        for r in rows
    ]
    series.reverse()
    return series


def usage_between(start_day: str, end_day: str) -> dict:
    """Per-agent totals + daily series for the inclusive date range [start_day, end_day]
    (both 'YYYY-MM-DD')."""
    with _conn() as c:
        rows = c.execute(
            "SELECT agent_name, SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens FROM token_usage "
            "WHERE substr(timestamp,1,10) BETWEEN ? AND ? GROUP BY agent_name",
            (start_day, end_day),
        ).fetchall()
        drows = c.execute(
            "SELECT substr(timestamp,1,10) AS day, SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens FROM token_usage "
            "WHERE substr(timestamp,1,10) BETWEEN ? AND ? GROUP BY day ORDER BY day",
            (start_day, end_day),
        ).fetchall()
    per_agent = {r["agent_name"]: {"input": r["input_tokens"], "output": r["output_tokens"]} for r in rows}
    series = [{"day": r["day"], "input": r["input_tokens"], "output": r["output_tokens"]} for r in drows]
    return {
        "per_agent": per_agent,
        "input": sum(v["input"] for v in per_agent.values()),
        "output": sum(v["output"] for v in per_agent.values()),
        "series": series,
    }


def usage_totals(where: str = "", params: tuple = ()) -> dict:
    """All-time totals, or scoped by an optional WHERE fragment."""
    sql = "SELECT SUM(input_tokens) AS i, SUM(output_tokens) AS o FROM token_usage"
    if where:
        sql += " WHERE " + where
    with _conn() as c:
        r = c.execute(sql, params).fetchone()
    return {"input": r["i"] or 0, "output": r["o"] or 0}


def usage_by_key(start_day: str, end_day: str) -> list[dict]:
    """Token totals grouped by API key for an inclusive date range.

    key_id is None for usage on the default/env key (or a per-connection BYOK key)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT key_id, SUM(input_tokens) AS i, SUM(output_tokens) AS o "
            "FROM token_usage WHERE substr(timestamp,1,10) BETWEEN ? AND ? GROUP BY key_id",
            (start_day, end_day),
        ).fetchall()
    return [{"key_id": r["key_id"], "input": r["i"] or 0, "output": r["o"] or 0} for r in rows]


# --- API keys (multi-key, per-agent assignment) -----------------------------
def add_api_key(label: str, provider: str, base_url: str, secret: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO api_keys (label, provider, base_url, secret, created_at) VALUES (?,?,?,?,?)",
            (label, provider, base_url, secret, _now()),
        )
        return cur.lastrowid


def list_api_keys() -> list[dict]:
    """Keys with the secret MASKED (never expose the full key to the client)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, label, provider, base_url, secret FROM api_keys ORDER BY id"
        ).fetchall()
    out = []
    for r in rows:
        s = r["secret"] or ""
        masked = (s[:7] + "…" + s[-4:]) if len(s) > 12 else ("•" * len(s) if s else "")
        out.append({"id": r["id"], "label": r["label"], "provider": r["provider"],
                    "base_url": r["base_url"] or "", "masked": masked, "has_secret": bool(s)})
    return out


def get_api_key(key_id: int) -> Optional[dict]:
    """Full row INCLUDING the secret — server-side use only (building a client)."""
    with _conn() as c:
        r = c.execute("SELECT id, label, provider, base_url, secret FROM api_keys WHERE id=?", (key_id,)).fetchone()
    return dict(r) if r else None


def delete_api_key(key_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM api_keys WHERE id=?", (key_id,))
        c.execute("DELETE FROM agent_api_map WHERE key_id=?", (key_id,))


def set_agent_key(agent_id: str, key_id: Optional[int]) -> None:
    with _conn() as c:
        if key_id:
            c.execute(
                "INSERT INTO agent_api_map (agent_id, key_id) VALUES (?,?) "
                "ON CONFLICT(agent_id) DO UPDATE SET key_id=excluded.key_id",
                (agent_id, key_id),
            )
        else:
            c.execute("DELETE FROM agent_api_map WHERE agent_id=?", (agent_id,))


def agent_key_map() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT agent_id, key_id FROM agent_api_map").fetchall()
    return {r["agent_id"]: r["key_id"] for r in rows}


def agent_key_id(agent_id: str) -> Optional[int]:
    with _conn() as c:
        r = c.execute("SELECT key_id FROM agent_api_map WHERE agent_id=?", (agent_id,)).fetchone()
    return r["key_id"] if r else None


# --- SOP library ------------------------------------------------------------
def add_sop(title: str, body: str) -> int:
    with _conn() as c:
        return c.execute("INSERT INTO sops (title, body, created_at) VALUES (?,?,?)",
                         (title, body, _now())).lastrowid

def list_sops() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT id, title, body, created_at FROM sops ORDER BY id DESC").fetchall()]


# --- Routines (recurring goals) ---------------------------------------------
def add_routine(goal: str, cadence: str) -> int:
    with _conn() as c:
        return c.execute("INSERT INTO routines (goal, cadence, created_at) VALUES (?,?,?)",
                         (goal, cadence, _now())).lastrowid

def list_routines() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT id, goal, cadence, created_at FROM routines ORDER BY id DESC").fetchall()]


# --- Approvals ---------------------------------------------------------------
def add_approval(session_id, agent_name: str, content: str) -> int:
    with _conn() as c:
        return c.execute("INSERT INTO approvals (session_id, agent_name, content, status, created_at) "
                         "VALUES (?,?,?, 'pending', ?)", (session_id, agent_name, content, _now())).lastrowid

def list_approvals(status: str = "pending") -> list[dict]:
    with _conn() as c:
        if status:
            rows = c.execute("SELECT * FROM approvals WHERE status=? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM approvals ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

def decide_approval(approval_id: int, status: str) -> None:
    with _conn() as c:
        c.execute("UPDATE approvals SET status=?, decided_at=? WHERE id=?", (status, _now(), approval_id))

def approvals_summary() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM approvals GROUP BY status").fetchall()
    d = {r["status"]: r["n"] for r in rows}
    return {"approved": d.get("approved",0), "rejected": d.get("rejected",0), "pending": d.get("pending",0)}


# --- Decision log ------------------------------------------------------------
def add_decision(kind: str, detail: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO decisions (kind, detail, created_at) VALUES (?,?,?)", (kind, detail, _now()))

def list_decisions(limit: int = 30) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT kind, detail, created_at FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

def decisions_count() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) n FROM decisions").fetchone()["n"]

def decisions_today() -> int:
    today = _now()[:10]
    with _conn() as c:
        return c.execute("SELECT COUNT(*) n FROM decisions WHERE substr(created_at,1,10)=?", (today,)).fetchone()["n"]


# --- Connectors --------------------------------------------------------------
def set_connector(name: str, status: str, config: str = "") -> None:
    with _conn() as c:
        c.execute("INSERT INTO connectors (name, status, config, updated_at) VALUES (?,?,?,?) "
                  "ON CONFLICT(name) DO UPDATE SET status=excluded.status, config=excluded.config, updated_at=excluded.updated_at",
                  (name, status, config, _now()))

def list_connectors() -> dict:
    with _conn() as c:
        return {r["name"]: r["status"] for r in c.execute("SELECT name, status FROM connectors").fetchall()}

def get_connector(name: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT status, config FROM connectors WHERE name=?", (name,)).fetchone()
    return dict(r) if r else None


# --- Social OAuth (BYO app per platform) ------------------------------------
def set_social_creds(platform: str, client_id: str, client_secret: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO social_oauth (platform, client_id, client_secret, updated_at) VALUES (?,?,?,?) "
                  "ON CONFLICT(platform) DO UPDATE SET client_id=excluded.client_id, "
                  "client_secret=excluded.client_secret, updated_at=excluded.updated_at",
                  (platform, client_id, client_secret, _now()))

def set_social_token(platform: str, access: str, refresh: str = "", extra: str = "") -> None:
    with _conn() as c:
        c.execute("UPDATE social_oauth SET access_token=?, refresh_token=?, extra=?, updated_at=? WHERE platform=?",
                  (access, refresh, extra, _now(), platform))

def get_social(platform: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT platform, client_id, client_secret, access_token, refresh_token, extra "
                      "FROM social_oauth WHERE platform=?", (platform,)).fetchone()
    return dict(r) if r else None

def list_social() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT platform, access_token FROM social_oauth").fetchall()
    return {r["platform"]: bool(r["access_token"]) for r in rows}

def delete_social(platform: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM social_oauth WHERE platform=?", (platform,))


# --- Tool Registry (G Office × n8n contract) --------------------------------
import json as _json

def upsert_tool(entry: dict) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO tools (tool_id, entry, owner_department, owner_agent, status, updated_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(tool_id) DO UPDATE SET entry=excluded.entry, "
            "owner_department=excluded.owner_department, owner_agent=excluded.owner_agent, "
            "status=excluded.status, updated_at=excluded.updated_at",
            (entry.get("tool_id"), _json.dumps(entry, ensure_ascii=False), entry.get("owner_department", ""),
             entry.get("owner_agent", ""), entry.get("status", "active"), _now()))

def list_tools() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT entry FROM tools ORDER BY tool_id").fetchall()
    out = []
    for r in rows:
        try: out.append(_json.loads(r["entry"]))
        except Exception: pass
    return out

def get_tool(tool_id: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT entry FROM tools WHERE tool_id=?", (tool_id,)).fetchone()
    if not r: return None
    try: return _json.loads(r["entry"])
    except Exception: return None

def delete_tool(tool_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM tools WHERE tool_id=?", (tool_id,))

def add_tool_run(task_id: str, tool_id: str, trace_id: str, inputs: dict) -> None:
    with _conn() as c:
        c.execute("INSERT INTO tool_runs (task_id, tool_id, trace_id, status, inputs, created_at) "
                  "VALUES (?,?,?,?,?,?)",
                  (task_id, tool_id, trace_id, "pending", _json.dumps(inputs, ensure_ascii=False), _now()))

def finish_tool_run(task_id: str, status: str, outputs=None, error=None) -> bool:
    with _conn() as c:
        cur = c.execute("UPDATE tool_runs SET status=?, outputs=?, error=?, finished_at=? WHERE task_id=?",
                        (status, _json.dumps(outputs or {}, ensure_ascii=False),
                         _json.dumps(error or {}, ensure_ascii=False), _now(), task_id))
        return cur.rowcount > 0

def list_tool_runs(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM tool_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# --- Shared Storage files (file_ref) ----------------------------------------
def add_file(fid: str, name: str, mime: str, path: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO files (id, name, mime, path, created_at) VALUES (?,?,?,?,?)",
                  (fid, name, mime, path, _now()))

def get_file(fid: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT id, name, mime, path FROM files WHERE id=?", (fid,)).fetchone()
    return dict(r) if r else None


# --- Per-agent MCP connectors -----------------------------------------------
def add_agent_mcp(agent_id: str, name: str, url: str, token: str, conn_id: str = "") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO agent_mcp (agent_id, name, url, token, conn_id, created_at) VALUES (?,?,?,?,?,?)",
            (agent_id, name, url, token, conn_id or None, _now()))
        return cur.lastrowid

def list_agent_mcp(agent_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, name, url, token, conn_id FROM agent_mcp WHERE agent_id=? ORDER BY id", (agent_id,)).fetchall()
    return [dict(r) for r in rows]

def delete_agent_mcp(mcp_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM agent_mcp WHERE id=?", (mcp_id,))

def agent_mcp_conn_ids(agent_id: str) -> set:
    """Directory connector ids currently enabled for this agent."""
    with _conn() as c:
        rows = c.execute("SELECT conn_id FROM agent_mcp WHERE agent_id=? AND conn_id IS NOT NULL", (agent_id,)).fetchall()
    return {r[0] for r in rows}

def delete_agent_mcp_conn(agent_id: str, conn_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM agent_mcp WHERE agent_id=? AND conn_id=?", (agent_id, conn_id))

def clear_agent_mcp_conn(conn_id: str) -> None:
    """Detach a directory connector from every agent (keeps the connection itself)."""
    with _conn() as c:
        c.execute("DELETE FROM agent_mcp WHERE conn_id=?", (conn_id,))

def conn_agent_ids(conn_id: str) -> set:
    """Agents that currently have this directory connector enabled."""
    with _conn() as c:
        rows = c.execute("SELECT agent_id FROM agent_mcp WHERE conn_id=?", (conn_id,)).fetchall()
    return {r[0] for r in rows}


# --- Directory connector credentials (connect once, reuse per agent) ---------
def connect_mcp(conn_id: str, name: str, url: str, token: str) -> None:
    """Manual-token connection (no refresh)."""
    with _conn() as c:
        c.execute(
            "INSERT INTO mcp_connections (conn_id, name, url, token, created_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(conn_id) DO UPDATE SET name=excluded.name, url=excluded.url, token=excluded.token, "
            "refresh_token=NULL, expires_at=NULL, token_url=NULL",
            (conn_id, name, url, token, _now()))

def connect_mcp_oauth(conn_id: str, name: str, url: str, token: str, refresh_token: str,
                      expires_at: float, token_url: str, client_id: str, client_secret: str) -> None:
    """OAuth connection with refresh material."""
    with _conn() as c:
        c.execute(
            "INSERT INTO mcp_connections (conn_id, name, url, token, refresh_token, expires_at, "
            "token_url, client_id, client_secret, created_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(conn_id) DO UPDATE SET name=excluded.name, url=excluded.url, token=excluded.token, "
            "refresh_token=excluded.refresh_token, expires_at=excluded.expires_at, token_url=excluded.token_url, "
            "client_id=excluded.client_id, client_secret=excluded.client_secret",
            (conn_id, name, url, token, refresh_token, expires_at, token_url, client_id, client_secret, _now()))

def update_mcp_tokens(conn_id: str, token: str, refresh_token: str, expires_at: float) -> None:
    """Store a freshly-refreshed access token."""
    with _conn() as c:
        c.execute("UPDATE mcp_connections SET token=?, refresh_token=COALESCE(NULLIF(?,''), refresh_token), "
                  "expires_at=? WHERE conn_id=?", (token, refresh_token, expires_at, conn_id))

_MCP_CONN_COLS = "conn_id, name, url, token, refresh_token, expires_at, token_url, client_id, client_secret"

def get_mcp_connection(conn_id: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute(f"SELECT {_MCP_CONN_COLS} FROM mcp_connections WHERE conn_id=?", (conn_id,)).fetchone()
    return dict(r) if r else None

def list_mcp_connections() -> list[dict]:
    with _conn() as c:
        rows = c.execute(f"SELECT {_MCP_CONN_COLS} FROM mcp_connections ORDER BY conn_id").fetchall()
    return [dict(r) for r in rows]

# Remember a dynamically-registered OAuth client so we don't re-register each time.
def save_mcp_oauth_client(conn_id: str, client_id: str, client_secret: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO mcp_oauth_clients (conn_id, client_id, client_secret, created_at) VALUES (?,?,?,?) "
                  "ON CONFLICT(conn_id) DO UPDATE SET client_id=excluded.client_id, client_secret=excluded.client_secret",
                  (conn_id, client_id, client_secret, _now()))

def get_mcp_oauth_client(conn_id: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT client_id, client_secret FROM mcp_oauth_clients WHERE conn_id=?", (conn_id,)).fetchone()
    return dict(r) if r else None

def connected_mcp_ids() -> set:
    with _conn() as c:
        rows = c.execute("SELECT conn_id FROM mcp_connections").fetchall()
    return {r[0] for r in rows}

def disconnect_mcp(conn_id: str) -> None:
    """Remove the connection and detach it from every agent that used it."""
    with _conn() as c:
        c.execute("DELETE FROM mcp_connections WHERE conn_id=?", (conn_id,))
        c.execute("DELETE FROM agent_mcp WHERE conn_id=?", (conn_id,))
        c.execute("DELETE FROM mcp_oauth_clients WHERE conn_id=?", (conn_id,))


# --- Social Inbox (unified comments/chats) ----------------------------------
def add_inbox(platform: str, sender: str, text: str, kind: str = "chat", thread: str = "") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO social_inbox (platform, sender, text, kind, status, thread, reply, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (platform, sender, text, kind, "pending", thread, "", _now()))
        return cur.lastrowid

def list_inbox(flt: str = "all", limit: int = 200) -> list[dict]:
    where = ""
    if flt == "comment": where = "WHERE kind='comment'"
    elif flt == "chat":  where = "WHERE kind='chat'"
    elif flt == "pending": where = "WHERE status='pending'"
    with _conn() as c:
        rows = c.execute(f"SELECT * FROM social_inbox {where} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

def get_inbox(item_id: int) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT * FROM social_inbox WHERE id=?", (item_id,)).fetchone()
    return dict(r) if r else None

def set_inbox_reply(item_id: int, reply: str) -> None:
    with _conn() as c:
        c.execute("UPDATE social_inbox SET reply=?, status='replied' WHERE id=?", (reply, item_id))

def inbox_counts() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT status, kind, COUNT(*) n FROM social_inbox GROUP BY status, kind").fetchall()
    total = sum(r["n"] for r in rows)
    pending = sum(r["n"] for r in rows if r["status"] == "pending")
    return {"total": total, "pending": pending}


# --- Custom agents (user-created, add/remove from the dashboard) -------------
def add_custom_agent(aid: str, name: str, title: str, emoji: str, color: str, tags: str,
                     system: str, category: str = "", skills: str = "") -> None:
    with _conn() as c:
        c.execute("INSERT INTO custom_agents (id, name, title, emoji, color, tags, system, category, skills, created_at) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                  "name=excluded.name, title=excluded.title, emoji=excluded.emoji, "
                  "color=excluded.color, tags=excluded.tags, system=excluded.system, "
                  "category=excluded.category, skills=excluded.skills",
                  (aid, name, title, emoji, color, tags, system, category, skills, _now()))

def list_custom_agents() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, name, title, emoji, color, tags, system, category, skills "
            "FROM custom_agents ORDER BY created_at").fetchall()]

def delete_custom_agent(aid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM custom_agents WHERE id=?", (aid,))


# --- Response cache (repeated question = 0 tokens) --------------------------
def cache_get(h: str):
    with _conn() as c:
        row = c.execute("SELECT answer FROM response_cache WHERE hash=?", (h,)).fetchone()
    return row["answer"] if row else None

def cache_set(h: str, agent: str, prompt: str, answer: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO response_cache (hash, agent, prompt, answer, created_at) VALUES (?,?,?,?,?) "
                  "ON CONFLICT(hash) DO UPDATE SET answer=excluded.answer, created_at=excluded.created_at",
                  (h, agent, prompt[:500], answer, _now()))

def cache_clear() -> None:
    with _conn() as c:
        c.execute("DELETE FROM response_cache")

def cache_count() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) n FROM response_cache").fetchone()["n"]


# --- Settings (key/value, e.g. automation + posting prefs) ------------------
def get_settings() -> dict:
    with _conn() as c:
        return {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM settings").fetchall()}

def set_settings(d: dict) -> None:
    with _conn() as c:
        for k, v in d.items():
            c.execute("INSERT INTO settings (key, value) VALUES (?,?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(k), str(v)))


# --- Pause / resume (per-agent + whole company) -----------------------------
def pause_state() -> dict:
    s = get_settings()
    agents = {k[len("paused_"):]: (v == "true") for k, v in s.items() if k.startswith("paused_")}
    return {"company": s.get("company_paused") == "true", "agents": agents}

def set_pause(target: str, paused: bool) -> None:
    key = "company_paused" if target == "company" else f"paused_{target}"
    set_settings({key: "true" if paused else "false"})


# --- Per-agent detail + summaries (day / week / month) ----------------------
def _period_expr(group: str, col: str) -> str:
    expr = f"replace({col},'T',' ')"
    if group == "week":  return f"strftime('%Y-W%W', {expr})"
    if group == "month": return f"substr({col},1,7)"
    return f"substr({col},1,10)"   # day

def agent_summary(agent_id: str, group: str = "day") -> list[dict]:
    limit = {"day": 14, "week": 8, "month": 6}.get(group, 14)
    pe_tok = _period_expr(group, "timestamp")
    pe_tsk = _period_expr(group, "created_at")
    with _conn() as c:
        tok = {r["p"]: (r["i"], r["o"]) for r in c.execute(
            f"SELECT {pe_tok} p, SUM(input_tokens) i, SUM(output_tokens) o "
            f"FROM token_usage WHERE agent_name=? GROUP BY p", (agent_id,)).fetchall()}
        tsk = {r["p"]: r["n"] for r in c.execute(
            f"SELECT {pe_tsk} p, COUNT(*) n FROM tasks WHERE agent_name=? GROUP BY p",
            (agent_id,)).fetchall()}
    periods = sorted(set(tok) | set(tsk))
    rows = [{"period": p, "tasks": tsk.get(p, 0),
             "input": tok.get(p, (0, 0))[0], "output": tok.get(p, (0, 0))[1]} for p in periods]
    return rows[-limit:]

def agent_works(agent_id: str, limit: int = 25) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT task, status, result, created_at FROM tasks WHERE agent_name=? "
            "ORDER BY id DESC LIMIT ?", (agent_id, limit)).fetchall()]

def add_image(agent_name: str, prompt: str, size: str, svg: str) -> int:
    with _conn() as c:
        return c.execute("INSERT INTO images (agent_name, prompt, size, svg, created_at) "
                         "VALUES (?,?,?,?,?)", (agent_name, prompt, size, svg, _now())).lastrowid

def list_images(agent_name: str, limit: int = 12) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, prompt, size, svg, created_at FROM images WHERE agent_name=? "
            "ORDER BY id DESC LIMIT ?", (agent_name, limit)).fetchall()]


def agent_totals(agent_id: str) -> dict:
    with _conn() as c:
        tasks = c.execute("SELECT COUNT(*) n FROM tasks WHERE agent_name=?", (agent_id,)).fetchone()["n"]
        tok = c.execute("SELECT COALESCE(SUM(input_tokens),0) i, COALESCE(SUM(output_tokens),0) o "
                        "FROM token_usage WHERE agent_name=?", (agent_id,)).fetchone()
    return {"tasks": tasks, "input": tok["i"], "output": tok["o"]}


# --- Workforce (tasks per agent, last 7 days) -------------------------------
def workforce() -> dict:
    cutoff = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
    with _conn() as c:
        rows = c.execute("SELECT agent_name, COUNT(*) n FROM tasks WHERE created_at >= ? GROUP BY agent_name",
                         (cutoff,)).fetchall()
    return {r["agent_name"]: r["n"] for r in rows}


# --- History (for replaying a session in the dashboard) ---------------------
def get_history(session_id: int) -> dict:
    with _conn() as c:
        messages = [
            dict(r)
            for r in c.execute(
                "SELECT agent_name, role, content, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]
        tasks = [
            dict(r)
            for r in c.execute(
                "SELECT id, agent_name, task, status, result FROM tasks "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]
    return {"messages": messages, "tasks": tasks}


def delete_task(task_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM tasks WHERE id=?", (task_id,))


# --- Trading bots -----------------------------------------------------------
import json as _json

def create_bot(name, exchange, symbol, mode, config: dict, state: dict) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO trade_bots (name, exchange, symbol, mode, status, config, state, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (name, exchange, symbol, mode, "stopped", _json.dumps(config), _json.dumps(state), _now()))
        return cur.lastrowid

def update_bot(bot_id, name, exchange, symbol, mode, config: dict) -> None:
    with _conn() as c:
        c.execute("UPDATE trade_bots SET name=?, exchange=?, symbol=?, mode=?, config=? WHERE id=?",
                  (name, exchange, symbol, mode, _json.dumps(config), bot_id))

def list_bots() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, name, exchange, symbol, mode, status, config, state FROM trade_bots ORDER BY id").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["config"] = _json.loads(d.get("config") or "{}")
        d["state"] = _json.loads(d.get("state") or "{}")
        out.append(d)
    return out

def get_bot(bot_id) -> "Optional[dict]":
    with _conn() as c:
        r = c.execute("SELECT id, name, exchange, symbol, mode, status, config, state FROM trade_bots WHERE id=?", (bot_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["config"] = _json.loads(d.get("config") or "{}")
    d["state"] = _json.loads(d.get("state") or "{}")
    return d

def set_bot_status(bot_id, status: str) -> None:
    with _conn() as c:
        c.execute("UPDATE trade_bots SET status=? WHERE id=?", (status, bot_id))

def set_bot_state(bot_id, state: dict) -> None:
    with _conn() as c:
        c.execute("UPDATE trade_bots SET state=? WHERE id=?", (_json.dumps(state), bot_id))

def delete_bot(bot_id) -> None:
    with _conn() as c:
        c.execute("DELETE FROM trade_bots WHERE id=?", (bot_id,))
        c.execute("DELETE FROM trade_log WHERE bot_id=?", (bot_id,))

def add_trade_log(bot_id, kind, text, pnl=0.0) -> None:
    with _conn() as c:
        c.execute("INSERT INTO trade_log (bot_id, ts, kind, text, pnl) VALUES (?,?,?,?,?)",
                  (bot_id, _now(), kind, text, pnl))

def list_trade_log(bot_id, limit=50) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT ts, kind, text, pnl FROM trade_log WHERE bot_id=? ORDER BY id DESC LIMIT ?",
                         (bot_id, limit)).fetchall()
    return [dict(r) for r in rows]


# --- Content factory --------------------------------------------------------
def get_content_rules() -> dict:
    v = get_settings().get("content_rules")
    if not v:
        return {}
    try:
        return _json.loads(v)
    except Exception:
        return {}

def set_content_rules(rules: dict) -> None:
    set_settings({"content_rules": _json.dumps(rules)})

def create_planned(kind, topic, caption, media_kind, media_ref, platforms, status, scheduled_at) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO planned_posts (kind, topic, caption, media_kind, media_ref, platforms, status, "
            "scheduled_at, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (kind, topic, caption, media_kind, media_ref, _json.dumps(platforms or []), status, scheduled_at, _now()))
        return cur.lastrowid

def list_planned(limit=60) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM planned_posts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["platforms"] = _json.loads(d.get("platforms") or "[]")
        except Exception:
            d["platforms"] = []
        out.append(d)
    return out

def get_planned(pid) -> "Optional[dict]":
    with _conn() as c:
        r = c.execute("SELECT * FROM planned_posts WHERE id=?", (pid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    try:
        d["platforms"] = _json.loads(d.get("platforms") or "[]")
    except Exception:
        d["platforms"] = []
    return d

def set_planned_status(pid, status, posted_at=None, result=None) -> None:
    with _conn() as c:
        c.execute("UPDATE planned_posts SET status=?, posted_at=COALESCE(?, posted_at), result=COALESCE(?, result) WHERE id=?",
                  (status, posted_at, result, pid))

def delete_planned(pid) -> None:
    with _conn() as c:
        c.execute("DELETE FROM planned_posts WHERE id=?", (pid,))

def count_planned_today() -> int:
    today = _now()[:10]
    with _conn() as c:
        r = c.execute("SELECT COUNT(*) FROM planned_posts WHERE substr(created_at,1,10)=?", (today,)).fetchone()
    return r[0] if r else 0

def count_planned_today_kind(media_kind: str) -> int:
    today = _now()[:10]
    with _conn() as c:
        r = c.execute("SELECT COUNT(*) FROM planned_posts WHERE substr(created_at,1,10)=? AND media_kind=?",
                      (today, media_kind)).fetchone()
    return r[0] if r else 0

def last_planned_iso() -> "Optional[str]":
    with _conn() as c:
        r = c.execute("SELECT created_at FROM planned_posts ORDER BY id DESC LIMIT 1").fetchone()
    return r[0] if r else None

def due_planned() -> list[dict]:
    """Queued posts whose scheduled time has arrived."""
    now = _now()
    with _conn() as c:
        rows = c.execute("SELECT * FROM planned_posts WHERE status='queued' AND (scheduled_at IS NULL OR scheduled_at<=?)",
                         (now,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["platforms"] = _json.loads(d.get("platforms") or "[]")
        except Exception:
            d["platforms"] = []
        out.append(d)
    return out


# --- Projects ---------------------------------------------------------------
def create_project(name, template, brief, stages: list) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO projects (name, template, brief, status, created_at) VALUES (?,?,?,?,?)",
                        (name, template, brief, "active", _now()))
        pid = cur.lastrowid
        for i, st in enumerate(stages):
            c.execute("INSERT INTO project_stages (project_id, idx, name, agents, goal, status, updated_at) "
                      "VALUES (?,?,?,?,?,?,?)",
                      (pid, i, st["name"], _json.dumps(st.get("agents") or []), st.get("goal", ""), "pending", _now()))
        return pid

def list_projects() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM projects ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            sts = c.execute("SELECT status FROM project_stages WHERE project_id=?", (d["id"],)).fetchall()
            d["stage_total"] = len(sts)
            d["stage_done"] = sum(1 for s in sts if s[0] == "done")
            out.append(d)
    return out

def get_project(pid) -> "Optional[dict]":
    with _conn() as c:
        r = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        rows = c.execute("SELECT * FROM project_stages WHERE project_id=? ORDER BY idx", (pid,)).fetchall()
    stages = []
    for s in rows:
        sd = dict(s)
        try:
            sd["agents"] = _json.loads(sd.get("agents") or "[]")
        except Exception:
            sd["agents"] = []
        stages.append(sd)
    d["stages"] = stages
    return d

def get_stage(stage_id) -> "Optional[dict]":
    with _conn() as c:
        r = c.execute("SELECT * FROM project_stages WHERE id=?", (stage_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    try:
        d["agents"] = _json.loads(d.get("agents") or "[]")
    except Exception:
        d["agents"] = []
    return d

def set_stage_result(stage_id, status, result) -> None:
    with _conn() as c:
        c.execute("UPDATE project_stages SET status=?, result=?, updated_at=? WHERE id=?",
                  (status, result, _now(), stage_id))

def set_project_summary(pid, summary) -> None:
    with _conn() as c:
        c.execute("UPDATE projects SET summary=? WHERE id=?", (summary, pid))
        c.execute("INSERT INTO project_summaries (project_id, summary, created_at) VALUES (?,?,?)",
                  (pid, summary, _now()))

def list_project_summaries(pid) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, summary, created_at FROM project_summaries WHERE project_id=? ORDER BY id DESC",
                         (pid,)).fetchall()
    return [dict(r) for r in rows]

def set_project_status(pid, status) -> None:
    with _conn() as c:
        c.execute("UPDATE projects SET status=? WHERE id=?", (status, pid))

def delete_project(pid) -> None:
    with _conn() as c:
        c.execute("DELETE FROM projects WHERE id=?", (pid,))
        c.execute("DELETE FROM project_stages WHERE project_id=?", (pid,))


def set_stage_engine(stage_id, key_id) -> None:
    with _conn() as c:
        c.execute("UPDATE project_stages SET key_id=? WHERE id=?", (key_id, stage_id))


# --- Bio Link Page ----------------------------------------------------------
def bio_create_page(slug, display_name, avatar_url, bio="", theme="light") -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO bio_pages (slug, display_name, avatar_url, bio, theme, created_at) "
                        "VALUES (?,?,?,?,?,?)", (slug, display_name, avatar_url, bio, theme, _now()))
        return cur.lastrowid

def bio_update_page(pid, slug, display_name, avatar_url, bio="", theme="light") -> None:
    with _conn() as c:
        c.execute("UPDATE bio_pages SET slug=?, display_name=?, avatar_url=?, bio=?, theme=? WHERE id=?",
                  (slug, display_name, avatar_url, bio, theme, pid))

def bio_list_pages() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM bio_pages ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

def bio_get_page(pid) -> "Optional[dict]":
    with _conn() as c:
        r = c.execute("SELECT * FROM bio_pages WHERE id=?", (pid,)).fetchone()
    return dict(r) if r else None

def bio_get_page_by_slug(slug) -> "Optional[dict]":
    with _conn() as c:
        r = c.execute("SELECT * FROM bio_pages WHERE slug=?", (slug,)).fetchone()
    return dict(r) if r else None

def bio_slug_taken(slug, exclude_id=None) -> bool:
    with _conn() as c:
        if exclude_id:
            r = c.execute("SELECT 1 FROM bio_pages WHERE slug=? AND id<>?", (slug, exclude_id)).fetchone()
        else:
            r = c.execute("SELECT 1 FROM bio_pages WHERE slug=?", (slug,)).fetchone()
    return bool(r)

def bio_delete_page(pid) -> None:
    with _conn() as c:
        c.execute("DELETE FROM bio_pages WHERE id=?", (pid,))
        c.execute("DELETE FROM bio_links WHERE page_id=?", (pid,))
        c.execute("DELETE FROM bio_views WHERE page_id=?", (pid,))
        c.execute("DELETE FROM bio_clicks WHERE page_id=?", (pid,))

def bio_add_link(page_id, label, url, ord=0, is_active=1) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO bio_links (page_id, label, url, ord, is_active, created_at) "
                        "VALUES (?,?,?,?,?,?)", (page_id, label, url, ord, is_active, _now()))
        return cur.lastrowid

def bio_update_link(lid, label, url, ord, is_active) -> None:
    with _conn() as c:
        c.execute("UPDATE bio_links SET label=?, url=?, ord=?, is_active=? WHERE id=?",
                  (label, url, ord, is_active, lid))

def bio_delete_link(lid) -> None:
    with _conn() as c:
        c.execute("DELETE FROM bio_links WHERE id=?", (lid,))
        c.execute("DELETE FROM bio_clicks WHERE link_id=?", (lid,))

def bio_list_links(page_id, active_only=False) -> list[dict]:
    q = "SELECT * FROM bio_links WHERE page_id=?"
    if active_only:
        q += " AND is_active=1"
    q += " ORDER BY ord, id"
    with _conn() as c:
        rows = c.execute(q, (page_id,)).fetchall()
    return [dict(r) for r in rows]

def bio_get_link(lid) -> "Optional[dict]":
    with _conn() as c:
        r = c.execute("SELECT * FROM bio_links WHERE id=?", (lid,)).fetchone()
    return dict(r) if r else None

def bio_record_view(page_id, ip_hash, ua) -> None:
    with _conn() as c:
        c.execute("INSERT INTO bio_views (page_id, visited_at, ip_hash, ua) VALUES (?,?,?,?)",
                  (page_id, _now(), ip_hash, (ua or "")[:200]))

def bio_record_click(link_id, page_id, ip_hash, referrer) -> None:
    with _conn() as c:
        c.execute("INSERT INTO bio_clicks (link_id, page_id, clicked_at, ip_hash, referrer) VALUES (?,?,?,?,?)",
                  (link_id, page_id, _now(), ip_hash, (referrer or "")[:300]))

def _since(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

def bio_analytics(page_id, days=90) -> dict:
    since = _since(days)
    with _conn() as c:
        views = c.execute("SELECT COUNT(*) FROM bio_views WHERE page_id=? AND visited_at>=?",
                          (page_id, since)).fetchone()[0]
        uniq = c.execute("SELECT COUNT(DISTINCT ip_hash) FROM bio_views WHERE page_id=? AND visited_at>=?",
                         (page_id, since)).fetchone()[0]
        clicks = c.execute("SELECT COUNT(*) FROM bio_clicks WHERE page_id=? AND clicked_at>=?",
                           (page_id, since)).fetchone()[0]
        daily = c.execute(
            "SELECT substr(clicked_at,1,10) d, COUNT(*) n FROM bio_clicks WHERE page_id=? AND clicked_at>=? "
            "GROUP BY d ORDER BY d", (page_id, since)).fetchall()
        daily_v = c.execute(
            "SELECT substr(visited_at,1,10) d, COUNT(*) n FROM bio_views WHERE page_id=? AND visited_at>=? "
            "GROUP BY d ORDER BY d", (page_id, since)).fetchall()
        per_link = c.execute(
            "SELECT l.id, l.label, l.url, COUNT(lc.id) clicks FROM bio_links l "
            "LEFT JOIN bio_clicks lc ON l.id=lc.link_id AND lc.clicked_at>=? "
            "WHERE l.page_id=? GROUP BY l.id ORDER BY clicks DESC", (since, page_id)).fetchall()
    return {
        "views": views, "unique": uniq, "clicks": clicks,
        "ctr": round(clicks / views * 100, 1) if views else 0.0,
        "daily_clicks": [dict(r) for r in daily],
        "daily_views": [dict(r) for r in daily_v],
        "per_link": [dict(r) for r in per_link],
    }
