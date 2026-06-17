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

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "agents.db"


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
                "SELECT agent_name, task, status, result FROM tasks "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]
    return {"messages": messages, "tasks": tasks}
