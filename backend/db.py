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
            """
        )


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
        c.execute(
            "INSERT INTO token_usage (session_id, agent_name, input_tokens, "
            "output_tokens, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, agent_name, int(input_tokens or 0), int(output_tokens or 0), _now()),
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


# --- Settings (key/value, e.g. automation + posting prefs) ------------------
def get_settings() -> dict:
    with _conn() as c:
        return {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM settings").fetchall()}

def set_settings(d: dict) -> None:
    with _conn() as c:
        for k, v in d.items():
            c.execute("INSERT INTO settings (key, value) VALUES (?,?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(k), str(v)))


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
