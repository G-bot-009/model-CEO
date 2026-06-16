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
from datetime import datetime
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
