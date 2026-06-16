# Multi-Agent Agentic OS

A dashboard where a team of AI agents work together autonomously. A **CEO**
agent receives a high-level goal, breaks it into sub-tasks, delegates each to a
specialist (**Researcher, CMO, Sales Rep, Developer, Data Analyst**), then
synthesizes a single deliverable — all streamed live to a web dashboard.

Built on the **Anthropic Claude API** (`claude-opus-4-8`, adaptive thinking) with
a **FastAPI + WebSocket** backend and a **vanilla HTML + Tailwind** frontend. No
external agent framework — just Python and the Anthropic SDK.

## Architecture

```
              ┌──────────────┐
  goal ─────► │  CEO (plan)  │  structured output → list of (agent, task)
              └──────┬───────┘
                     │  delegate (concurrent)
        ┌────────────┼─────────────┬───────────────┐
        ▼            ▼             ▼               ▼
   Researcher      CMO        Sales Rep     Dev / Data Analyst   (each streams)
        └────────────┴─────────────┴───────────────┘
                     │  results
              ┌──────▼───────┐
              │ CEO (synth)  │  ─────► final deliverable
              └──────────────┘
```

| File | Role |
|------|------|
| `backend/agents/registry.py` | Agent roles + system prompts |
| `backend/orchestrator.py`    | Plan → delegate → synthesize logic |
| `backend/db.py`              | SQLite persistence (sessions, messages, tasks, status) |
| `backend/main.py`            | FastAPI app + `/ws` WebSocket + static frontend |
| `frontend/index.html`        | Dashboard (sessions, agent status, chat box, live output) |
| `demo_cli.py`                | Standalone pure-Python demo (no web layer) |
| `start.command`              | Double-click launcher for macOS (no terminal typing) |

## Easiest start (macOS) — double-click

Double-click **`start.command`**. On the very first run it opens `.env` so you can
paste your `ANTHROPIC_API_KEY`; save, close, and double-click again. It then
installs dependencies, starts the server, and opens the dashboard automatically.
(If macOS blocks it: right-click → Open → Open.) See `วิธีใช้งาน.txt` for Thai
instructions.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

## Run the dashboard

```bash
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000, type a goal into the CEO command box (e.g.
*"Find our best leads and suggest an outreach strategy"*) and watch the agents
work. Each agent card shows **idle / working / done**; outputs stream in live.

## Run the CLI demo (CEO + Researcher + Sales Rep)

```bash
python demo_cli.py "Find our best leads and suggest an outreach strategy"
```

## Extending

- **Add an agent:** append an `Agent(...)` to `_AGENTS` in
  `backend/agents/registry.py`. It automatically appears in the roster, the
  planner's allowed delegates, and the dashboard.
- **Give agents tools:** add Anthropic tool definitions to a specialist's
  `messages.stream(...)` call in `orchestrator.py` and handle the tool-use loop.
- **Agent-to-agent delegation:** today the CEO does all delegation up front. To
  let specialists delegate, give them a `delegate` tool whose handler calls
  `Orchestrator.run_specialist`.

## Persistence (SQLite)

All activity is saved to `agents.db` (created automatically, standard-library
`sqlite3`, no extra install):

- **sessions** — each run-group; "New Session" starts a fresh one without
  deleting old data.
- **messages** — user directives, each agent's output, and the CEO's final answer.
- **tasks** — every delegated sub-task with its status and result.
- **agent_status** — the idle/working/done history.

On startup the dashboard continues the most recent session and replays its
history. Use the session dropdown to browse previous sessions.

## Notes

- Specialists run **concurrently** via `asyncio.gather`, so the dashboard shows
  several agents `working` at once.
- `agents.db` and `.env` are git-ignored (local data and secrets stay on your
  machine).
