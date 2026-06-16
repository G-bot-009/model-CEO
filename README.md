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
| `backend/main.py`            | FastAPI app + `/ws` WebSocket + static frontend |
| `frontend/index.html`        | Dashboard (agent status, chat box, live output) |
| `demo_cli.py`                | Standalone pure-Python demo (no web layer) |

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

## Notes

- The API is stateless per goal; each run plans fresh. Conversation memory could
  be added by persisting prior `messages`.
- Specialists run **concurrently** via `asyncio.gather`, so the dashboard shows
  several agents `working` at once.
