"""CEO orchestrator.

Flow for a single goal:

    1. CEO plans  -> structured list of (agent_id, task) sub-tasks
    2. Specialists run concurrently, each streaming its output
    3. CEO synthesizes a single final deliverable from all the results

All progress is reported through an async ``emit`` callback so the API layer can
stream it to the dashboard over WebSocket. The orchestrator itself knows nothing
about transport.
"""

from __future__ import annotations

import json
import re
from typing import Awaitable, Callable

import anthropic

from .agents import get_agent, sub_agents

MODEL = "claude-opus-4-8"

def get_model() -> str:
    return MODEL

def set_model(m: str) -> None:
    """Switch the model used by all agents (e.g. to Haiku to cut token cost)."""
    global MODEL
    if m:
        MODEL = m

def thinking_kwargs(model: str | None = None) -> dict:
    """Adaptive thinking on Opus/Sonnet/Fable; omit on Haiku (not supported)."""
    m = model or MODEL
    return {} if m.startswith("claude-haiku") else {"thinking": {"type": "adaptive"}}

# Emit signature: emit(event_dict) -> awaitable
Emit = Callable[[dict], Awaitable[None]]


def _extract_subtasks(text: str, allowed: set | None = None) -> list[dict]:
    """Pull a {"subtasks": [...]} object out of the model's reply.

    The CEO is instructed to return only JSON, but we tolerate stray prose by
    grabbing the outermost ``{ ... }`` block. Agents not in ``allowed`` are dropped.
    """
    allowed = allowed if allowed is not None else set(sub_agents().keys())
    candidate = text.strip()
    if not candidate.startswith("{"):
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        candidate = match.group(0) if match else "{}"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    subtasks = []
    for st in data.get("subtasks", []):
        if isinstance(st, dict) and st.get("agent") in allowed and st.get("task"):
            subtasks.append({"agent": st["agent"], "task": st["task"]})
    return subtasks


async def _emit_usage(emit: Emit, agent: str, usage) -> None:
    if usage is None:
        return
    await emit({
        "type": "token_usage",
        "agent": agent,
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    })


class Orchestrator:
    def __init__(self, client_for, model_for=None):
        """``client_for`` maps an agent id -> AsyncAnthropic client; ``model_for``
        maps an agent id -> model id (per-agent model). A bare AsyncAnthropic may
        be passed for client_for (all agents use it)."""
        if isinstance(client_for, anthropic.AsyncAnthropic):
            self.client_for = lambda _aid, _c=client_for: _c
        else:
            self.client_for = client_for
        self.model_for = model_for or (lambda _aid: MODEL)
        self.mcp_for = (lambda _aid: [])   # set by the app to attach per-agent MCP servers

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """The CEO's client (used for ad-hoc calls)."""
        return self.client_for("ceo")

    # -- Step 1: planning ----------------------------------------------------
    async def plan(self, goal: str, emit: Emit, paused: set | None = None) -> list[dict]:
        paused = paused or set()
        ceo = get_agent("ceo")
        await emit({"type": "agent_status", "agent": "ceo", "status": "working"})
        await emit({"type": "log", "agent": "ceo", "text": "Breaking the goal into sub-tasks…"})

        available = [a for a in sub_agents().values() if a.id not in paused]
        if not available:
            await emit({"type": "agent_status", "agent": "ceo", "status": "idle"})
            await emit({"type": "error", "message": "ทุกเอเจนต์ถูกพักงานอยู่ — ไม่มีใครรับงานได้"})
            return []
        roster = "\n".join(f"- {a.id}: {a.name} — {a.title}" for a in available)
        valid = ", ".join(a.id for a in available)
        prompt = (
            f"Goal from the user:\n{goal}\n\n"
            f"Available specialists:\n{roster}\n\n"
            "Assign 2-5 focused sub-tasks to the specialists best suited to them.\n\n"
            "Respond with ONLY a JSON object and no other text, in this exact shape:\n"
            '{"subtasks": [{"agent": "<id>", "task": "<สิ่งที่ต้องทำ — เขียนเป็นภาษาไทย>"}]}\n'
            f'where "agent" is one of: {valid}. '
            'IMPORTANT: เขียนข้อความใน "task" เป็นภาษาไทยที่กระชับ เข้าใจง่าย (ชื่อเฉพาะ/โค้ดคงภาษาเดิมได้).'
        )

        _model = self.model_for("ceo")
        resp = await self.client_for("ceo").messages.create(
            model=_model,
            max_tokens=2000,
            **thinking_kwargs(_model),
            system=ceo.system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        subtasks = _extract_subtasks(text, {a.id for a in available})

        await _emit_usage(emit, "ceo", resp.usage)
        await emit({"type": "plan", "subtasks": subtasks})
        await emit({"type": "agent_status", "agent": "ceo", "status": "idle"})
        return subtasks

    # -- Step 2: one specialist ---------------------------------------------
    async def run_specialist(self, agent_id: str, task: str, goal: str, emit: Emit) -> str:
        agent = get_agent(agent_id)
        await emit({"type": "agent_status", "agent": agent_id, "status": "working"})

        prompt = (
            f"Overall company goal (for context): {goal}\n\n"
            f"Your assigned sub-task:\n{task}"
        )

        _model = self.model_for(agent_id)
        client = self.client_for(agent_id)
        mcp = self.mcp_for(agent_id)
        base = dict(model=_model, max_tokens=4000, system=agent.system,
                    messages=[{"role": "user", "content": prompt}], **thinking_kwargs(_model))

        async def _stream(use_mcp):
            buf: list[str] = []
            if use_mcp:
                cm = client.beta.messages.stream(**base, mcp_servers=mcp, betas=["mcp-client-2025-04-04"])
            else:
                cm = client.messages.stream(**base)
            async with cm as stream:
                async for chunk in stream.text_stream:
                    buf.append(chunk)
                    await emit({"type": "agent_output", "agent": agent_id, "chunk": chunk})
                fm = await stream.get_final_message()
            return "".join(buf), fm

        try:
            result, final_msg = await _stream(bool(mcp))
        except Exception as exc:
            if mcp:   # MCP unsupported/misconfigured → fall back to a normal run
                await emit({"type": "log", "agent": agent_id, "text": f"(MCP ใช้ไม่ได้ — รันแบบปกติ: {type(exc).__name__})"})
                result, final_msg = await _stream(False)
            else:
                raise
        await _emit_usage(emit, agent_id, final_msg.usage)
        await emit({"type": "agent_status", "agent": agent_id, "status": "done"})
        return result

    # -- Step 3: synthesis ---------------------------------------------------
    async def synthesize(self, goal: str, results: list[dict], emit: Emit) -> str:
        ceo = get_agent("ceo")
        await emit({"type": "agent_status", "agent": "ceo", "status": "working"})
        await emit({"type": "log", "agent": "ceo", "text": "Synthesizing the final deliverable…"})

        sections = []
        for r in results:
            agent = get_agent(r["agent"])
            sections.append(f"### {agent.name} — task: {r['task']}\n{r['result']}")
        body = "\n\n".join(sections)

        prompt = (
            f"User's goal:\n{goal}\n\n"
            f"Your specialists have completed their work:\n\n{body}\n\n"
            "Synthesize a single, unified executive deliverable for the user. "
            "Integrate the findings, resolve any conflicts, and end with clear "
            "recommended next steps."
        )

        _model = self.model_for("ceo")
        collected: list[str] = []
        async with self.client_for("ceo").messages.stream(
            model=_model,
            max_tokens=4000,
            **thinking_kwargs(_model),
            system=ceo.system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                collected.append(chunk)
                await emit({"type": "agent_output", "agent": "ceo", "chunk": chunk})
            final_msg = await stream.get_final_message()

        final = "".join(collected)
        await _emit_usage(emit, "ceo", final_msg.usage)
        await emit({"type": "agent_status", "agent": "ceo", "status": "done"})
        await emit({"type": "final", "output": final})
        return final

    # -- Full run ------------------------------------------------------------
    async def run(self, goal: str, emit: Emit, paused: set | None = None) -> None:
        import asyncio

        subtasks = await self.plan(goal, emit, paused)
        if not subtasks:
            await emit({"type": "error", "message": "The CEO produced no sub-tasks."})
            return

        # Run all specialists concurrently; each streams independently.
        async def _run(st: dict) -> dict:
            result = await self.run_specialist(st["agent"], st["task"], goal, emit)
            return {"agent": st["agent"], "task": st["task"], "result": result}

        results = await asyncio.gather(*(_run(st) for st in subtasks))

        await self.synthesize(goal, list(results), emit)
        await emit({"type": "done"})
