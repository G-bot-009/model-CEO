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

from .agents import SUB_AGENTS, get_agent

MODEL = "claude-opus-4-8"

# Emit signature: emit(event_dict) -> awaitable
Emit = Callable[[dict], Awaitable[None]]


def _extract_subtasks(text: str) -> list[dict]:
    """Pull a {"subtasks": [...]} object out of the model's reply.

    The CEO is instructed to return only JSON, but we tolerate stray prose by
    grabbing the outermost ``{ ... }`` block. Invalid agents are dropped.
    """
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
        if isinstance(st, dict) and st.get("agent") in SUB_AGENTS and st.get("task"):
            subtasks.append({"agent": st["agent"], "task": st["task"]})
    return subtasks


class Orchestrator:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    # -- Step 1: planning ----------------------------------------------------
    async def plan(self, goal: str, emit: Emit) -> list[dict]:
        ceo = get_agent("ceo")
        await emit({"type": "agent_status", "agent": "ceo", "status": "working"})
        await emit({"type": "log", "agent": "ceo", "text": "Breaking the goal into sub-tasks…"})

        roster = "\n".join(f"- {a.id}: {a.name} — {a.title}" for a in SUB_AGENTS.values())
        valid = ", ".join(SUB_AGENTS.keys())
        prompt = (
            f"Goal from the user:\n{goal}\n\n"
            f"Available specialists:\n{roster}\n\n"
            "Assign 2-5 focused sub-tasks to the specialists best suited to them.\n\n"
            "Respond with ONLY a JSON object and no other text, in this exact shape:\n"
            '{"subtasks": [{"agent": "<id>", "task": "<what to do>"}]}\n'
            f'where "agent" is one of: {valid}.'
        )

        resp = await self.client.messages.create(
            model=MODEL,
            max_tokens=2000,
            thinking={"type": "adaptive"},
            system=ceo.system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        subtasks = _extract_subtasks(text)

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

        collected: list[str] = []
        async with self.client.messages.stream(
            model=MODEL,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            system=agent.system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                collected.append(chunk)
                await emit({"type": "agent_output", "agent": agent_id, "chunk": chunk})

        result = "".join(collected)
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

        collected: list[str] = []
        async with self.client.messages.stream(
            model=MODEL,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            system=ceo.system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                collected.append(chunk)
                await emit({"type": "agent_output", "agent": "ceo", "chunk": chunk})

        final = "".join(collected)
        await emit({"type": "agent_status", "agent": "ceo", "status": "done"})
        await emit({"type": "final", "output": final})
        return final

    # -- Full run ------------------------------------------------------------
    async def run(self, goal: str, emit: Emit) -> None:
        import asyncio

        subtasks = await self.plan(goal, emit)
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
