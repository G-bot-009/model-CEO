"""Minimal multi-agent demo — pure Python + the Anthropic SDK, no web layer.

Demonstrates the core idea with three agents (CEO, Researcher, Sales Rep):
the CEO breaks a goal into sub-tasks, the specialists run independently, and the
CEO synthesizes the final output.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo_cli.py "Find our best leads and suggest an outreach strategy"
"""

from __future__ import annotations

import json
import re
import sys

import anthropic
from dotenv import load_dotenv

from backend.agents import get_agent

load_dotenv()

MODEL = "claude-opus-4-8"
DEMO_AGENTS = ["research", "marketing"]  # specialists used in this demo


def _extract_subtasks(text: str) -> list[dict]:
    candidate = text.strip()
    if not candidate.startswith("{"):
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        candidate = match.group(0) if match else "{}"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    return [
        {"agent": st["agent"], "task": st["task"]}
        for st in data.get("subtasks", [])
        if isinstance(st, dict) and st.get("agent") in DEMO_AGENTS and st.get("task")
    ]


def run_agent(client: anthropic.Anthropic, agent_id: str, prompt: str) -> str:
    agent = get_agent(agent_id)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=agent.system,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def main(goal: str) -> None:
    client = anthropic.Anthropic()
    ceo = get_agent("ceo")

    print(f"\n🧠 CEO is planning for goal: {goal!r}\n")
    roster = "\n".join(f"- {get_agent(a).id}: {get_agent(a).name} — {get_agent(a).title}" for a in DEMO_AGENTS)
    valid = ", ".join(DEMO_AGENTS)
    plan_resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=ceo.system,
        messages=[{"role": "user", "content": (
            f"Goal:\n{goal}\n\nAvailable specialists:\n{roster}\n\n"
            "Respond with ONLY a JSON object and no other text, in this shape:\n"
            '{"subtasks": [{"agent": "<id>", "task": "<what to do>"}]}\n'
            f'where "agent" is one of: {valid}.'
        )}],
    )
    plan_text = next((b.text for b in plan_resp.content if b.type == "text"), "{}")
    subtasks = _extract_subtasks(plan_text)

    results = []
    for st in subtasks:
        agent = get_agent(st["agent"])
        print(f"➡️  Delegating to {agent.name}: {st['task']}")
        out = run_agent(
            client, st["agent"],
            f"Overall goal (context): {goal}\n\nYour sub-task:\n{st['task']}",
        )
        print(f"\n--- {agent.name} result ---\n{out}\n")
        results.append({"agent": st["agent"], "task": st["task"], "result": out})

    print("🧠 CEO is synthesizing the final answer…\n")
    sections = "\n\n".join(
        f"### {get_agent(r['agent']).name} — {r['task']}\n{r['result']}" for r in results
    )
    final = run_agent(
        client, "ceo",
        f"User goal:\n{goal}\n\nSpecialist results:\n\n{sections}\n\n"
        "Synthesize one unified deliverable with clear recommended next steps.",
    )
    print("=" * 70)
    print("✅ FINAL DELIVERABLE\n")
    print(final)


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) or "Find our best leads and suggest an outreach strategy"
    main(goal)
