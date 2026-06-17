"""Skill library loader.

The "Claude Skills Ultimate Bundle" is shipped as two JSON files under
``backend/skills/``:

* ``catalog.json`` — lightweight metadata (categories -> skills with id/name/desc)
  used by the dashboard's agent-creation picker.
* ``data.json``    — the full SKILL.md body for every skill, keyed by skill id.
  Used server-side only, to bake selected skills into a new agent's system
  prompt at creation time.

Skill ids look like ``"<category>:<skill-slug>"`` e.g. ``"ads-paid-media:ad-copy"``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

_DIR = os.path.join(os.path.dirname(__file__), "skills")

# Per-skill / total caps so an agent loaded with many skills can't blow up the
# token budget on every call. The skill body is trimmed, the description is kept.
_PER_SKILL_CHARS = 6000
_TOTAL_CHARS = 30000


@lru_cache(maxsize=1)
def _catalog() -> list[dict]:
    path = os.path.join(_DIR, "catalog.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _data() -> dict:
    path = os.path.join(_DIR, "data.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def catalog() -> list[dict]:
    """Categories + skill metadata for the picker (no heavy bodies)."""
    return _catalog()


@lru_cache(maxsize=1)
def _th_index() -> dict:
    """skill_id -> Thai name, from the catalog."""
    idx = {}
    for c in _catalog():
        for s in c.get("skills", []):
            if s.get("th"):
                idx[s["id"]] = s["th"]
    return idx


def skill_meta(skill_id: str) -> dict | None:
    d = _data().get(skill_id)
    if not d:
        return None
    return {
        "id": skill_id,
        "title": d.get("title", skill_id),
        "th": _th_index().get(skill_id, d.get("title", skill_id)),
        "desc": d.get("desc", ""),
    }


def valid_skill_ids(ids: list[str]) -> list[str]:
    data = _data()
    return [i for i in ids if i in data]


def compose_system(name: str, role: str, footer: str, skill_ids: list[str]) -> str:
    """Build a system prompt for a custom agent, embedding chosen skills.

    ``footer`` is the shared specialist footer (Thai-output rule etc.).
    """
    parts = [f"You are {name}, {role}."]
    ids = valid_skill_ids(skill_ids)
    if ids:
        data = _data()
        names = ", ".join(data[i]["title"] for i in ids)
        parts.append(
            f"\n\nYou are equipped with {len(ids)} professional skill(s): {names}. "
            "When a task matches one of these skills, follow that skill's method "
            "exactly — its phases, templates, and quality bar. Below are the full "
            "playbooks for each of your skills:\n"
        )
        budget = _TOTAL_CHARS
        for i in ids:
            body = (data[i].get("body") or "").strip()
            if not body:
                continue
            body = body[:_PER_SKILL_CHARS]
            block = f"\n===== SKILL: {data[i]['title']} =====\n{body}\n"
            if budget - len(block) < 0:
                break
            budget -= len(block)
            parts.append(block)
    return "".join(parts) + footer
