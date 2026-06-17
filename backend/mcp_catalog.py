"""Curated directory of remote MCP connectors that agents can use via the Claude API.

Each entry has a documented remote MCP endpoint. A connector is only ever offered
to an agent when (a) its capabilities match that agent and (b) the user has actually
connected it (provided a token / secret_ref). Matching is keyword-based against the
agent's id, name, title, desc, tags and category.

URLs are sensible defaults — the connect dialog lets the user override the URL with
the exact one their provider hands them. Tokens are never stored in this file.
"""

from __future__ import annotations

# id, name, emoji, default url, Thai description, page to get the key, match keywords
CATALOG = [
    {
        "id": "notion", "oauth": True, "name": "Notion", "emoji": "📓",
        "url": "https://mcp.notion.com/mcp",
        "desc": "ค้นหา/อัปเดตหน้างานใน Notion workspace ของคุณ — โน้ต เอกสาร ฐานข้อมูล",
        "get": "https://www.notion.so/my-integrations",
        "match": ["admin", "research", "content", "ops_bot", "เอกสาร", "สรุป", "รีพอร์ต",
                  "วิจัย", "โน้ต", "แอดมิน", "workflow"],
    },
    {
        "id": "canva", "oauth": True, "name": "Canva", "emoji": "🎨",
        "url": "https://mcp.canva.com/mcp",
        "desc": "ค้นหา สร้าง autofill และ export งานออกแบบใน Canva — thumbnail โลโก้ แบนเนอร์",
        "get": "https://www.canva.com/developers/",
        "match": ["designer", "content", "ดีไซ", "กราฟิก", "thumbnail", "โลโก้",
                  "แบนเนอร์", "ออกแบบ", "คลิป", "วิดีโอ"],
    },
    {
        "id": "clickup", "oauth": True, "name": "ClickUp", "emoji": "✅",
        "url": "https://mcp.clickup.com/mcp",
        "desc": "สร้าง/อัปเดตงาน รายการ และเอกสารใน ClickUp — จัดการโปรเจกต์และงานประจำวัน",
        "get": "https://app.clickup.com/settings/apps",
        "match": ["ops_bot", "admin", "workflow", "งาน", "นัด", "โปรเจ", "ออโต", "sync"],
    },
    {
        "id": "asana", "oauth": True, "name": "Asana", "emoji": "📋",
        "url": "https://mcp.asana.com/sse",
        "desc": "จัดการงานและโปรเจกต์ใน Asana — มอบหมายงาน ติดตามสถานะ",
        "get": "https://app.asana.com/0/my-apps",
        "match": ["ops_bot", "admin", "workflow", "งาน", "โปรเจ", "ออโต"],
    },
    {
        "id": "linear", "oauth": True, "name": "Linear", "emoji": "📐",
        "url": "https://mcp.linear.app/sse",
        "desc": "จัดการ issue และรอบงานพัฒนาใน Linear — บั๊ก ฟีเจอร์ สถานะงานทีมดีฟ",
        "get": "https://linear.app/settings/api",
        "match": ["developer", "โค้ด", "api", "เว็บ", "แอป", "bug", "issue", "pr"],
    },
    {
        "id": "github", "name": "GitHub", "emoji": "🐙",
        "url": "https://api.githubcopilot.com/mcp/",
        "desc": "อ่าน/จัดการ repo, PR, issue บน GitHub — รีวิวโค้ดและติดตามงานพัฒนา",
        "get": "https://github.com/settings/tokens",
        "match": ["developer", "โค้ด", "pr", "commit", "api", "เว็บ", "แอป", "รีวิว"],
    },
    {
        "id": "atlassian", "oauth": True, "name": "Atlassian (Jira/Confluence)", "emoji": "🧩",
        "url": "https://mcp.atlassian.com/v1/sse",
        "desc": "เข้าถึง Jira และ Confluence จาก Claude — ตั๋วงาน เอกสารทีม",
        "get": "https://id.atlassian.com/manage-profile/security/api-tokens",
        "match": ["developer", "ops_bot", "jira", "confluence", "เอกสาร", "workflow"],
    },
    {
        "id": "sentry", "oauth": True, "name": "Sentry", "emoji": "🪲",
        "url": "https://mcp.sentry.dev/mcp",
        "desc": "ดู error และ issue ของแอปจาก Sentry — ช่วยดีบั๊กของจริง",
        "get": "https://sentry.io/settings/account/api/auth-tokens/",
        "match": ["developer", "bug", "error", "ดีบั๊ก", "api"],
    },
    {
        "id": "stripe", "name": "Stripe", "emoji": "💳",
        "url": "https://mcp.stripe.com",
        "desc": "ดูยอดขาย ลูกค้า และการชำระเงินใน Stripe — สรุปรายได้/ธุรกรรม",
        "get": "https://dashboard.stripe.com/apikeys",
        "match": ["marketing", "trader", "admin", "รายได้", "ชำระเงิน", "แอด", "payment"],
    },
    {
        "id": "intercom", "oauth": True, "name": "Intercom", "emoji": "💬",
        "url": "https://mcp.intercom.com/sse",
        "desc": "อ่าน/ตอบแชทลูกค้าใน Intercom — งานซัพพอร์ตและการตลาดสนทนา",
        "get": "https://app.intercom.com/a/apps/_/developer-hub",
        "match": ["admin", "marketing", "ลูกค้า", "แชท", "support", "อีเมล"],
    },
    {
        "id": "zapier", "name": "Zapier", "emoji": "⚡",
        "url": "https://mcp.zapier.com/api/mcp/mcp",
        "desc": "เชื่อมแอปนับพันผ่าน Zapier — ทริกเกอร์ ออโตเมชัน และซิงก์ข้อมูล",
        "get": "https://mcp.zapier.com/",
        "match": ["ops_bot", "workflow", "ออโต", "sync", "alert", "แจ้งเตือน"],
    },
]

CATALOG_BY_ID = {c["id"]: c for c in CATALOG}


def _agent_text(agent) -> str:
    """Build a lowercase searchable string from an Agent (dataclass or dict)."""
    g = (lambda k, d="": (agent.get(k, d) if isinstance(agent, dict) else getattr(agent, k, d)))
    tags = g("tags", ()) or ()
    if isinstance(tags, str):
        tags = tags.split(",")
    parts = [str(g("id")), str(g("name")), str(g("title")), str(g("desc")),
             str(g("category")), " ".join(tags)]
    return " ".join(parts).lower()


def matches(agent, entry) -> bool:
    """True when a catalog connector fits this agent's capabilities."""
    text = _agent_text(agent)
    return any(kw.lower() in text for kw in entry.get("match", []))


def for_agent(agent) -> list:
    """Catalog connectors whose capabilities match this agent."""
    return [c for c in CATALOG if matches(agent, c)]
