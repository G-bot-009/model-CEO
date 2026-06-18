"""Multi-stage project templates.

A project bundles several stages into one unit of work; each stage is run by
specific agent(s) and its output is stored, so the user can open one project
and see the whole pipeline + a CEO summary in one place. The Analyst agent
closes the loop (measure → analyze → improve).

Stage goals use {brief} — the project's one-line description.
"""

from __future__ import annotations

TEMPLATES = {
    "seo": {
        "name": "Project SEO",
        "emoji": "🔍",
        "desc": "ทำ SEO ครบวงจร ตั้งแต่วิจัยคีย์เวิร์ดจนวัดผล",
        "stages": [
            {"name": "1. Research", "agents": ["research"],
             "goal": "วิจัยคีย์เวิร์ด กลุ่มเป้าหมาย และคู่แข่งสำหรับ: {brief}. สรุปคีย์เวิร์ดหลัก/รอง, search intent และช่องว่างของคู่แข่ง"},
            {"name": "2. Strategy & Planning", "agents": ["ceo", "marketing"],
             "goal": "จากผลวิจัย วางกลยุทธ์ SEO + แผนคอนเทนต์/โครงสร้างเว็บ + KPI สำหรับ: {brief}"},
            {"name": "3. Content Creation", "agents": ["content", "designer"],
             "goal": "เขียนคอนเทนต์ที่ปรับ SEO (title, meta description, H1-H2, เนื้อหา) + ไอเดียภาพประกอบ สำหรับ: {brief}"},
            {"name": "4. Technical SEO", "agents": ["developer"],
             "goal": "ตรวจ/แนะนำ Technical SEO: ความเร็ว, schema, sitemap, internal link, mobile, การ index สำหรับ: {brief}"},
            {"name": "5. Publish & Distribute", "agents": ["marketing", "admin"],
             "goal": "แผนเผยแพร่และกระจายคอนเทนต์ (on-page, backlink, social, อีเมล) + ออโตเมชัน n8n สำหรับ: {brief}"},
            {"name": "6. Monitor & Analyze", "agents": ["analyst"],
             "goal": "วางระบบวัดผล (Google Search Console / GA4), เมตริกที่ต้องดู และวิธีปรับปรุงต่อเนื่อง (ปิดลูป) สำหรับ: {brief}"},
        ],
    },
    "ads_fb": {
        "name": "Project Ads Facebook",
        "emoji": "📣",
        "desc": "ยิงแอด Facebook/Instagram ครบวงจร",
        "stages": [
            {"name": "1. Research", "agents": ["research"],
             "goal": "วิจัยกลุ่มเป้าหมาย ความสนใจ คู่แข่ง และ angle สำหรับแอด: {brief}"},
            {"name": "2. Strategy & Funnel", "agents": ["ceo", "marketing"],
             "goal": "วางโครงสร้างแคมเปญ/ad set/งบ/ฟันเนล/KPI (CPA, ROAS) สำหรับ: {brief}"},
            {"name": "3. Creative", "agents": ["content", "designer"],
             "goal": "เขียน ad copy หลายแบบ + hook + ไอเดียภาพ/วิดีโอโฆษณา สำหรับ: {brief}"},
            {"name": "4. Setup & Tracking", "agents": ["developer", "admin"],
             "goal": "ตั้ง Pixel/Conversions API, UTM, และเหตุการณ์ที่ต้องวัดผล สำหรับ: {brief}"},
            {"name": "5. Launch & Distribute", "agents": ["marketing"],
             "goal": "แผนปล่อยแอด, ตารางเวลา, การทดสอบ A/B และออโตเมชัน สำหรับ: {brief}"},
            {"name": "6. Monitor & Optimize", "agents": ["analyst"],
             "goal": "วิเคราะห์ผล (CTR/CPC/CPA/ROAS), สรุปสิ่งที่ควรหยุด/สเกล และปรับปรุง (ปิดลูป) สำหรับ: {brief}"},
        ],
    },
}


def template_list() -> list:
    return [{"id": k, "name": v["name"], "emoji": v["emoji"], "desc": v["desc"],
             "stages": [{"name": s["name"], "agents": s["agents"]} for s in v["stages"]]}
            for k, v in TEMPLATES.items()]
