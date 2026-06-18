"""Autonomous content-factory logic for G Office.

Pure, testable helpers: the adjustable rule set, daily-quota + min-gap +
posting-window scheduling decisions, the Claude model mapping for the "brain"
toggle, and the prompt the CEO uses to produce one social post per cycle.

The actual Claude / external-media calls live in main.py + media.py; this module
holds no network code so its scheduling rules can be unit-tested offline.
"""

from __future__ import annotations

import time

DEFAULT_RULES = {
    "team_on": True,         # ทีมทำงานเอง (ปิด = ร่างแต่ไม่โพสต์เอง)
    "post_mode": "draft",    # draft = รออนุมัติ · auto = โพสต์เลย
    "brain": "flash",        # flash (ถูก/เร็ว) · pro (ฉลาด ใช้ token มาก)
    "posts_per_day": 3,      # 0 = ให้ AI ตัดสิน (เพดานนุ่ม)
    "media_per_day": 3,      # 0 = ปิดโพสต์รูป/ข้อความอัตโนมัติ
    "videos_per_day": 1,
    "window": "peak",        # peak = ช่วงคนเล่นเยอะ · ai = ตัดสินเอง (ตลอด)
    "min_gap_hours": 3.0,    # กันโพสต์ติดกัน
}

_BRAIN_MODEL = {"flash": "claude-haiku-4-5", "pro": "claude-opus-4-8"}

# Bangkok "peak" hours (local) when window == "peak"
PEAK_HOURS = set(list(range(11, 14)) + list(range(17, 22)))   # 11–13, 17–21


def normalize_rules(r: dict) -> dict:
    out = dict(DEFAULT_RULES)
    for k, v in (r or {}).items():
        if k in DEFAULT_RULES:
            out[k] = v
    out["team_on"] = bool(out["team_on"])
    out["post_mode"] = "auto" if str(out["post_mode"]).lower() == "auto" else "draft"
    out["brain"] = "pro" if str(out["brain"]).lower() == "pro" else "flash"
    out["window"] = "ai" if str(out["window"]).lower() == "ai" else "peak"
    for k in ("posts_per_day", "media_per_day", "videos_per_day"):
        try:
            out[k] = max(0, int(out[k]))
        except Exception:
            out[k] = DEFAULT_RULES[k]
    try:
        out["min_gap_hours"] = max(0.0, float(out["min_gap_hours"]))
    except Exception:
        out["min_gap_hours"] = 3.0
    return out


def brain_model(rules: dict) -> str:
    return _BRAIN_MODEL.get(normalize_rules(rules)["brain"], "claude-haiku-4-5")


def in_window(rules: dict, now: float | None = None) -> bool:
    rules = normalize_rules(rules)
    if rules["window"] == "ai":
        return True
    hour = int(time.strftime("%H", time.localtime(now if now is not None else time.time())))
    return hour in PEAK_HOURS


def _hours_since(ts_iso: str | None, now: float) -> float:
    if not ts_iso:
        return 1e9
    try:
        t = time.mktime(time.strptime(ts_iso[:19], "%Y-%m-%dT%H:%M:%S"))
        return (now - t) / 3600.0
    except Exception:
        return 1e9


def should_produce(rules: dict, posts_today: int, last_post_iso: str | None,
                   now: float | None = None) -> tuple:
    """Decide whether the factory should create another post this cycle.

    Returns (ok: bool, reason: str). Pure — counts/last-time are passed in.
    """
    rules = normalize_rules(rules)
    now = now if now is not None else time.time()
    if not rules["team_on"]:
        return False, "ทีมถูกปิดอยู่ (team_on = false)"
    cap = rules["posts_per_day"] or rules["media_per_day"]
    if rules["media_per_day"] == 0:
        return False, "ปิดโพสต์รูป/ข้อความอัตโนมัติ (media_per_day = 0)"
    if cap and posts_today >= cap:
        return False, f"ครบโควต้าวันนี้แล้ว ({posts_today}/{cap})"
    gap = _hours_since(last_post_iso, now)
    if gap < rules["min_gap_hours"]:
        return False, f"ยังไม่ถึงเวลาเว้นระยะ (เพิ่งโพสต์ {gap:.1f} ชม.ที่แล้ว < {rules['min_gap_hours']})"
    if not in_window(rules, now):
        return False, "นอกช่วงเวลาโพสต์ (window = peak)"
    return True, "พร้อมผลิตคอนเทนต์"


CYCLE_PROMPT = (
    "คุณคือทีมคอนเทนต์ของบริษัท ทำหน้าที่ผลิตโพสต์โซเชียล 1 ชิ้นต่อรอบ.\n"
    "ขั้นตอนในหัว: (1) ส่องเทรนด์ที่เกี่ยวกับธุรกิจ (2) เลือกมุมที่คุ้มที่สุด "
    "(3) เขียนแคปชั่นพร้อมโพสต์ (4) อธิบายภาพประกอบ.\n"
    "บริบทธุรกิจ/โทน: {brief}\n\n"
    "ตอบเป็น JSON อย่างเดียวตามนี้:\n"
    '{{"topic":"<มุม/เทรนด์สั้นๆ>","caption":"<แคปชั่นไทยพร้อมโพสต์ มีอิโมจิและแฮชแท็กพอดี>",'
    '"image_prompt":"<คำอธิบายภาพประกอบเป็นภาษาอังกฤษสั้นๆ>","hashtags":["#..."]}}'
)

SAFETY_PROMPT = (
    "ตรวจแคปชั่นโซเชียลนี้ว่าปลอดภัยจะโพสต์ไหม (ไม่ผิดกฎหมาย ไม่หลอกลวง "
    "ไม่กล่าวอ้างเกินจริง/การันตีผลตอบแทน ไม่ละเมิด ไม่หยาบคาย).\n"
    "ตอบ JSON: {{\"safe\": true/false, \"reason\": \"<สั้นๆ>\"}}\n\nแคปชั่น:\n{caption}"
)
