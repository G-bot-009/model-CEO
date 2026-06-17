#!/bin/bash
# ดับเบิลคลิกไฟล์นี้เพื่อเปิดใช้งาน Multi-Agent Agentic OS
# (ครั้งแรกจะติดตั้งสิ่งที่ต้องใช้ และให้ใส่ API key ก่อน)

cd "$(dirname "$0")" || exit 1

PORT=8000
URL="http://127.0.0.1:$PORT"

echo "==============================================="
echo "   🧠  Multi-Agent Agentic OS"
echo "==============================================="
echo ""

pause_exit() { echo ""; read -n 1 -s -r -p "กด Enter เพื่อปิดหน้าต่างนี้..."; echo ""; exit "${1:-0}"; }

# 0) ต้องมี Python 3
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "❌ ไม่พบ Python 3 บนเครื่อง"
  echo "   ติดตั้งก่อนที่ https://www.python.org/downloads/ แล้วลองใหม่"
  pause_exit 1
fi

# 1) อัปเดตเป็นเวอร์ชันล่าสุดเสมอ — ถ้ามีของใหม่ ให้รันตัวเปิดเวอร์ชันใหม่ทันที
if [ -d .git ] && [ -z "$AGENTOS_REEXEC" ]; then
  echo "⬇️  กำลังตรวจอัปเดต..."
  before="$(git rev-parse HEAD 2>/dev/null)"
  git pull --ff-only 2>/dev/null || echo "   (ข้ามการอัปเดต — ใช้เวอร์ชันที่มีอยู่)"
  after="$(git rev-parse HEAD 2>/dev/null)"
  if [ -n "$after" ] && [ "$before" != "$after" ]; then
    echo "   อัปเดตเป็นเวอร์ชันล่าสุดแล้ว ✓ (กำลังรีสตาร์ตตัวเปิด)"
    export AGENTOS_REEXEC=1
    exec "$0" "$@"     # รันตัวเปิดเวอร์ชันใหม่ที่เพิ่งดึงมา
  fi
  echo "   เป็นเวอร์ชันล่าสุดแล้ว ✓"
fi

# 2) ตรวจ API key — ถ้ายังไม่ตั้ง ให้พิมพ์ใส่ตรงนี้เลย แล้วไปต่อในรอบเดียว
key_ok() {
  [ -f .env ] || return 1
  local v
  v="$(grep -E '^ANTHROPIC_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d ' \r\"')"
  case "$v" in
    sk-ant-...*|"") return 1 ;;   # ค่าตัวอย่าง/ว่าง = ยังไม่ตั้ง
    sk-ant-*)      return 0 ;;    # คีย์จริง
    *)             return 1 ;;
  esac
}

if ! key_ok; then
  echo "🔑 ยังไม่ได้ตั้งค่า ANTHROPIC API KEY"
  echo "   หาคีย์ได้ที่:  https://console.anthropic.com/settings/keys"
  echo "   (คีย์ขึ้นต้นด้วย sk-ant-... — เก็บในเครื่องนี้เท่านั้น ไม่ส่งไปไหน)"
  echo ""
  read -r -p "วางคีย์แล้วกด Enter: " NEWKEY
  NEWKEY="$(printf '%s' "$NEWKEY" | tr -d ' \r\"')"
  case "$NEWKEY" in
    sk-ant-*)
      [ -f .env ] || cp .env.example .env 2>/dev/null || touch .env
      grep -v '^ANTHROPIC_API_KEY=' .env > .env.tmp 2>/dev/null || : > .env.tmp
      echo "ANTHROPIC_API_KEY=$NEWKEY" >> .env.tmp
      mv .env.tmp .env
      echo "✅ บันทึกคีย์แล้ว — เริ่มเซิร์ฟเวอร์ต่อเลย"
      echo ""
      ;;
    *)
      echo "❌ คีย์ต้องขึ้นต้นด้วย sk-ant- — ยังไม่ได้บันทึก"
      echo "   เปิดไฟล์ .env ให้แก้เองแทน แล้วดับเบิลคลิก start.command อีกครั้ง"
      [ -f .env ] || cp .env.example .env 2>/dev/null
      open -e .env
      pause_exit 1
      ;;
  esac
fi

# 3) ติดตั้งสิ่งที่ต้องใช้ (ครั้งแรกอาจช้าหน่อย) — ถ้าพลาดให้แสดง log จริง
echo "📦 กำลังตรวจสอบ/ติดตั้งสิ่งที่ต้องใช้..."
if ! "$PY" -m pip install -q -r requirements.txt; then
  echo "⚠️  ติดตั้งไม่สำเร็จ — ลองใหม่แบบแสดงรายละเอียด:"
  "$PY" -m pip install -r requirements.txt || { echo "❌ ติดตั้ง dependencies ไม่สำเร็จ"; pause_exit 1; }
fi

# 4) ปิดเซิร์ฟเวอร์เก่าที่ค้างอยู่บนพอร์ตเดียวกัน (กัน ERR_CONNECTION_REFUSED จากพอร์ตชน)
STALE="$(lsof -ti "tcp:$PORT" 2>/dev/null)"
if [ -n "$STALE" ]; then
  echo "♻️  ปิดเซิร์ฟเวอร์เก่าที่ค้างบนพอร์ต $PORT..."
  kill -9 $STALE 2>/dev/null
  sleep 1
fi

# 5) รัน server เบื้องหลัง + เก็บ log ไว้ดูถ้าพัง
LOG="$(mktemp -t agentos 2>/dev/null || echo /tmp/agentos.log)"
echo "🚀 กำลังเริ่มเซิร์ฟเวอร์ที่ $URL"
"$PY" -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!

# 6) รอจนเซิร์ฟเวอร์ตอบจริง แล้วเปิดเบราว์เซอร์ให้ทันที (เช็กถี่ทุก 0.5 วิ สูงสุด ~40 วิ)
echo "⏳ รอเซิร์ฟเวอร์พร้อม..."
UP=0
for _ in $(seq 1 80); do
  kill -0 "$SERVER_PID" 2>/dev/null || break          # เซิร์ฟเวอร์ดับไปแล้ว
  if curl -s -o /dev/null "$URL"; then UP=1; break; fi  # ตอบแล้ว → เปิดทันที
  sleep 0.5
done

if [ "$UP" = "1" ]; then
  echo "✅ พร้อมแล้ว — กำลังเปิดหน้าเข้าสู่ระบบ..."
  open "$URL/login"
  echo ""
  echo "   เข้าสู่ระบบที่ $URL/login  (user: admin · pass: admin)"
  echo "   (ปิดโปรแกรม: ปิดหน้าต่างนี้ หรือกด Control + C — อย่าเพิ่งปิดถ้ายังใช้งานอยู่)"
  echo ""
  wait "$SERVER_PID"   # คาหน้าต่างไว้ให้เซิร์ฟเวอร์ทำงานต่อ
else
  echo ""
  echo "❌ เซิร์ฟเวอร์เริ่มไม่สำเร็จ — นี่คือข้อความผิดพลาดล่าสุด:"
  echo "-----------------------------------------------------------"
  tail -n 30 "$LOG"
  echo "-----------------------------------------------------------"
  echo "เคล็ดลับที่พบบ่อย:"
  echo "  • ANTHROPIC_API_KEY ในไฟล์ .env ผิด/ว่าง → แก้แล้วเปิดใหม่"
  echo "  • พอร์ต $PORT ถูกใช้งานอยู่ → ปิดโปรแกรมที่ใช้พอร์ตนี้"
  kill -9 "$SERVER_PID" 2>/dev/null
  pause_exit 1
fi
