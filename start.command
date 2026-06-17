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

# 1) ครั้งแรก: เตรียมไฟล์ .env และให้ผู้ใช้ใส่ API key
if [ ! -f .env ] || grep -q 'ANTHROPIC_API_KEY=sk-ant-\.\.\.' .env; then
  [ -f .env ] || cp .env.example .env
  echo "⚠️  ยังไม่ได้ตั้งค่า API key"
  echo "    กำลังเปิดไฟล์ .env ให้ใส่คีย์..."
  echo "    1) วาง ANTHROPIC_API_KEY ของคุณต่อท้ายเครื่องหมาย ="
  echo "    2) กด Command(⌘)+S เพื่อบันทึก แล้วปิดหน้าต่าง"
  echo "    3) ดับเบิลคลิกไฟล์ start.command นี้อีกครั้ง"
  open -e .env
  pause_exit 0
fi

# 2) ดึงโค้ดเวอร์ชันล่าสุด (ถ้าเป็น git repo)
if [ -d .git ]; then
  echo "⬇️  กำลังอัปเดตเป็นเวอร์ชันล่าสุด..."
  git pull --ff-only 2>/dev/null || echo "   (ข้ามการอัปเดต — ใช้เวอร์ชันที่มีอยู่)"
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

# 6) รอจนเซิร์ฟเวอร์ตอบจริง แล้วค่อยเปิดเบราว์เซอร์ (สูงสุด ~40 วินาที)
echo "⏳ รอเซิร์ฟเวอร์พร้อม..."
UP=0
for _ in $(seq 1 40); do
  kill -0 "$SERVER_PID" 2>/dev/null || break          # เซิร์ฟเวอร์ดับไปแล้ว
  if curl -s -o /dev/null "$URL"; then UP=1; break; fi  # ตอบแล้ว
  sleep 1
done

if [ "$UP" = "1" ]; then
  echo "✅ พร้อมแล้ว — กำลังเปิดเบราว์เซอร์..."
  open "$URL"
  echo ""
  echo "   เปิดใช้งานได้ที่ $URL"
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
