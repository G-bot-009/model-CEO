#!/bin/bash
# ดับเบิลคลิกไฟล์นี้เพื่อเปิดใช้งาน Multi-Agent Agentic OS
# (ครั้งแรกจะติดตั้งสิ่งที่ต้องใช้ และให้ใส่ API key ก่อน)

cd "$(dirname "$0")" || exit 1

echo "==============================================="
echo "   🧠  Multi-Agent Agentic OS"
echo "==============================================="
echo ""

# 1) ครั้งแรก: เตรียมไฟล์ .env และให้ผู้ใช้ใส่ API key
if [ ! -f .env ] || grep -q 'ANTHROPIC_API_KEY=sk-ant-\.\.\.' .env; then
  [ -f .env ] || cp .env.example .env
  echo "⚠️  ยังไม่ได้ตั้งค่า API key"
  echo "    กำลังเปิดไฟล์ .env ให้ใส่คีย์..."
  echo "    1) วาง ANTHROPIC_API_KEY ของคุณต่อท้ายเครื่องหมาย ="
  echo "    2) กด Command(⌘)+S เพื่อบันทึก แล้วปิดหน้าต่าง"
  echo "    3) ดับเบิลคลิกไฟล์ start.command นี้อีกครั้ง"
  open -e .env
  echo ""
  read -n 1 -s -r -p "กด Enter เพื่อปิดหน้าต่างนี้..."
  exit 0
fi

# 2) ติดตั้งสิ่งที่ต้องใช้ (เงียบ ๆ — ครั้งแรกอาจช้าหน่อย)
echo "📦 กำลังตรวจสอบ/ติดตั้งสิ่งที่ต้องใช้..."
python3 -m pip install -q -r requirements.txt 2>/dev/null

# 3) เปิดเบราว์เซอร์ให้อัตโนมัติหลัง server พร้อม
( sleep 3; open "http://127.0.0.1:8000" ) &

# 4) รัน server (ปิดได้ด้วยการกด Control+C หรือปิดหน้าต่าง)
echo "🚀 กำลังเริ่มเซิร์ฟเวอร์ที่ http://127.0.0.1:8000"
echo "   (ปิดโปรแกรม: กด Control + C)"
echo ""
python3 -m uvicorn backend.main:app
