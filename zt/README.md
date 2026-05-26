# ZENTRA — Zone Environment Network Thermal Risk Analysis
## ระบบ AI ตรวจความปลอดภัยในโรงงาน | CEDT Innovation Summit 2026

---

## 🗂️ โครงสร้างโปรเจกต์

```
TestDeteckl/
├── main.py                  # ← รันตัวนี้เพื่อเริ่มระบบ
├── config.py                # การตั้งค่าทั้งหมด
├── ppe_config.py            # เลือก PPE ที่จะตรวจ
├── train_all.py             # เทรนโมเดล
├── test_system.py           # ตรวจสอบระบบก่อนรัน
├── get_line_group_id.py     # หา LINE Group ID
├── setup.bat                # ติดตั้งแบบ Windows (ดับเบิลคลิก)
├── requirements.txt
├── docker-compose.yml
├── .env.example             # → copy เป็น .env
│
├── modules/
│   ├── ppe.py               # Module 1: PPE Detection
│   ├── safety_zone.py       # Module 2: Safety Zone (ByteTrack)
│   └── heat_stroke.py       # Module 3: Heat Stroke / Fall
│
├── alerts/
│   └── line_notify.py       # LINE OA Alert System
│
├── utils/
│   ├── tracker.py           # ByteTrack Multi-Object Tracking
│   └── collector.py         # บันทึก frame สำหรับเทรน
│
├── reports/
│   └── daily_report.py      # รายงานความปลอดภัยรายวัน
│
├── data/
│   ├── zones.json           # polygon ของ Safety Zone (auto-save)
│   └── collected/           # frame ที่เก็บไว้เทรน
│       ├── ppe_violations/
│       ├── zone_intrusions/
│       ├── fall_events/
│       └── normal/
│
├── models/                  # โมเดลที่เทรนแล้ว
│   ├── ppe_finetuned.pt
│   ├── pose_finetuned.pt
│   └── fall_lstm.pt
│
└── logs/                    # log ประจำวัน
```

---

## 🔧 ขั้นที่ 1 — ติดตั้งระบบ

### วิธีที่ 1: อัตโนมัติ (แนะนำ)
```
ดับเบิลคลิก setup.bat
```

### วิธีที่ 2: Manual

**1.1 ติดตั้ง Python 3.11**
- ดาวน์โหลด: https://www.python.org/downloads/
- ☑️ เลือก "Add Python to PATH"

**1.2 Clone โปรเจกต์**
```powershell
git clone https://github.com/ThePpoon/TestDeteckl.git
cd TestDeteckl
```

**1.3 ติดตั้ง packages**
```powershell
pip install -r requirements.txt

# ถ้ามี NVIDIA GPU (แนะนำ — เร็วกว่า 10x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**1.4 ตั้งค่า .env**
```powershell
copy .env.example .env
# เปิด .env ด้วย Notepad แล้วใส่ค่า
```

---

## 🐋 ขั้นที่ 2 — ติดตั้ง Docker + Inference Server

**2.1 ติดตั้ง Docker Desktop**
- ดาวน์โหลด: https://www.docker.com/products/docker-desktop
- เปิด Docker Desktop และรอให้ขึ้น "Docker is running"

**2.2 เปิด Roboflow Inference Server**
```powershell
docker compose up inference -d
```

ตรวจสอบว่า server พร้อม:
```powershell
docker compose logs inference
# ต้องเห็น: Application startup complete
```

> **ถ้าไม่มี GPU** แก้ `docker-compose.yml` บรรทัด image:
> ```
> image: roboflow/roboflow-inference-server-cpu:latest
> ```
> แล้วลบส่วน `deploy:` ออก

---

## 📱 ขั้นที่ 3 — ตั้งค่า LINE OA

**3.1 สมัคร LINE Developers**
- https://developers.line.biz/
- Log in ด้วย LINE account

**3.2 สร้าง Messaging API Channel**
1. Create a new provider
2. Create channel → Messaging API
3. ตั้งชื่อ: "ZENTRA Safety"
4. กด Agree

**3.3 เปิด Auto-reply (ปิด) และ Webhooks (เปิด)**
- Channel → Messaging API → LINE Official Account features
- Auto-reply messages → **Disabled**
- Use webhooks → **Enabled**

**3.4 ออก Channel Access Token**
- Messaging API tab → Channel access token → Issue

**3.5 หา Group ID**
```powershell
pip install flask pyngrok
python get_line_group_id.py
```
- ทำตามขั้นตอนที่แสดงในหน้าต่าง

**3.6 ใส่ค่าใน .env**
```env
LINE_OA_CHANNEL_ACCESS_TOKEN=eyJ...
LINE_OA_GROUP_SUPERVISOR=C1234567890abcdef
LINE_OA_GROUP_SAFETY=C1234567890abcdef
LINE_OA_GROUP_EMERGENCY=C1234567890abcdef
```

---

## ✅ ขั้นที่ 4 — ตรวจสอบระบบ

```powershell
python test_system.py          # ตรวจสอบทุกอย่าง
python test_system.py --webcam # ทดสอบกล้องด้วย
python test_system.py --line   # ทดสอบส่ง LINE
```

---

## 🚀 ขั้นที่ 5 — รันระบบ

```powershell
python main.py
```

### Controls ในหน้าต่างกล้อง

| ปุ่ม | การทำงาน |
|------|----------|
| `Z` | เริ่มวาด Safety Zone ใหม่ |
| คลิกซ้าย | เพิ่มจุด (ต้องอย่างน้อย 3 จุด) |
| คลิกขวา | บันทึก Zone |
| `Z` (อีกครั้ง) | ยกเลิกการวาด |
| `C` | ลบ Zone ทั้งหมด |
| `S` | ดูสถิติ |
| `T` | เริ่มเทรนโมเดลใน background |
| `H` | ดูคำสั่งทั้งหมด |
| `Q` | ออกระบบ |

---

## 🏋️ ขั้นที่ 6 — เทรนโมเดล (เพิ่มความแม่นยำ)

### ตรวจสอบก่อนเทรน
```powershell
python train_all.py --check
```

### เทรนโมเดล PPE (แนะนำเริ่มที่นี่)
```powershell
# GPU (เร็ว ~2-4 ชม.)
python train_all.py --task ppe

# CPU (ช้ามาก ~24+ ชม. ไม่แนะนำ)
python train_all.py --task ppe --fast
```

### เทรนโมเดล Pose/Fall
```powershell
python train_all.py --task pose
```

### เทรน LSTM (ต้องมีวิดีโอใน videos/<label>/ ก่อน)
```powershell
python train_all.py --task lstm
```

### เทรนทั้งหมดพร้อมกัน
```powershell
python train_all.py --task ppe pose lstm --epochs 150
```

### หลังเทรนเสร็จ
ไฟล์โมเดลจะอยู่ใน `models/`:
- `models/ppe_finetuned.pt` — ใช้ตรวจ PPE
- `models/pose_finetuned.pt` — ใช้ตรวจ Fall/Pose
- `models/fall_lstm.pt` — LSTM classifier

เปิดใช้โมเดลที่เทรนเองใน `.env`:
```env
USE_LOCAL_MODEL=true
```

---

## 🔄 เปลี่ยนจาก Webcam เป็นกล้อง IP

แก้ใน `.env`:
```env
CAMERA_SOURCE=rtsp
RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream1
```

URL format ขึ้นอยู่กับยี่ห้อกล้อง:
- Hikvision: `rtsp://admin:pass@IP:554/Streaming/Channels/101`
- Dahua:     `rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0`
- Generic:   `rtsp://admin:pass@IP:554/stream1`

---

## 🔄 เปลี่ยน PPE ที่ต้องการตรวจ

แก้ใน `ppe_config.py`:
```python
PPE_PROFILE = "helmet_vest"   # แค่หมวก + เสื้อกั๊ก
# "full"           — ตรวจทุกอย่าง
# "helmet_vest"    — หมวก + เสื้อกั๊ก
# "helmet_only"    — แค่หมวก
# "custom"         — กำหนดเองใน CUSTOM_ITEMS
```

---

## 🐛 แก้ปัญหาที่พบบ่อย

| ปัญหา | วิธีแก้ |
|--------|---------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| กล้องไม่เปิด | ตรวจ `WEBCAM_INDEX=0` ใน .env |
| Inference Server ไม่ตอบ | `docker compose up inference -d` |
| `401 Unauthorized` | ROBOFLOW_API_KEY ผิด |
| LINE ไม่ส่ง | ตรวจ ACCESS_TOKEN และ GROUP_ID |
| FPS ต่ำมาก | เพิ่ม `INFER_EVERY_N_FRAMES=5` ใน .env |
| GPU ไม่ถูกใช้ | `pip install torch --index-url .../cu124` |
| MediaPipe error | `pip install mediapipe==0.10.14` |

---

## 📊 LINE Alert Format

**ระดับเฝ้าระวัง (PPE ขาด):**
```
🪖 ZENTRA PPE Alert
⚠️ ขาดอุปกรณ์ PPE:
   ไม่สวมหมวก
📋 Profile: ตรวจ: หมวกนิรภัย | เสื้อกั๊กสะท้อนแสง
🕐 12/04/2026 14:35:22
[รูปภาพ]
```

**ระดับตักเตือน (เข้า Zone):**
```
⛔ ZENTRA Zone Alert
🚨 พบบุคคลเข้าเขตอันตราย 2 คน
📍 Zone: Zone 1
🕐 12/04/2026 14:36:10
[รูปภาพ]
```

**ระดับฉุกเฉิน (ล้ม/หมดสติ):**
```
🆘 ZENTRA Emergency Alert
🚨 ตรวจพบความเสี่ยงสูง!
💥 Sudden Fall
📊 Risk Score: 0.87
🕐 12/04/2026 14:37:55
[รูปภาพ]
```

---

## 📈 รายงานประจำวัน (ส่งทุกวัน 20:00 น.)

```
📊 ZENTRA รายงานความปลอดภัยประจำวัน
📅 2026-04-12  ⏰ 20:00 น.
───────────────────────────────────
🪖 PPE Violations    : 3 ครั้ง
⛔ Zone Intrusions   : 1 ครั้ง
🆘 Fall / Heat Stroke: 0 ครั้ง
───────────────────────────────────
📈 รวมเหตุการณ์       : 4 ครั้ง
✅ ดัชนีความปลอดภัย  : 80%
🎥 Frame ที่ประมวลผล  : 1,245,600
⏱️  ทำงาน             : 12.3 ชม.

💡 กรุณาดำเนินมาตรการป้องกันต่อเนื่อง
```

---

*ZENTRA | CEDT Innovation Summit 2026 | วาจาไม่สำคัญ ดวงใจเท่านั้นที่ชัดเจน*
