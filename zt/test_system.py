#!/usr/bin/env python3.11
"""
test_system.py — ZENTRA System Health Check
============================================
รันก่อน main.py เพื่อตรวจสอบว่าระบบพร้อมทำงาน

  python test_system.py
  python test_system.py --webcam     # ทดสอบกล้องด้วย
  python test_system.py --line       # ทดสอบส่ง LINE
============================================
"""

from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path

# ── Color output ───────────────────────────────────────────────────
OK   = "✅"
WARN = "⚠️ "
FAIL = "❌"
INFO = "ℹ️ "


def _ok(msg):   print(f"  {OK}  {msg}")
def _warn(msg): print(f"  {WARN} {msg}")
def _fail(msg): print(f"  {FAIL} {msg}")
def _info(msg): print(f"  {INFO}  {msg}")


# ================================================================
# CHECKS
# ================================================================
def check_python():
    print("\n[1] Python")
    v = sys.version_info
    if v >= (3, 10):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor} — ต้องการ 3.10+ (แนะนำ 3.11)")
        return False
    return True


def check_packages():
    print("\n[2] Python Packages")
    packages = {
        "cv2":           "opencv-python",
        "numpy":         "numpy",
        "ultralytics":   "ultralytics",
        "mediapipe":     "mediapipe",
        "requests":      "requests",
        "dotenv":        "python-dotenv",
        "schedule":      "schedule",
    }
    optional = {
        "inference_sdk": "inference-sdk (Roboflow)",
        "torch":         "torch (PyTorch — ต้องการสำหรับเทรน)",
        "roboflow":      "roboflow (ต้องการสำหรับ download dataset)",
        "linebot":       "line-bot-sdk",
    }
    all_ok = True
    for mod, pkg in packages.items():
        try:
            __import__(mod)
            _ok(pkg)
        except ImportError:
            _fail(f"{pkg} — pip install {pkg}")
            all_ok = False
    for mod, pkg in optional.items():
        try:
            __import__(mod)
            _ok(f"{pkg} (optional)")
        except ImportError:
            _warn(f"{pkg} (optional) — pip install {pkg.split()[0]}")
    return all_ok


def check_env():
    print("\n[3] Environment Variables (.env)")
    from dotenv import load_dotenv
    load_dotenv()

    required = {
        "ROBOFLOW_API_KEY":              "Roboflow API Key",
        "LINE_OA_CHANNEL_ACCESS_TOKEN":  "LINE Channel Access Token",
    }
    optional = {
        "LINE_OA_GROUP_SUPERVISOR": "LINE Group (หัวหน้างาน)",
        "LINE_OA_GROUP_SAFETY":     "LINE Group (Safety)",
        "LINE_OA_GROUP_EMERGENCY":  "LINE Group (Emergency)",
    }
    all_ok = True
    for key, desc in required.items():
        val = os.getenv(key, "")
        if val and val != f"ใส่_{key}_ที่นี่":
            _ok(f"{desc} — {val[:8]}...")
        else:
            _fail(f"{desc} — ไม่พบใน .env")
            all_ok = False
    for key, desc in optional.items():
        val = os.getenv(key, "")
        if val and val.startswith("C"):
            _ok(f"{desc} — {val[:12]}...")
        else:
            _warn(f"{desc} — ยังไม่ได้ตั้งค่า (แจ้งเตือนไม่ส่ง)")
    return all_ok


def check_docker():
    print("\n[4] Docker & Inference Server")
    import subprocess, requests as req
    # ตรวจ Docker
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if r.returncode == 0:
            _ok("Docker Desktop is running")
        else:
            _warn("Docker พบแต่ไม่ตอบสนอง — เปิด Docker Desktop ก่อน")
            return False
    except Exception:
        _fail("Docker ไม่พบ — ติดตั้งจาก https://www.docker.com/products/docker-desktop")
        return False

    # ตรวจ Inference Server
    try:
        resp = req.get("http://localhost:9001/", timeout=3)
        if resp.status_code == 200:
            _ok("Roboflow Inference Server (localhost:9001) — READY")
        else:
            _warn(f"Inference Server ตอบ {resp.status_code}")
    except Exception:
        _warn("Inference Server ไม่ตอบสนอง — รัน: docker compose up inference -d")
        return False
    return True


def check_camera(test_webcam: bool = False):
    print("\n[5] Camera")
    import cv2
    if not test_webcam:
        _info("ข้ามการทดสอบกล้อง (ใช้ --webcam เพื่อทดสอบ)")
        return True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        _fail("เปิดกล้องไม่ได้ — ตรวจสอบ WEBCAM_INDEX ใน .env")
        return False
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        _ok(f"Webcam index=0 — {w}x{h}")
        return True
    else:
        _fail("อ่าน frame จากกล้องไม่ได้")
        return False


def check_folders():
    print("\n[6] Folder Structure")
    required = [
        "data/collected/ppe_violations",
        "data/collected/zone_intrusions",
        "data/collected/fall_events",
        "data/collected/normal",
        "models", "logs", "runs", "reports",
    ]
    all_ok = True
    for folder in required:
        p = Path(folder)
        if p.exists():
            _ok(folder)
        else:
            p.mkdir(parents=True, exist_ok=True)
            _ok(f"{folder} — สร้างใหม่")
    return all_ok


def check_models():
    print("\n[7] Local Models (Optional)")
    models = {
        "models/ppe_finetuned.pt":  "PPE fine-tuned model",
        "models/pose_finetuned.pt": "Pose fine-tuned model",
        "models/fall_lstm.pt":      "Fall LSTM model",
    }
    for path, desc in models.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            _ok(f"{desc} ({size_mb:.1f} MB)")
        else:
            _info(f"{desc} — ไม่มี (จะใช้ Roboflow cloud แทน)")


def test_line_notify():
    print("\n[LINE] ทดสอบส่งข้อความ LINE...")
    from dotenv import load_dotenv
    load_dotenv()
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import config as cfg
        from alerts.line_notify import start_sender, stop_sender, send_line_notify
        start_sender()
        time.sleep(0.5)
        send_line_notify(
            "🧪 ZENTRA Test Alert\n"
            "✅ ระบบแจ้งเตือนทำงานปกติ\n"
            "🎉 การตั้งค่า LINE OA สำเร็จ!",
            level=cfg.ALERT_LEVEL_WARNING,
            cooldown_key="test_alert",
            cooldown_sec=0,
        )
        time.sleep(3)
        stop_sender()
        _ok("ส่งข้อความ LINE สำเร็จ — ตรวจสอบกลุ่ม LINE")
    except Exception as e:
        _fail(f"LINE Error: {e}")


# ================================================================
# MAIN
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="ZENTRA System Health Check")
    ap.add_argument("--webcam", action="store_true", help="ทดสอบกล้อง")
    ap.add_argument("--line",   action="store_true", help="ทดสอบส่ง LINE")
    args = ap.parse_args()

    print("=" * 55)
    print("  ZENTRA — System Health Check")
    print("=" * 55)

    results = []
    results.append(("Python",      check_python()))
    results.append(("Packages",    check_packages()))
    results.append(("ENV",         check_env()))
    results.append(("Docker",      check_docker()))
    results.append(("Camera",      check_camera(args.webcam)))
    results.append(("Folders",     check_folders()))
    check_models()

    if args.line:
        test_line_notify()

    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    all_pass = True
    for name, ok in results:
        status = OK if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"  {OK} ระบบพร้อมทำงาน! รัน: python main.py")
    else:
        print(f"  {FAIL} แก้ปัญหาด้านบนก่อนรัน main.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
