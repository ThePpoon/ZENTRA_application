#!/usr/bin/env python3
"""
debug_line.py -- ZENTRA LINE Debug Tool
========================================
Test LINE notification without running main.py

  python debug_line.py           # test text only
  python debug_line.py --image   # test with image
  python debug_line.py --info    # show config info
========================================
"""

from __future__ import annotations
import sys
import io
import os

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_file = Path(".env")
if not env_file.exists():
    print("ERROR: .env not found. Run from ZENTRA project folder.")
    sys.exit(1)
load_dotenv()

import config as cfg


def _is_valid_id(gid: str) -> bool:
    if not gid:
        return False
    if gid.startswith("Cxxxxxxxx"):
        return False
    if len(gid) < 20:
        return False
    return True


def show_info():
    print("\n======== ZENTRA LINE Config Info ========")
    token = cfg.LINE_OA_CHANNEL_ACCESS_TOKEN
    print(f"  Channel Token : {token[:15]}...{token[-5:] if len(token) > 20 else '(too short!)'}")

    ids = {
        "SUPERVISOR": cfg.LINE_OA_GROUP_SUPERVISOR,
        "SAFETY":     cfg.LINE_OA_GROUP_SAFETY,
        "EMERGENCY":  cfg.LINE_OA_GROUP_EMERGENCY,
    }
    for role, gid in ids.items():
        valid = _is_valid_id(gid)
        status = "OK" if valid else "INVALID/PLACEHOLDER"
        print(f"  {role:<12}: {gid[:25] if gid else '(empty)'} [{status}]")

    # Count valid
    valid_count = sum(1 for g in ids.values() if _is_valid_id(g))
    print(f"\n  Valid IDs: {valid_count}/3")
    if valid_count == 0:
        print("\n  [!] No valid LINE IDs found!")
        print("      Run: python get_line_group_id.py")
        print("      Then update .env with the real group IDs")
    print("=========================================\n")


def test_api_connection():
    """Check if the LINE token is valid"""
    print("[Test] Checking LINE token...")
    try:
        resp = requests.get(
            "https://api.line.me/v2/bot/info",
            headers={"Authorization": f"Bearer {cfg.LINE_OA_CHANNEL_ACCESS_TOKEN}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"[Test] OK - Bot name: {data.get('displayName', 'N/A')}")
            return True
        else:
            print(f"[Test] FAILED - Status: {resp.status_code}")
            print(f"       Response: {resp.text[:200]}")
            if resp.status_code == 401:
                print("       --> Token is invalid! Check LINE_OA_CHANNEL_ACCESS_TOKEN in .env")
            return False
    except Exception as e:
        print(f"[Test] Connection error: {e}")
        return False


def test_send_text(group_id: str, level: str = "warning"):
    """Send a test text message"""
    from datetime import datetime
    ts  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    msg = (
        f"⚠️ [ZENTRA TEST]\n"
        f"This is a test alert from ZENTRA.\n"
        f"System is working correctly!\n\n"
        f"📅 {ts}"
    )
    print(f"[Test] Sending text to {group_id[:14]}...")
    resp = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Authorization": f"Bearer {cfg.LINE_OA_CHANNEL_ACCESS_TOKEN}",
            "Content-Type":  "application/json",
        },
        json={
            "to": group_id,
            "messages": [{"type": "text", "text": msg}]
        },
        timeout=12,
    )
    if resp.status_code == 200:
        print(f"[Test] SENT OK!")
        return True
    else:
        print(f"[Test] FAILED: {resp.status_code}")
        print(f"       {resp.text[:300]}")
        return False


def test_send_with_image(group_id: str):
    """Send a test message with a generated image"""
    import cv2
    import numpy as np

    print("[Test] Creating test image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 50)

    # Draw ZENTRA logo-style
    cv2.putText(img, "ZENTRA", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                (100, 255, 100), 5, cv2.LINE_AA)
    cv2.putText(img, "Safety System - TEST", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "PPE Detection | Zone Monitor | Fall Detection", (30, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (20, 20), (620, 460), (100, 255, 100), 2)

    print("[Test] Uploading test image...")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        print("[Test] Failed to encode image")
        return False

    try:
        resp = requests.post(
            cfg.IMAGE_UPLOAD_URL,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": ("zentra_test.jpg", buf.tobytes(), "image/jpeg")},
            timeout=15,
        )
        if resp.status_code == 200 and resp.text.strip().startswith("https://"):
            img_url = resp.text.strip()
            print(f"[Test] Image URL: {img_url}")

            from datetime import datetime
            ts  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            msg = f"⚠️ [ZENTRA TEST - WITH IMAGE]\nSystem is working!\n\n📅 {ts}"

            resp2 = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Authorization": f"Bearer {cfg.LINE_OA_CHANNEL_ACCESS_TOKEN}",
                    "Content-Type":  "application/json",
                },
                json={
                    "to": group_id,
                    "messages": [
                        {"type": "text", "text": msg},
                        {"type": "image",
                         "originalContentUrl": img_url,
                         "previewImageUrl":    img_url},
                    ]
                },
                timeout=12,
            )
            if resp2.status_code == 200:
                print("[Test] Message + Image SENT OK!")
                return True
            else:
                print(f"[Test] Send failed: {resp2.status_code} {resp2.text[:200]}")
        else:
            print(f"[Test] Image upload failed: {resp.status_code} {resp.text[:100]}")
    except Exception as e:
        print(f"[Test] Error: {e}")
    return False


def main():
    ap = argparse.ArgumentParser(description="ZENTRA LINE Debug Tool")
    ap.add_argument("--info",  action="store_true", help="Show config info only")
    ap.add_argument("--image", action="store_true", help="Test with image")
    ap.add_argument("--id",    default=None,        help="Override recipient ID")
    args = ap.parse_args()

    print("\n======== ZENTRA LINE Debug ========")
    show_info()

    if args.info:
        return

    # Verify token first
    if not test_api_connection():
        print("\nERROR: Fix your LINE token first!")
        return

    # Find a valid recipient
    target_id = args.id
    if not target_id:
        for gid in [cfg.LINE_OA_GROUP_SUPERVISOR, cfg.LINE_OA_GROUP_SAFETY,
                    cfg.LINE_OA_GROUP_EMERGENCY]:
            if _is_valid_id(gid):
                target_id = gid
                break

    if not target_id:
        print("\nERROR: No valid recipient ID found!")
        print("  Options:")
        print("  1. Run: python get_line_group_id.py")
        print("  2. Set .env LINE_OA_GROUP_SUPERVISOR=U... or C...")
        print("  3. Use: python debug_line.py --id YOUR_ID")
        return

    print(f"\n[Test] Target: {target_id}")

    if args.image:
        ok = test_send_with_image(target_id)
    else:
        ok = test_send_text(target_id)

    print("\n======== Result ========")
    if ok:
        print("  SUCCESS! Check your LINE app.")
    else:
        print("  FAILED. Check error messages above.")
    print("========================\n")


if __name__ == "__main__":
    main()
