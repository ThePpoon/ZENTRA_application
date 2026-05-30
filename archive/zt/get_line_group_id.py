#!/usr/bin/env python3.11
"""
get_line_group_id.py — ZENTRA LINE Group ID Helper
====================================================
วิธีใช้:
  1. python get_line_group_id.py
  2. เปิด URL ที่แสดงขึ้นมา
  3. ไปที่ LINE Developers → Webhook URL → ใส่ URL นั้น
  4. พิมพ์อะไรก็ได้ในกลุ่ม LINE ที่ต้องการ
  5. กลับมาดูหน้าต่าง terminal — จะเห็น Group ID

หมายเหตุ:
  - ต้องใส่ CHANNEL_SECRET ใน .env ก่อน
  - ต้องเพิ่ม LINE Bot เป็นสมาชิกในกลุ่มก่อน
  - ต้องมี internet connection
====================================================
"""

from __future__ import annotations
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

CHANNEL_SECRET = os.getenv("LINE_OA_CHANNEL_SECRET", "")
ACCESS_TOKEN   = os.getenv("LINE_OA_CHANNEL_ACCESS_TOKEN", "")

print("=" * 55)
print("  ZENTRA — LINE Group ID Helper")
print("=" * 55)

if not CHANNEL_SECRET or not ACCESS_TOKEN:
    print("\n⚠️  กรุณาตั้งค่าใน .env ก่อน:")
    print("    LINE_OA_CHANNEL_SECRET=...")
    print("    LINE_OA_CHANNEL_ACCESS_TOKEN=...")
    print("\n  หา Channel Secret ได้ที่:")
    print("  https://developers.line.biz/ → Channel settings → Basic settings")
    import sys; sys.exit(1)

try:
    from flask import Flask, request, abort
    from linebot.v3 import WebhookHandler
    from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
    from linebot.v3.webhooks import MessageEvent, TextMessageContent
    from linebot.v3.exceptions import InvalidSignatureError
except ImportError:
    print("\n⚠️  กรุณา pip install flask line-bot-sdk")
    import sys; sys.exit(1)

app = Flask(__name__)
handler = WebhookHandler(CHANNEL_SECRET)
found_groups: set[str] = set()


@app.route("/webhook", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    src = event.source
    if src.type == "group":
        gid = src.group_id
        if gid not in found_groups:
            found_groups.add(gid)
            print(f"\n✅ พบ Group ID!")
            print(f"   Group ID = {gid}")
            print(f"   ใส่ใน .env:")
            print(f"   LINE_OA_GROUP_SUPERVISOR={gid}")
            print(f"   LINE_OA_GROUP_SAFETY={gid}")
            print(f"   LINE_OA_GROUP_EMERGENCY={gid}")
            print(f"\n   (เปลี่ยนตามบทบาทที่เหมาะสม)\n")
    elif src.type == "room":
        print(f"\n✅ Room ID = {src.room_id}")
    elif src.type == "user":
        print(f"\n✅ User ID = {src.user_id}")


if __name__ == "__main__":
    try:
        import subprocess, sys
        # ลอง ngrok ก่อน (ถ้ามี)
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(5000).public_url
            webhook_url = f"{public_url}/webhook"
            print(f"\n  🌐 Webhook URL (via ngrok):")
            print(f"  {webhook_url}")
        except Exception:
            print("\n  ⚠️  ngrok ไม่พบ — ใช้ localhost")
            print(f"  กรุณาตั้งค่า Webhook URL ด้วยตนเอง:")
            print(f"  http://YOUR_PUBLIC_IP:5000/webhook")

        print("\n  วิธีตั้งค่า Webhook ใน LINE Developers:")
        print("  1. ไปที่ https://developers.line.biz/")
        print("  2. เลือก Channel → Messaging API")
        print("  3. Webhook URL → ใส่ URL ด้านบน")
        print("  4. Use webhook = ON")
        print("  5. พิมพ์อะไรก็ได้ในกลุ่ม LINE")
        print("  6. ดู Group ID ที่แสดงใน terminal นี้")
        print("\n  กด Ctrl+C เพื่อหยุด\n")
        print("-" * 55)

        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n  หยุดแล้ว")
        if found_groups:
            print(f"  Group IDs ที่พบ: {found_groups}")
