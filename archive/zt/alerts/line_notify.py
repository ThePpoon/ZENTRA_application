# alerts/line_notify.py — ZENTRA LINE OA Alert Engine (PRODUCTION)
# ================================================================

from __future__ import annotations
import time, threading, traceback
import requests, cv2
from datetime import datetime
from typing import Optional
from pathlib import Path
import uuid              

def _cfg():
    import config as c
    return c

# ── State ─────────────────────────────────────────────────────────
_lock          = threading.Lock()
_last_sent:    dict[str, float] = {}
_queue:        list[dict]       = []
_queue_lock    = threading.Lock()
_running       = False
_sender_thread: Optional[threading.Thread] = None

CATBOX_URL     = "https://catbox.moe/user/api.php"
LINE_PUSH_URL  = "https://api.line.me/v2/bot/message/push"
MAX_RETRIES    = 3


# ================================================================
# HELPERS
# ================================================================
def _valid(gid: str) -> bool:
    """Real LINE ID: starts with U/C/R, length >= 28, no placeholder"""
    return bool(gid) and len(gid) >= 28 and "xxxx" not in gid.lower()


def _recipients(level: str) -> list[str]:
    """Get filtered real recipient IDs for the alert level"""
    cfg = _cfg()
    raw = cfg.ALERT_RECIPIENTS.get(level, [])

    # Add supervisor as universal fallback (catches warning/alert/emergency)
    fallback = [cfg.LINE_OA_GROUP_SUPERVISOR]
    all_ids  = list(dict.fromkeys(raw + fallback))   # deduplicate, keep order
    result   = [r for r in all_ids if _valid(r)]

    if not result:
        print(f"[LINE] No valid IDs for level='{level}' -- check .env")
    return result


# ================================================================
# IMAGE UPLOAD -- catbox.moe with retry
# ================================================================
# def upload_image(image, max_side: int = 1280, quality: int = 85) -> str:
#     """
#     Encode frame as JPEG -> upload to catbox.moe -> return public URL.
#     Retries up to 3 times. Returns "" on failure.
#     """
#     if image is None:
#         return ""

#     # Resize (keeps aspect ratio)
#     h, w = image.shape[:2]
#     if max(h, w) > max_side:
#         scale  = max_side / max(h, w)
#         image  = cv2.resize(image, (int(w * scale), int(h * scale)),
#                             interpolation=cv2.INTER_AREA)

#     ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
#     if not ok:
#         print("[LINE] JPEG encode failed")
#         return ""

#     jpg_bytes = buf.tobytes()
#     print(f"[LINE] Uploading image ({len(jpg_bytes)//1024} KB)...")

#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             resp = requests.post(
#                 CATBOX_URL,
#                 data  = {"reqtype": "fileupload"},
#                 files = {"fileToUpload": ("zentra.jpg", jpg_bytes, "image/jpeg")},
#                 timeout = 20,
#             )
#             url = resp.text.strip()
#             if resp.status_code == 200 and url.startswith("http"):
#                 print(f"[LINE] Upload OK: {url}")
#                 return url
#             print(f"[LINE] Upload attempt {attempt} failed "
#                   f"({resp.status_code}): {resp.text[:80]}")
#         except requests.exceptions.Timeout:
#             print(f"[LINE] Upload timeout (attempt {attempt}/{MAX_RETRIES})")
#         except Exception as e:
#             print(f"[LINE] Upload error (attempt {attempt}): {e}")

#         if attempt < MAX_RETRIES:
#             time.sleep(1.5)

#     print("[LINE] All upload attempts failed -- sending text only")
#     return ""

SAVE_DIR = Path("D:/ZENTRA_V1/public_images")
BASE_URL = "https://singing-unusual-ipod.ngrok-free.dev"

def upload_image(image) -> str:
    """Upload image to local server via ngrok. image should be raw (no UI overlay)."""
    if image is None:
        return ""

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.jpg"
    filepath = SAVE_DIR / filename

    with open(filepath, "wb") as f:
        f.write(buf.tobytes())

    url = f"{BASE_URL}/{filename}"
    print(f"[LOCAL] {url}")
    return url


# ================================================================
# SEND TO ONE RECIPIENT
# ================================================================
def _push(group_id: str, msg: str, img_url: str, token: str) -> bool:
    headers  = {"Authorization": f"Bearer {token}",
                "Content-Type": "application/json"}
    messages: list[dict] = [{"type": "text", "text": msg}]
    if img_url:
        messages.append({
            "type":               "image",
            "originalContentUrl": img_url,
            "previewImageUrl":    img_url,
        })

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                LINE_PUSH_URL,
                headers = headers,
                json    = {"to": group_id, "messages": messages},
                timeout = 12,
            )
            if resp.status_code == 200:
                print(f"[LINE] Sent OK -> {group_id[:14]}...")
                return True
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "5"))
                print(f"[LINE] Rate limit -- waiting {wait}s")
                time.sleep(wait)
                continue
            print(f"[LINE] Push failed ({resp.status_code}): {resp.text[:120]}")
            return False
        except requests.exceptions.Timeout:
            print(f"[LINE] Push timeout (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(2)
        except Exception as e:
            print(f"[LINE] Push error: {e}")
            return False
    return False


# ================================================================
# DISPATCH ONE PAYLOAD
# ================================================================
def _dispatch(payload: dict):
    cfg   = _cfg()
    token = cfg.LINE_OA_CHANNEL_ACCESS_TOKEN
    if not token:
        print("[LINE] No token -- skipping dispatch")
        return

    img_url = upload_image(payload.get("image")) if payload.get("image") is not None else ""
    msg     = payload["msg"]

    for gid in payload["recipients"]:
        _push(gid, msg, img_url, token)


# ================================================================
# BACKGROUND SENDER THREAD
# ================================================================
def _sender_loop():
    while _running:
        payload = None
        with _queue_lock:
            if _queue:
                payload = _queue.pop(0)
        if payload:
            try:
                _dispatch(payload)
            except Exception:
                traceback.print_exc()
        else:
            time.sleep(0.1)


def start_sender():
    global _sender_thread, _running
    _running = True
    _sender_thread = threading.Thread(
        target=_sender_loop, daemon=True, name="LINE-Sender"
    )
    _sender_thread.start()
    print("[LINE] Sender thread started")


def stop_sender():
    global _running
    _running = False
    with _queue_lock:
        remaining = len(_queue)
    if remaining:
        print(f"[LINE] Flushing {remaining} queued items...")
        time.sleep(2.0)
    print("[LINE] Sender stopped")


# ================================================================
# PUBLIC API
# ================================================================
def send_line_notify(
    msg:          str,
    image                   = None,
    level:        str       = "warning",
    cooldown_key: str       = "default",
    cooldown_sec: float     = None,
) -> bool:
    cfg = _cfg()

    # Cooldown
    if cooldown_sec is None:
        cooldown_map = {
            cfg.ALERT_LEVEL_EMERGENCY: cfg.FALL_COOLDOWN_SECONDS,
            cfg.ALERT_LEVEL_ALERT:     cfg.ZONE_COOLDOWN_SECONDS,
            cfg.ALERT_LEVEL_WARNING:   cfg.VIOLATION_COOLDOWN_SECONDS,
        }
        cooldown_sec = cooldown_map.get(level, 30)

    now = time.time()
    with _lock:
        if now - _last_sent.get(cooldown_key, 0.0) < cooldown_sec:
            remaining = int(cooldown_sec - (now - _last_sent[cooldown_key]))
            print(f"[LINE] Cooldown '{cooldown_key}': {remaining}s left")
            return False
        _last_sent[cooldown_key] = now

    # Recipients
    ids = _recipients(level)
    if not ids:
        return False

    # Build message
    ts   = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    icon = {"warning": "⚠️", "alert": "🔴", "emergency": "🆘"}.get(level, "⚠️")
    full = f"{icon} [ZENTRA {level.upper()}]\n{msg}\n📅 {ts}"

    # Queue
    payload = {
        "msg":        full,
        "image":      image.copy() if image is not None else None,
        "recipients": ids,
    }
    with _queue_lock:
        _queue.append(payload)

    print(f"[LINE] Queued [{level}] -> {ids}")
    return True


# ================================================================
# DAILY REPORT
# ================================================================
def send_daily_report(stats: dict, report_image=None):
    cfg = _cfg()
    ts  = datetime.now().strftime("%d/%m/%Y")
    msg = (
        f"📊 ZENTRA Daily Safety Report\n"
        f"📅 Date: {ts}\n"
        f"{'━'*24}\n"
        f"🪖 PPE Violations   : {stats.get('ppe_violations', 0)}\n"
        f"⛔ Zone Intrusions  : {stats.get('zone_intrusions', 0)}\n"
        f"🆘 Fall Events      : {stats.get('fall_events', 0)}\n"
        f"{'━'*24}\n"
        f"✅ ZENTRA running normally"
    )
    ids     = [r for r in [cfg.LINE_OA_GROUP_SUPERVISOR,
                            cfg.LINE_OA_GROUP_SAFETY] if _valid(r)]
    img_url = upload_image(report_image) if report_image is not None else ""
    token   = cfg.LINE_OA_CHANNEL_ACCESS_TOKEN
    for gid in ids:
        _push(gid, msg, img_url, token)
    print(f"[LINE] Daily report -> {len(ids)} group(s)")


send_line_message = send_line_notify