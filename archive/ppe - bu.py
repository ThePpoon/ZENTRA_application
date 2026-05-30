# modules/ppe.py — Module 1: PPE Detection

import cv2
import time
from config import PPE_CLASSES, VIOLATION_COOLDOWN_SECONDS
from alerts.line_notify import send_line_notify

# ---------- Stats ----------
stats = {
    "frames": 0,
    "violations": 0,
    "start_time": time.time(),
}


def get_fps() -> float:
    elapsed = time.time() - stats["start_time"]
    return stats["frames"] / elapsed if elapsed > 0 else 0.0


def on_frame(frame, metadata, window_title: str):
    """วาด overlay stats บน frame แล้ว imshow"""
    stats["frames"] += 1

    fps = get_fps()
    text = f"FPS: {fps:.1f}  |  Frames: {stats['frames']}  |  Violations: {stats['violations']}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow(window_title, frame)


def on_data(data: dict, metadata, frame=None):
    """
    วิเคราะห์ predictions จาก Roboflow workflow
    - นับ violations
    - ส่ง LINE alert (มี cooldown กัน spam)
    """
    predictions = data.get("predictions") or data.get("detection_predictions") or []

    if not predictions:
        return

    detected = [p.get("class", "") for p in predictions]
    violations = [cls for cls in detected if cls.startswith("no_")]

    if violations:
        stats["violations"] += 1
        missing = [PPE_CLASSES.get(v, {}).get("label", v) for v in violations]
        missing_str = ", ".join(missing)

        print(f"⚠️  Frame {metadata.frame_id}: ขาด PPE — {missing_str}")

        # LINE Notify
        msg = (
            f"⚠️ [ZENTRA PPE Alert]\n"
            f"Frame: {metadata.frame_id}\n"
            f"ขาดอุปกรณ์: {missing_str}\n"
            f"เวลา: {time.strftime('%H:%M:%S')}"
        )
        send_line_notify(
            msg,
            image=frame,
            cooldown_key="ppe_violation",
            cooldown_sec=VIOLATION_COOLDOWN_SECONDS,
        )

    # log ทุก 100 frames
    if stats["frames"] % 100 == 0:
        print(f"📊 {stats['frames']} frames | {stats['violations']} violations | FPS: {get_fps():.1f}")
