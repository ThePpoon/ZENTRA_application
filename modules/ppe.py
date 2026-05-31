# modules/ppe.py — ZENTRA PPE Detection Module
# Slide Module 1: YOLOv8m | 5 PPE items | mAP ≥ 85%
# Workflow: RTSP → Decode → Resize 640×640 → YOLOv8 → LINE alert
# ================================================================

from __future__ import annotations
import cv2
import time
import numpy as np
from typing import Optional

import config as cfg
from alerts.line_notify import send_line_notify
from utils.collector    import get_collector

# ── State ───────────────────────────────────────────────────
stats = {
    "frames":      0,
    "violations":  0,
    "alerts_sent": 0,
    "start_time":  time.time(),
}
_last_alert: float = 0.0
_violation_streak: int = 0   # consecutive frames with a violation (debounce)


# ================================================================
# HELPERS
# ================================================================
def get_fps() -> float:
    elapsed = time.time() - stats["start_time"]
    return stats["frames"] / elapsed if elapsed > 0 else 0.0


def _info(cls: str) -> dict:
    return cfg.PPE_CLASSES.get(cls, {"label": cls, "label_th": cls,
                                     "color": (160, 160, 160), "violation": False})


# ================================================================
# DRAW PREDICTIONS
# ================================================================
def draw_predictions(frame: np.ndarray, predictions: list[dict]) -> np.ndarray:
    for pred in predictions:
        cls  = pred.get("class", "")
        conf = pred.get("confidence", 0.0)
        x    = int(pred.get("x", 0))
        y    = int(pred.get("y", 0))
        w    = int(pred.get("width",  0))
        h    = int(pred.get("height", 0))

        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        info  = _info(cls)
        color = info["color"]
        label = f"{info['label']} {conf:.0%}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      cfg.FONT_SCALE, cfg.FONT_THICKNESS)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
                    (255, 255, 255), cfg.FONT_THICKNESS, cv2.LINE_AA)
    return frame


# ================================================================
# ON_FRAME — OSD overlay + imshow
# ================================================================
def on_frame(frame: np.ndarray, metadata, window_title: str):
    stats["frames"] += 1
    fps = get_fps()

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), cfg.OSD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    osd = (
        f"ZENTRA  |  FPS:{fps:.1f}  "
        f"Frames:{stats['frames']:,}  "
        f"Violations:{stats['violations']}  "
        f"Alerts:{stats['alerts_sent']}"
    )
    cv2.putText(frame, osd, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
                cfg.OSD_COLOR, cfg.FONT_THICKNESS, cv2.LINE_AA)
    if window_title:
        cv2.imshow(window_title, frame)


# ================================================================
# ON_DATA — violation detection + alert
# ================================================================
def on_data(data: dict, metadata, frame: Optional[np.ndarray] = None):
    global _last_alert, _violation_streak

    predictions: list[dict] = (
        data.get("predictions") or
        data.get("detection_predictions") or []
    )

    # Normal frame collection
    if not predictions:
        _violation_streak = 0
        if frame is not None:
            get_collector().collect_normal(frame, [], metadata.frame_id)
        return

    detected   = [p.get("class", "") for p in predictions]
    violations = [c for c in detected if _info(c)["violation"]]

    if detected:
        print(f"[PPE] Frame {metadata.frame_id}: {detected}")

    if not violations:
        _violation_streak = 0
        if frame is not None:
            get_collector().collect_normal(frame, predictions, metadata.frame_id)
        return

    # Debounce: require N consecutive violation frames before confirming
    # (one-frame misdetections no longer raise a false alarm)
    _violation_streak += 1
    confirm = getattr(cfg, "PPE_CONFIRM_FRAMES", 3)
    if _violation_streak < confirm:
        return

    # Confirmed violation
    stats["violations"] += 1
    missing_en = ", ".join(sorted({_info(v)["label"]    for v in violations}))
    missing_th = ", ".join(sorted({_info(v)["label_th"] for v in violations}))
    print(f"[PPE] ⚠️  VIOLATION: {missing_en}")

    # Auto-collect
    if frame is not None:
        get_collector().collect(frame, predictions, "ppe_violations")

    # Alert (cooldown)
    now = time.time()
    if now - _last_alert >= cfg.VIOLATION_COOLDOWN_SECONDS:
        _last_alert        = now
        stats["alerts_sent"] += 1
        msg = (
            f"🪖 ZENTRA PPE Alert\n"
            f"⚠️ ขาดอุปกรณ์ PPE:\n"
            f"   {missing_th}\n"
        )
        send_line_notify(
            msg,
            image        = frame,
            level        = cfg.ALERT_LEVEL_WARNING,
            cooldown_key = "ppe_violation",
            cooldown_sec = cfg.VIOLATION_COOLDOWN_SECONDS,
        )
