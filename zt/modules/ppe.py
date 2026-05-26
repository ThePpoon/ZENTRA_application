# # modules/ppe.py — ZENTRA PPE Detection Module (v2 — Profile Aware)
# # ================================================================
# # ใช้ ppe_config.py เลือก PPE ที่จะตรวจ
# # แก้ PPE_PROFILE ใน ppe_config.py ได้เลย ไม่ต้องแก้ไฟล์นี้
# # ================================================================

# from __future__ import annotations
# import cv2
# import time
# import numpy as np
# from typing import Optional

# import config as cfg
# from ppe_config import (
#     get_active_classes,
#     get_required_ppe,
#     is_violation,
#     describe_profile,
#     print_profile_summary,
#     PPE_PROFILE,
#     MIN_VIOLATIONS_TO_ALERT,
# )
# from alerts.line_notify import send_line_notify
# from utils.collector    import get_collector

# # ── Load active PPE classes from profile ────────────────────────
# _PPE_CLASSES: dict[str, dict] = get_active_classes()
# _REQUIRED_PPE: set[str]       = get_required_ppe()

# # ── State ────────────────────────────────────────────────────────
# stats = {
#     "frames":      0,
#     "violations":  0,
#     "alerts_sent": 0,
#     "start_time":  time.time(),
# }
# _last_alert: float = 0.0

# # ── Print profile on load ─────────────────────────────────────────
# print_profile_summary()


# # ================================================================
# # HELPERS
# # ================================================================
# def get_fps() -> float:
#     elapsed = time.time() - stats["start_time"]
#     return stats["frames"] / elapsed if elapsed > 0 else 0.0


# def _info(cls: str) -> dict:
#     """คืน metadata ของ class (ใช้ active classes จาก profile)"""
#     return _PPE_CLASSES.get(
#         cls,
#         {"label": cls, "label_th": cls,
#          "color": (160, 160, 160), "violation": False}
#     )


# def reload_profile():
#     """
#     โหลด profile ใหม่ (เรียกหลังแก้ ppe_config.py แบบ hot-reload)
#     ตัวอย่าง:
#         import ppe_config
#         ppe_config.PPE_PROFILE = "helmet_vest"
#         import modules.ppe as ppe
#         ppe.reload_profile()
#     """
#     global _PPE_CLASSES, _REQUIRED_PPE
#     import importlib, ppe_config
#     importlib.reload(ppe_config)
#     _PPE_CLASSES  = ppe_config.get_active_classes()
#     _REQUIRED_PPE = ppe_config.get_required_ppe()
#     print(f"[PPE] 🔄 Profile reloaded → {ppe_config.PPE_PROFILE}")
#     ppe_config.print_profile_summary()


# # ================================================================
# # DRAW PREDICTIONS
# # ================================================================
# def draw_predictions(frame: np.ndarray,
#                      predictions: list[dict]) -> np.ndarray:
#     for pred in predictions:
#         cls  = pred.get("class", "")
#         conf = pred.get("confidence", 0.0)

#         # ข้ามถ้า class ไม่ได้อยู่ใน profile ที่เปิดใช้
#         if cls not in _PPE_CLASSES:
#             continue

#         x    = int(pred.get("x", 0))
#         y    = int(pred.get("y", 0))
#         w    = int(pred.get("width",  0))
#         h    = int(pred.get("height", 0))
#         x1, y1 = x - w // 2, y - h // 2
#         x2, y2 = x + w // 2, y + h // 2

#         info  = _info(cls)
#         color = info["color"]
#         label = f"{info['label']} {conf:.0%}"

#         # Box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         # Label background
#         (tw, th), _ = cv2.getTextSize(
#             label, cv2.FONT_HERSHEY_SIMPLEX,
#             cfg.FONT_SCALE, cfg.FONT_THICKNESS)
#         ly = max(y1 - 4, th + 4)
#         cv2.rectangle(frame,
#                       (x1, ly - th - 4), (x1 + tw + 6, ly),
#                       color, -1)
#         cv2.putText(frame, label, (x1 + 3, ly - 2),
#                     cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
#                     (255, 255, 255), cfg.FONT_THICKNESS, cv2.LINE_AA)
#     return frame


# # ================================================================
# # ON_FRAME — OSD overlay + imshow
# # ================================================================
# def on_frame(frame: np.ndarray, metadata, window_title: str = ""):
#     """
#     นับ frame และ update stats เท่านั้น
#     ✅ ไม่เรียก cv2.imshow — main.py จัดการ imshow + OSD เอง
#     """
#     stats["frames"] += 1


# # ================================================================
# # ON_DATA — violation detection + alert
# # ================================================================
# def on_data(data: dict, metadata, frame: Optional[np.ndarray] = None, raw_frame: Optional[np.ndarray] = None):
#     global _last_alert

#     predictions: list[dict] = (
#         data.get("predictions") or
#         data.get("detection_predictions") or []
#     )

#     # Normal frame collection
#     if not predictions:
#         if frame is not None:
#             get_collector().collect_normal(frame, [], metadata.frame_id)
#         return

#     detected   = [p.get("class", "") for p in predictions]

#     # กรองเฉพาะ class ที่อยู่ใน profile ที่เปิดใช้
#     detected   = [c for c in detected if c in _PPE_CLASSES]
#     violations = [c for c in detected if _info(c)["violation"]]

#     if detected:
#         # Log เฉพาะ violation
#         v_labels = [_info(v)["label"] for v in violations]
#         if v_labels:
#             print(f"[PPE] Frame {metadata.frame_id}: VIOLATION "
#                   f"{v_labels}")

#     if not violations:
#         if frame is not None:
#             get_collector().collect_normal(frame, predictions,
#                                            metadata.frame_id)
#         return

#     # ตรวจ threshold จำนวน violation
#     # MIN_VIOLATIONS_TO_ALERT=0 หมายถึง alert ทันทีที่มี >= 1 violation
#     threshold = max(MIN_VIOLATIONS_TO_ALERT, 1)
#     if len(violations) < threshold:
#         return

#     # Violation found
#     stats["violations"] += 1
#     missing_en = ", ".join(sorted({_info(v)["label"]    for v in violations}))
#     missing_th = ", ".join(sorted({_info(v)["label_th"] for v in violations}))
#     print(f"[PPE] ⚠️  VIOLATION: {missing_en}")

#     # Auto-collect (เก็บ annotated เพื่อ debug)
#     if frame is not None:
#         get_collector().collect(frame, predictions, "ppe_violations")

#     # Alert (cooldown)
#     now = time.time()
#     if now - _last_alert >= cfg.VIOLATION_COOLDOWN_SECONDS:
#         _last_alert        = now
#         stats["alerts_sent"] += 1

#         # สร้างข้อความตาม profile
#         active_th = describe_profile()
#         msg = (
#             f"🪖 ZENTRA PPE Alert\n"
#             f"⚠️ ขาดอุปกรณ์ PPE:\n"
#             f"   {missing_th}\n"
#             f"📋 Profile: {active_th}\n"
#         )
#         # ส่ง raw_frame ไป LINE (ไม่มี UI overlay) ถ้าไม่มีให้ใช้ frame
#         alert_img = raw_frame if raw_frame is not None else frame
#         send_line_notify(
#             msg,
#             image        = alert_img,
#             level        = cfg.ALERT_LEVEL_WARNING,
#             cooldown_key = "ppe_violation",
#             cooldown_sec = cfg.VIOLATION_COOLDOWN_SECONDS,
#         )

""" """

# modules/ppe.py — ZENTRA PPE Detection Module (v3 — FIXED LOGIC)
# ================================================================
# ✅ FIX:
#   - เปลี่ยนเป็น "missing PPE detection"
#   - ไม่พึ่ง no_helmet / no_vest จาก model แล้ว
# ================================================================

from __future__ import annotations
import cv2
import time
import numpy as np
from typing import Optional

import config as cfg
from ppe_config import (
    get_active_classes,
    get_required_ppe,
    describe_profile,
    print_profile_summary,
    PPE_PROFILE,
    MIN_VIOLATIONS_TO_ALERT,
)
from alerts.line_notify import send_line_notify
from utils.collector    import get_collector

# ── Load active PPE classes ─────────────────────────────────────
_PPE_CLASSES: dict[str, dict] = get_active_classes()
_REQUIRED_PPE: set[str]       = get_required_ppe()

# ── State ───────────────────────────────────────────────────────
stats = {
    "frames":      0,
    "violations":  0,
    "alerts_sent": 0,
    "start_time":  time.time(),
}
_last_alert: float = 0.0

print_profile_summary()


# ================================================================
# HELPERS
# ================================================================
def get_fps() -> float:
    elapsed = time.time() - stats["start_time"]
    return stats["frames"] / elapsed if elapsed > 0 else 0.0


def _info(cls: str) -> dict:
    return _PPE_CLASSES.get(
        cls,
        {"label": cls, "label_th": cls,
         "color": (160, 160, 160), "violation": False}
    )


def reload_profile():
    global _PPE_CLASSES, _REQUIRED_PPE
    import importlib, ppe_config
    importlib.reload(ppe_config)
    _PPE_CLASSES  = ppe_config.get_active_classes()
    _REQUIRED_PPE = ppe_config.get_required_ppe()
    print(f"[PPE] 🔄 Profile reloaded → {ppe_config.PPE_PROFILE}")
    ppe_config.print_profile_summary()


# ================================================================
# DRAW
# ================================================================
def draw_predictions(frame: np.ndarray, predictions: list[dict]) -> np.ndarray:
    for pred in predictions:
        cls  = pred.get("class", "")
        conf = pred.get("confidence", 0.0)

        if cls not in _PPE_CLASSES:
            continue

        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))
        w = int(pred.get("width",  0))
        h = int(pred.get("height", 0))
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        info  = _info(cls)
        color = info["color"]
        label = f"{info['label']} {conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            cfg.FONT_SCALE, cfg.FONT_THICKNESS)

        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
                    (255, 255, 255), cfg.FONT_THICKNESS, cv2.LINE_AA)

    return frame


# ================================================================
# ON_FRAME
# ================================================================
def on_frame(frame: np.ndarray, metadata, window_title: str = ""):
    stats["frames"] += 1


# ================================================================
# ON_DATA — FIXED LOGIC
# ================================================================
def on_data(data: dict, metadata,
            frame: Optional[np.ndarray] = None,
            raw_frame: Optional[np.ndarray] = None):

    global _last_alert

    predictions: list[dict] = (
        data.get("predictions") or
        data.get("detection_predictions") or []
    )

    if not predictions:
        if frame is not None:
            get_collector().collect_normal(frame, [], metadata.frame_id)
        return

    # ── STEP 1: collect detected classes ─────────────────────────
    detected = [p.get("class", "") for p in predictions]

    # filter only active classes
    detected = [c for c in detected if c in _PPE_CLASSES]
    detected_set = set(detected)

    # ── STEP 2: check missing PPE ────────────────────────────────
    missing = []
    for req in _REQUIRED_PPE:
        if req not in detected_set:
            missing.append(req)

    if not missing:
        if frame is not None:
            get_collector().collect_normal(frame, predictions, metadata.frame_id)
        return

    # ── STEP 3: apply threshold ─────────────────────────────────
    threshold = max(MIN_VIOLATIONS_TO_ALERT, 1)
    if len(missing) < threshold:
        return

    stats["violations"] += 1

    missing_en = ", ".join(sorted([_info(m)["label"] for m in missing]))
    missing_th = ", ".join(sorted([_info(m)["label_th"] for m in missing]))

    print(f"[PPE] ⚠️  MISSING: {missing_en}")

    # collect annotated (debug)
    if frame is not None:
        get_collector().collect(frame, predictions, "ppe_violations")

    # ── ALERT ───────────────────────────────────────────────────
    now = time.time()
    if now - _last_alert >= cfg.VIOLATION_COOLDOWN_SECONDS:
        _last_alert = now
        stats["alerts_sent"] += 1

        msg = (
            f"🪖 ZENTRA PPE Alert\n"
            f"📋 การตรวจ: {describe_profile()}\n"
            f"⚠️ ขาดอุปกรณ์ PPE:\n"
            f"   {missing_th}"
        )

        alert_img = raw_frame if raw_frame is not None else frame

        send_line_notify(
            msg,
            image        = alert_img,
            level        = cfg.ALERT_LEVEL_WARNING,
            cooldown_key = "ppe_violation",
            cooldown_sec = cfg.VIOLATION_COOLDOWN_SECONDS,
        )