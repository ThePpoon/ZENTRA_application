# modules/ppe.py — ZENTRA PPE Detection Module
# Slide Module 1: YOLOv8m | 5 PPE items | mAP ≥ 85%
# Workflow: RTSP → Decode → Resize 640×640 → YOLOv8 → LINE alert
# ================================================================

from __future__ import annotations
import cv2
import time
import collections
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
_last_alert: float = 0.0       # legacy global cooldown (standalone fallback)
_violation_streak: int = 0     # legacy global streak (standalone fallback)

# Per-track confirmation + cooldown (used when the pipeline supplies shared
# tracks). Keyed by persistent track ID so two people never share one streak.
_viol_buffer:     dict = {}    # track_id -> deque[bool]  (violation present per frame)
_viol_labels:     dict = {}    # track_id -> set[str]     (violation classes seen)
_viol_last_alert: dict = {}    # track_id -> float        (last alert time)


# ================================================================
# HELPERS
# ================================================================
def get_fps() -> float:
    elapsed = time.time() - stats["start_time"]
    return stats["frames"] / elapsed if elapsed > 0 else 0.0


def _info(cls: str) -> dict:
    return cfg.PPE_CLASSES.get(cls, {"label": cls, "label_th": cls,
                                     "color": (160, 160, 160), "violation": False})


def _pred_box(p: dict) -> list:
    """Roboflow-style center box (x,y,w,h) → [x1,y1,x2,y2]."""
    x, y = p.get("x", 0), p.get("y", 0)
    w, h = p.get("width", 0), p.get("height", 0)
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def _overlap_inside(inner: list, outer: list) -> float:
    """Fraction of `inner` box area that lies inside `outer` box (0..1)."""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area  = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


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


_FONT = cv2.FONT_HERSHEY_SIMPLEX
# (chip label, violation class) for the per-person PPE checklist
_PPE_CHECK = [("HELMET", "No Helmet"), ("VEST", "No Vest"),
              ("GLOVES", "No Gloves"), ("GLASSES", "No Glasses")]


def _rounded_rect(img, p1, p2, color, thickness=2, r=12):
    """A rounded rectangle (cleaner than a plain box) — outline only."""
    x1, y1 = p1; x2, y2 = p2
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness, cv2.LINE_AA)


def _chip(frame, x, y, text, bg, fg=(255, 255, 255), scale=0.42, pad=5):
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
    cv2.rectangle(frame, (x, y), (x + tw + pad * 2, y + th + pad * 2), bg, -1)
    cv2.putText(frame, text, (x + pad, y + th + pad), _FONT, scale, fg, 1, cv2.LINE_AA)
    return x + tw + pad * 2 + 4   # next x (with gap)


def draw_person_status(frame: np.ndarray, predictions: list[dict], tracks=None) -> np.ndarray:
    """Professional per-person display: ONE rounded box per person + a header
    (ID · SAFE/PPE ALERT) + a PPE checklist (HELMET/VEST/GLOVES/GLASSES,
    green=worn, red=missing). ASCII only (cv2 can't render Thai); no confidence %."""
    H, W = frame.shape[:2]
    persons = [p for p in predictions if str(p.get("class", "")).lower() == "person"]
    viols   = [p for p in predictions if _info(p.get("class", ""))["violation"]]
    track_boxes = {t.track_id: list(t.bbox) for t in (tracks or [])}

    GREEN, RED, WHITE = (76, 175, 80), (48, 48, 229), (255, 255, 255)

    for person in persons:
        pb = _pred_box(person)
        x1 = max(0, min(int(pb[0]), W - 1)); y1 = max(0, min(int(pb[1]), H - 1))
        x2 = max(0, min(int(pb[2]), W - 1)); y2 = max(0, min(int(pb[3]), H - 1))
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue

        missing = {_info(v.get("class", ""))["label"]
                   for v in viols if _overlap_inside(_pred_box(v), pb) >= 0.30}
        ok   = not missing
        main = GREEN if ok else RED

        _rounded_rect(frame, (x1, y1), (x2, y2), main, 2, 14)

        # match a track id (display only)
        tid, best = None, 0.5
        for k, tb in track_boxes.items():
            ov = _overlap_inside(pb, tb)
            if ov >= best:
                best, tid = ov, k

        # ── header bar: ID · status ──
        head = (f"ID {tid}   " if tid is not None else "PERSON   ") + ("SAFE" if ok else "PPE ALERT")
        (hw, hh), _ = cv2.getTextSize(head, _FONT, 0.55, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + hw + 14, y1 + hh + 10), main, -1)
        cv2.putText(frame, head, (x1 + 7, y1 + hh + 3), _FONT, 0.55, WHITE, 1, cv2.LINE_AA)

        # ── PPE checklist row ──
        cx, cy = x1 + 2, y1 + hh + 14
        for label, neg in _PPE_CHECK:
            cx = _chip(frame, cx, cy, label, GREEN if neg not in missing else RED, WHITE)
            if cx > W - 50:
                break
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
    """Detect PPE violations. When the pipeline supplies shared person tracks
    (metadata.tracks), violations are associated to each tracked person and
    confirmed per-track (accurate for multi-worker scenes). Falls back to the
    legacy global behaviour when run standalone (no tracks)."""
    predictions: list[dict] = (
        data.get("predictions") or
        data.get("detection_predictions") or []
    )
    tracks = getattr(metadata, "tracks", None)
    if tracks is None:
        _on_data_legacy(predictions, metadata, frame)
    else:
        _on_data_tracked(predictions, tracks, metadata, frame)


def _on_data_tracked(predictions, tracks, metadata, frame):
    """Per-person PPE violation detection using shared track IDs."""
    viol_preds = [p for p in predictions if _info(p.get("class", ""))["violation"]]

    # No violation detections at all → a normal frame (collect for dataset)
    if not viol_preds and frame is not None:
        get_collector().collect_normal(frame, predictions, metadata.frame_id)

    required = getattr(cfg, "PPE_CONFIRM_FRAMES", 3)
    window   = getattr(cfg, "PPE_CONFIRM_WINDOW", 5)
    min_ov   = getattr(cfg, "PPE_ASSOC_OVERLAP", 0.30)

    track_boxes = {t.track_id: list(t.bbox) for t in tracks}
    current_ids = set(track_boxes)

    # Associate each violation box to the person track it sits inside (best overlap).
    # A violation that belongs to no tracked person is ignored (validity gate —
    # removes floating / unattributable detections, a key false-positive source).
    per_track: dict[int, set] = {}
    for vp in viol_preds:
        vbox = _pred_box(vp)
        best_id, best_ov = None, min_ov
        for tid, tb in track_boxes.items():
            ov = _overlap_inside(vbox, tb)
            if ov >= best_ov:
                best_ov, best_id = ov, tid
        if best_id is not None:
            per_track.setdefault(best_id, set()).add(vp.get("class", ""))

    now      = time.time()
    cooldown = getattr(cfg, "VIOLATION_COOLDOWN_SECONDS", 30)

    for tid in current_ids:
        buf   = _viol_buffer.setdefault(tid, collections.deque(maxlen=window))
        viols = per_track.get(tid)
        buf.append(bool(viols))
        if viols:
            _viol_labels.setdefault(tid, set()).update(viols)
        elif sum(buf) == 0:
            _viol_labels.pop(tid, None)          # person is fully compliant again

        # Confirmed = violated in ≥ `required` of the last `window` frames
        if sum(buf) >= required and (now - _viol_last_alert.get(tid, 0.0)) >= cooldown:
            _viol_last_alert[tid] = now
            _raise_ppe_alert(tid, _viol_labels.get(tid) or set(viols or []),
                             predictions, frame)

    # Drop state for tracks that disappeared (avoid stale streaks / leaks)
    for tid in list(_viol_buffer):
        if tid not in current_ids:
            _viol_buffer.pop(tid, None)
            _viol_labels.pop(tid, None)
            _viol_last_alert.pop(tid, None)


def _raise_ppe_alert(track_id, viol_classes, predictions, frame):
    stats["violations"] += 1
    missing_en = ", ".join(sorted({_info(v)["label"]    for v in viol_classes})) or "PPE"
    missing_th = ", ".join(sorted({_info(v)["label_th"] for v in viol_classes})) or "PPE"
    print(f"[PPE] ⚠️  VIOLATION (ID:{track_id}): {missing_en}")

    if frame is not None:
        get_collector().collect(frame, predictions, "ppe_violations")

    stats["alerts_sent"] += 1
    msg = (
        f"🪖 ZENTRA PPE Alert\n"
        f"⚠️ พบผู้ไม่สวม PPE (ID:{track_id})\n"
        f"   {missing_th}\n"
    )
    send_line_notify(
        msg,
        image        = frame,
        level        = cfg.ALERT_LEVEL_WARNING,
        cooldown_key = f"ppe_violation_{track_id}",   # per-track cooldown
        cooldown_sec = cfg.VIOLATION_COOLDOWN_SECONDS,
    )


def _on_data_legacy(predictions, metadata, frame):
    """Original global-streak behaviour (standalone use, no shared tracks)."""
    global _last_alert, _violation_streak

    if not predictions:
        _violation_streak = 0
        if frame is not None:
            get_collector().collect_normal(frame, [], metadata.frame_id)
        return

    detected   = [p.get("class", "") for p in predictions]
    violations = [c for c in detected if _info(c)["violation"]]

    if not violations:
        _violation_streak = 0
        if frame is not None:
            get_collector().collect_normal(frame, predictions, metadata.frame_id)
        return

    _violation_streak += 1
    if _violation_streak < getattr(cfg, "PPE_CONFIRM_FRAMES", 3):
        return

    stats["violations"] += 1
    missing_th = ", ".join(sorted({_info(v)["label_th"] for v in violations}))
    print(f"[PPE] ⚠️  VIOLATION: {missing_th}")
    if frame is not None:
        get_collector().collect(frame, predictions, "ppe_violations")

    now = time.time()
    if now - _last_alert >= cfg.VIOLATION_COOLDOWN_SECONDS:
        _last_alert = now
        stats["alerts_sent"] += 1
        msg = (f"🪖 ZENTRA PPE Alert\n⚠️ ขาดอุปกรณ์ PPE:\n   {missing_th}\n")
        send_line_notify(msg, image=frame, level=cfg.ALERT_LEVEL_WARNING,
                         cooldown_key="ppe_violation",
                         cooldown_sec=cfg.VIOLATION_COOLDOWN_SECONDS)
