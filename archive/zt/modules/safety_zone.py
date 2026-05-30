# modules/safety_zone.py — ZENTRA Safety Zone Module
# ✅ v2: ลบ cv2.imshow ออก — main.py จัดการ imshow เอง
# ================================================================

from __future__ import annotations
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

import config as cfg
from alerts.line_notify import send_line_notify
from utils.tracker      import ByteTracker
from utils.collector    import get_collector

# ── State ────────────────────────────────────────────────────────
zones:            list[dict] = []
_active_zone_idx: int        = -1
draw_mode:        bool       = False

_tracker = ByteTracker(
    track_thresh = cfg.BYTETRACK_TRACK_THRESH,
    track_buffer = cfg.BYTETRACK_TRACK_BUFFER,
    match_thresh = cfg.BYTETRACK_MATCH_THRESH,
)

stats        = {"intrusions": 0, "alerts_sent": 0}
_last_alert: float = 0.0

ZONE_COLORS = [
    (0, 0, 220), (220, 0, 0), (180, 0, 220),
    (0, 128, 220), (220, 128, 0), (0, 220, 128),
]


# ================================================================
# PERSISTENCE
# ================================================================
def _save_zones():
    pf = Path(cfg.ZONE_POLYGON_FILE)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(
        [{"points": z["points"], "name": z["name"]}
         for z in zones if z.get("ready")],
        indent=2,
    ))


def _load_zones():
    global zones
    pf = Path(cfg.ZONE_POLYGON_FILE)
    if not pf.exists():
        return
    try:
        data  = json.loads(pf.read_text())
        zones = [
            {"points": d["points"],
             "name":   d.get("name", f"Zone {i+1}"),
             "ready":  True}
            for i, d in enumerate(data)
        ]
        if zones:
            print(f"[Zone] Loaded {len(zones)} zone(s)")
    except Exception as e:
        print(f"[Zone] Load error: {e}")


_load_zones()


# ================================================================
# ZONE MANAGEMENT
# ================================================================
def toggle_draw_mode():
    global draw_mode, _active_zone_idx
    draw_mode = not draw_mode
    if draw_mode:
        if len(zones) >= cfg.MAX_ZONES:
            print(f"[Zone] Limit reached ({cfg.MAX_ZONES})")
            draw_mode = False
            return
        zones.append({"points": [], "name": f"Zone {len(zones)+1}",
                       "ready": False})
        _active_zone_idx = len(zones) - 1
        print(f"[Zone] Draw ON -> {zones[_active_zone_idx]['name']}")
        print("       Left click=point  Right click=save  Z=cancel")
    else:
        print("[Zone] Draw OFF")


def clear_all_zones():
    global zones, _active_zone_idx, draw_mode
    zones            = []
    _active_zone_idx = -1
    draw_mode        = False
    _save_zones()
    print("[Zone] All zones cleared")


def mouse_callback(event, x, y, flags, param):
    global draw_mode, _active_zone_idx
    if not draw_mode or _active_zone_idx < 0:
        return
    zone = zones[_active_zone_idx]
    if event == cv2.EVENT_LBUTTONDOWN:
        zone["points"].append([x, y])
        print(f"[Zone] +Point ({x},{y}) total={len(zone['points'])}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(zone["points"]) >= 3:
            zone["ready"] = True
            draw_mode     = False
            _save_zones()
            print(f"[Zone] ✅ {zone['name']} saved "
                  f"({len(zone['points'])} pts)")
        else:
            print("[Zone] Need at least 3 points")


# ================================================================
# GEOMETRY
# ================================================================
def _is_inside(zone: dict, cx: float, cy: float) -> bool:
    pts = zone.get("points", [])
    if not zone.get("ready") or len(pts) < 3:
        return False
    arr = np.array(pts, dtype=np.int32)
    return cv2.pointPolygonTest(arr, (float(cx), float(cy)), False) >= 0


# ================================================================
# DRAW
# ================================================================
def _draw_zones(frame: np.ndarray):
    for i, zone in enumerate(zones):
        pts = zone.get("points", [])
        if not pts:
            continue
        color   = ZONE_COLORS[i % len(ZONE_COLORS)]
        pts_arr = np.array(pts, dtype=np.int32)

        if zone.get("ready"):
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_arr], color)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts_arr], True, color, 2)
            cv2.putText(frame, zone["name"], tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        else:
            for pt in pts:
                cv2.circle(frame, tuple(pt), 5, (0, 240, 240), -1)
            if len(pts) > 1:
                cv2.polylines(frame, [pts_arr], False, (0, 240, 240), 2)


def draw_tracks(frame: np.ndarray, tracks):
    for t in tracks:
        x1,y1,x2,y2 = (int(t.bbox[0]), int(t.bbox[1]),
                        int(t.bbox[2]), int(t.bbox[3]))
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,185,0), 2)
        cv2.putText(frame, f"ID:{t.track_id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,185,0), 2, cv2.LINE_AA)
        if hasattr(t, "history") and t.history:
            history = list(t.history)
            if len(history) > 2:
                trail = np.array(
                    [(int(c[0]), int(c[1])) for c in history[-10:]],
                    dtype=np.int32
                )
                cv2.polylines(frame, [trail], False, (200,145,0), 1)


# ================================================================
# ON_FRAME — วาด zone overlay (ไม่ imshow)
# ================================================================
def on_frame(frame: np.ndarray, metadata, window_title: str = ""):
    """
    วาด zone polygons และ status text ลงบน frame
    ✅ ไม่เรียก cv2.imshow — main.py จัดการ imshow เอง
    """
    _draw_zones(frame)
    ready = [z for z in zones if z.get("ready")]

    if draw_mode:
        status = "[DRAW] Left=Point | Right=Save | Z=Cancel"
        color  = (0, 240, 240)
    elif not ready:
        status = "Press Z to draw Safety Zone"
        color  = (0, 0, 250)
    else:
        status = (f"Zones:{len(ready)} | "
                  f"Intrusions:{stats['intrusions']} | Z=Add | C=Clear")
        color  = (0, 200, 200)

    cv2.putText(frame, status, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


# ================================================================
# ON_DATA — ByteTrack + zone check + alert
# ================================================================
def on_data(data: dict, metadata, frame: Optional[np.ndarray] = None, raw_frame: Optional[np.ndarray] = None):
    global _last_alert

    ready_zones = [z for z in zones if z.get("ready")]
    if not ready_zones:
        return

    predictions = data.get("predictions") or []
    person_dets = [p for p in predictions
                   if p.get("class", "").lower() == "person"]
    tracks = _tracker.update(person_dets)

    if frame is not None:
        draw_tracks(frame, tracks)

    intruders = []
    for t in tracks:
        cx, cy = t.center
        for zone in ready_zones:
            if _is_inside(zone, cx, cy):
                intruders.append({"track_id": t.track_id,
                                   "zone": zone["name"]})
                break

    if not intruders:
        return

    stats["intrusions"] += len(intruders)

    # ✅ frame ที่รับมาเป็น annotated frame (มี bbox ครบ)
    if frame is not None:
        get_collector().collect(frame, predictions, "zone_intrusions")

    now = time.time()
    if now - _last_alert >= cfg.ZONE_COOLDOWN_SECONDS:
        _last_alert       = now
        stats["alerts_sent"] += 1
        count     = len(intruders)
        zone_list = ", ".join({i["zone"] for i in intruders})
        ids_str   = ", ".join({str(i["track_id"]) for i in intruders})
        print(f"[Zone] INTRUSION: {count} person(s) in {zone_list} (IDs:{ids_str})")

        # ส่ง raw_frame ไป LINE (ไม่มี UI overlay)
        alert_img = raw_frame if raw_frame is not None else frame
        send_line_notify(
            f"🔴 [ZENTRA ALERT]\n"
            f"⛔ ZENTRA Zone Alert\n"
            f"🚨 พบบุคคลเข้าเขตอันตราย {count} คน\n"
            f"📍 โซน {zone_list}",
            image        = alert_img,
            level        = cfg.ALERT_LEVEL_ALERT,
            cooldown_key = "red_zone",
            cooldown_sec = cfg.ZONE_COOLDOWN_SECONDS,
        )
