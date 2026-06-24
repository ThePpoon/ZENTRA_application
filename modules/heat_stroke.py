# modules/heat_stroke.py — ZENTRA Heat Stroke Risk Detection
# Slide Module 3: MediaPipe Pose (33 keypoints)
# 3 รูปแบบ:
#   1. Sudden Fall     — Keypoint Velocity + Joint Angle
#   2. Abnormal Posture — Bounding Box Ratio
#   3. Gait Anomaly    — Center of Gravity Trajectory
# Hybrid: Rule-based + ML (slide mentions LSTM)
# ================================================================

from __future__ import annotations
import cv2
import time
import math
import collections
import numpy as np
from typing import Optional

import config as cfg
from alerts.line_notify import send_line_notify
from utils.collector    import get_collector

# ── MediaPipe (graceful fallback) ───────────────────────────
try:
    import mediapipe as mp
    _mp_pose   = mp.solutions.pose
    _mp_draw   = mp.solutions.drawing_utils
    _mp_styles = mp.solutions.drawing_styles
    _pose_model = _mp_pose.Pose(
        model_complexity         = cfg.MEDIAPIPE_MODEL_COMPLEXITY,
        enable_segmentation      = False,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )
    MP_OK = True
    print("[HeatStroke] ✅ MediaPipe Pose loaded")
except Exception as _e:
    MP_OK = False
    print(f"[HeatStroke] MediaPipe unavailable ({_e}) → Roboflow fallback")

if MP_OK:
    _PL       = _mp_pose.PoseLandmark
    HIP_L     = _PL.LEFT_HIP.value
    HIP_R     = _PL.RIGHT_HIP.value
    NOSE      = _PL.NOSE.value
    ANKLE_L   = _PL.LEFT_ANKLE.value
    ANKLE_R   = _PL.RIGHT_ANKLE.value
    SHOULDER_L = _PL.LEFT_SHOULDER.value
    SHOULDER_R = _PL.RIGHT_SHOULDER.value


# ================================================================
# PER-PERSON POSE STATE
# Slide: COG Trajectory / Keypoint Velocity per person
# ================================================================
class _PoseState:
    __slots__ = ("pid", "cog_history", "kp_history", "fall_hits")

    def __init__(self, pid: int = 0):
        self.pid         = pid
        self.cog_history = collections.deque(maxlen=cfg.GAIT_HISTORY_FRAMES)
        self.kp_history  = collections.deque(maxlen=12)
        self.fall_hits   = 0

    def update_kp(self, arr: np.ndarray):
        self.kp_history.append(arr.copy())

    def update_cog(self, nx: float, ny: float):
        self.cog_history.append((nx, ny))

    @property
    def kp_velocity(self) -> float:
        """ความเร็วเฉลี่ย keypoints (Sudden Fall detection)"""
        if len(self.kp_history) < 2:
            return 0.0
        prev, curr = self.kp_history[-2], self.kp_history[-1]
        vis  = (prev[:, 2] > 0.5) & (curr[:, 2] > 0.5)
        if vis.sum() < 3:
            return 0.0
        return float(np.mean(np.linalg.norm(curr[vis, :2] - prev[vis, :2], axis=1)))

    @property
    def cog_variance(self) -> float:
        """COG trajectory variance (Gait Anomaly detection)"""
        if len(self.cog_history) < 10:
            return 0.0
        pts = np.array(self.cog_history)
        return float(np.var(pts[:, 0]) + np.var(pts[:, 1]))


_pose_states: dict[int, _PoseState] = {0: _PoseState(0)}

# ── Module State ────────────────────────────────────────────
stats = {"falls": 0, "alerts_sent": 0, "frames_analyzed": 0}
_last_alert:          float = 0.0
_fall_confirm_counter: int  = 0
_pose_risk_now:       bool  = False   # pose flagged abnormal posture/velocity this frame
_yolo_fall_streak:    int   = 0       # consecutive inferred frames with a YOLO fall box (legacy/global)

# Per-track fall state (used when the pipeline supplies shared tracks)
_fall_buffer:       dict  = {}        # track_id -> deque[bool] (fall box on this person per frame)
_fall_last_alert:   dict  = {}        # track_id -> last alert time
_fall_unattr_streak: int  = 0         # streak of falls NOT attributable to any track (recall net)
_fall_unattr_last:  float = 0.0       # last unattributed-fall alert time


# ================================================================
# YOLO FALL HELPERS (hybrid fusion)
# ================================================================
_NEG_TOKENS = ("no", "not", "stand", "normal", "ok", "safe", "upright")


def _is_fall_class(name: str) -> bool:
    """True if a Roboflow class name means 'fallen' (and not a negative like
    'no fall' / 'standing')."""
    n = (name or "").lower()
    return ("fall" in n or "lying" in n or "down" in n) and not any(t in n for t in _NEG_TOKENS)


def _count_yolo_falls(predictions: list) -> int:
    import config as _c
    thr = getattr(_c, "FALL_YOLO_CONFIDENCE", 0.5)
    return sum(1 for p in (predictions or [])
               if _is_fall_class(p.get("class", "")) and p.get("confidence", 0.0) >= thr)


def _pred_box(p: dict) -> list:
    """Roboflow-style center box (x,y,w,h) → [x1,y1,x2,y2]."""
    x, y = p.get("x", 0), p.get("y", 0)
    w, h = p.get("width", 0), p.get("height", 0)
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def _overlap_inside(inner: list, outer: list) -> float:
    """Fraction of `inner` box area inside `outer` box (0..1)."""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area  = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


# ================================================================
# GEOMETRY HELPERS
# ================================================================
def _angle(a, b, c) -> float:
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=np.float32)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=np.float32)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def _lm(lm_list, idx: int, w: int, h: int):
    lm = lm_list[idx]
    return lm.x * w, lm.y * h


# ================================================================
# POSE ANALYSIS (MediaPipe)
# ================================================================
def _analyze_pose(lm_list, w: int, h: int, state: _PoseState) -> dict:
    """
    Slide ตรวจ 3 รูปแบบ:
    1. Sudden Fall     (Keypoint Velocity + Joint Angles)
    2. Abnormal Posture (BBox Ratio)
    3. Gait Anomaly    (COG Trajectory)
    """
    result = {"sudden_fall": False, "abnormal_posture": False,
              "gait_anomaly": False, "details": ""}

    # array (33, 3): x, y, visibility
    arr = np.array([[lm.x, lm.y, lm.visibility] for lm in lm_list], dtype=np.float32)
    state.update_kp(arr)

    # COG = midpoint of hips
    hlx, hly = _lm(lm_list, HIP_L, w, h)
    hrx, hry = _lm(lm_list, HIP_R, w, h)
    state.update_cog((hlx + hrx) / 2 / w, (hly + hry) / 2 / h)

    vis = arr[:, 2] > 0.40

    # ── 1. Abnormal Posture (BBox Ratio) ────────────────────
    if vis.sum() > 6:
        xs, ys = arr[vis, 0], arr[vis, 1]
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        if bh > 1e-3:
            ratio = bw / bh
            if ratio > cfg.FALL_BBOX_RATIO_THRESH:
                result["abnormal_posture"] = True
                result["details"] += f"ratio={ratio:.2f} "

    # ── 2. Sudden Fall (Keypoint Velocity) ──────────────────
    vel = state.kp_velocity
    if vel > cfg.FALL_KEYPOINT_VELOCITY_THRESH:
        result["sudden_fall"] = True
        result["details"] += f"vel={vel:.3f} "

    # ── 3. Gait Anomaly (COG Variance) ──────────────────────
    var = state.cog_variance
    if var > cfg.GAIT_ANOMALY_THRESH:
        result["gait_anomaly"] = True
        result["details"] += f"var={var:.3f} "

    return result


# ================================================================
# DRAW HELPERS
# ================================================================
def _draw_landmarks(frame: np.ndarray, results):
    if not MP_OK or not results.pose_landmarks:
        return
    _mp_draw.draw_landmarks(
        frame, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec = _mp_styles.get_default_pose_landmarks_style(),
    )


def draw_fall_predictions(frame: np.ndarray, predictions: list[dict]) -> np.ndarray:
    """วาด Roboflow fall model boxes (fallback path)"""
    for pred in predictions:
        cls  = pred.get("class", "").lower()
        conf = pred.get("confidence", 0.0)
        x, y = int(pred.get("x", 0)), int(pred.get("y", 0))
        w, h  = int(pred.get("width", 0)), int(pred.get("height", 0))
        x1, y1, x2, y2 = x-w//2, y-h//2, x+w//2, y+h//2

        is_fall     = "fall" in cls
        color, label = ((0,0,230), f"FALL {conf:.0%}") if is_fall else ((0,200,0), f"OK {conf:.0%}")
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return frame


# ================================================================
# ON_FRAME — MediaPipe per-frame analysis
# ================================================================
def on_frame(frame: np.ndarray, metadata, window_title: str):
    if not MP_OK:
        return

    stats["frames_analyzed"] += 1
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _pose_model.process(rgb)
    _draw_landmarks(frame, results)

    global _fall_confirm_counter, _pose_risk_now

    if not results.pose_landmarks:
        _pose_risk_now = False

    if results.pose_landmarks:
        h_f, w_f = frame.shape[:2]
        lm        = results.pose_landmarks.landmark
        state     = _pose_states.setdefault(0, _PoseState(0))
        analysis  = _analyze_pose(lm, w_f, h_f, state)

        # Confirm counter
        is_risk = analysis["sudden_fall"] or analysis["abnormal_posture"]
        _pose_risk_now = is_risk
        if is_risk:
            _fall_confirm_counter += 1
            cv2.putText(
                frame,
                f"⚠ FALL RISK [{_fall_confirm_counter}/{cfg.FALL_CONFIRM_FRAMES}]",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 40, 210), 2, cv2.LINE_AA,
            )
        else:
            _fall_confirm_counter = max(0, _fall_confirm_counter - 1)

        if analysis["gait_anomaly"]:
            cv2.putText(frame, "⚠ GAIT ANOMALY",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 110, 210), 2, cv2.LINE_AA)

        # COG trail
        if len(state.cog_history) > 2:
            trail = np.array([
                (int(c[0] * w_f), int(c[1] * h_f))
                for c in list(state.cog_history)[-15:]
            ], dtype=np.int32)
            cv2.polylines(frame, [trail], False, (0, 210, 210), 2)


# ================================================================
# ON_DATA
# ================================================================
def on_data(data: dict, metadata, frame: Optional[np.ndarray] = None):
    global _last_alert, _fall_confirm_counter, _yolo_fall_streak
    global _fall_unattr_streak, _fall_unattr_last

    predictions = data.get("predictions") or []
    mode = getattr(cfg, "FALL_MODE", "hybrid").lower()

    # Force pose-only if MediaPipe is the only thing available
    if not MP_OK and mode != "yolo":
        mode = "yolo"

    # ── Pose-only: original MediaPipe behaviour ─────────────────
    if mode == "pose":
        if MP_OK and _fall_confirm_counter >= cfg.FALL_CONFIRM_FRAMES:
            _trigger_alert(1, "MediaPipe Pose", frame)
            _fall_confirm_counter = 0
        return

    thr = getattr(cfg, "FALL_YOLO_CONFIDENCE", 0.5)
    fall_boxes = [p for p in predictions
                  if _is_fall_class(p.get("class", "")) and p.get("confidence", 0.0) >= thr]
    need = max(1, getattr(cfg, "FALL_YOLO_CONFIRM_FRAMES", 4))
    half = (need + 1) // 2
    tracks = getattr(metadata, "tracks", None)

    # ── Legacy global path (standalone, no shared tracks) ───────
    if tracks is None:
        _yolo_fall_streak = _yolo_fall_streak + 1 if fall_boxes else 0
        if mode == "yolo":
            if _yolo_fall_streak >= need:
                _trigger_alert(max(1, len(fall_boxes)), "YOLO Fall", frame, predictions)
                _yolo_fall_streak = 0
            return
        fire = (_yolo_fall_streak >= need) or (_yolo_fall_streak >= half and _pose_risk_now)
        if fire:
            method = "Hybrid (YOLO+Pose)" if _pose_risk_now else "Hybrid (YOLO)"
            _trigger_alert(max(1, len(fall_boxes)), method, frame, predictions)
            _yolo_fall_streak = 0
            _fall_confirm_counter = 0
        return

    # ── Per-track path: confirm a fall PER PERSON (multi-worker) ─
    window   = getattr(cfg, "FALL_CONFIRM_WINDOW", need + 2)
    min_ov   = getattr(cfg, "FALL_ASSOC_OVERLAP", 0.30)
    cooldown = getattr(cfg, "FALL_COOLDOWN_SECONDS", 15)
    now      = time.time()

    track_boxes = {t.track_id: list(t.bbox) for t in tracks}
    current_ids = set(track_boxes)

    # Associate each fall box to the person track it overlaps most
    attributed: set = set()
    unattributed = 0
    for fb in fall_boxes:
        box = _pred_box(fb)
        best_id, best = None, min_ov
        for tid, tb in track_boxes.items():
            ov = _overlap_inside(box, tb)
            if ov >= best:
                best, best_id = ov, tid
        if best_id is not None:
            attributed.add(best_id)
        else:
            unattributed += 1

    # Per-track confirmation + per-track cooldown
    for tid in current_ids:
        buf = _fall_buffer.setdefault(tid, collections.deque(maxlen=window))
        buf.append(tid in attributed)
        if mode == "yolo":
            fire = sum(buf) >= need
        else:  # hybrid — pose cross-check lets it fire sooner (pose alone never fires)
            fire = (sum(buf) >= need) or (sum(buf) >= half and _pose_risk_now)
        if fire and (now - _fall_last_alert.get(tid, 0.0)) >= cooldown:
            _fall_last_alert[tid] = now
            method = ("Hybrid (YOLO+Pose)" if (mode != "yolo" and _pose_risk_now)
                      else ("YOLO Fall" if mode == "yolo" else "Hybrid (YOLO)"))
            _send_fall_alert(1, f"{method} ID:{tid}", frame, predictions, f"fall_{tid}")

    # Recall safety net: falls not attributable to any track still alert
    _fall_unattr_streak = _fall_unattr_streak + 1 if unattributed > 0 else 0
    if _fall_unattr_streak >= need and (now - _fall_unattr_last) >= cooldown:
        _fall_unattr_last = now
        _send_fall_alert(unattributed, "YOLO Fall (untracked)", frame, predictions, "fall_unattr")
        _fall_unattr_streak = 0

    # Prune state for tracks that disappeared
    for tid in list(_fall_buffer):
        if tid not in current_ids:
            _fall_buffer.pop(tid, None)
            _fall_last_alert.pop(tid, None)


def _trigger_alert(count: int, method: str, frame, preds=None):
    global _last_alert
    now = time.time()
    if now - _last_alert < cfg.FALL_COOLDOWN_SECONDS:
        return
    _last_alert           = now
    stats["falls"]       += count
    stats["alerts_sent"] += 1

    if frame is not None:
        get_collector().collect(frame, preds or [], "fall_events", force=True)

    print(f"[HeatStroke] 🆘 FALL ALERT: {count} person(s) [{method}]")
    msg = (
        f"🆘 ZENTRA Emergency Alert\n"
        f"🚨 ตรวจพบการล้ม / หมดสติ {count} ราย\n"
        f"⏰ ต้องช่วยเหลือภายใน 30 นาที!\n"
    )
    send_line_notify(
        msg,
        image        = frame,
        level        = cfg.ALERT_LEVEL_EMERGENCY,
        cooldown_key = "fall_detection",
        cooldown_sec = cfg.FALL_COOLDOWN_SECONDS,
    )


def _send_fall_alert(count: int, method: str, frame, preds, cooldown_key: str):
    """Emit a fall EMERGENCY. Caller has already applied its (per-track or
    global) cooldown gate; cooldown_key keeps the LINE sender de-duplicated
    per person too."""
    stats["falls"]       += count
    stats["alerts_sent"] += 1
    if frame is not None:
        get_collector().collect(frame, preds or [], "fall_events", force=True)
    print(f"[HeatStroke] 🆘 FALL ALERT: {count} person(s) [{method}]")
    msg = (
        f"🆘 ZENTRA Emergency Alert\n"
        f"🚨 ตรวจพบการล้ม / หมดสติ {count} ราย\n"
        f"⏰ ต้องช่วยเหลือภายใน 30 นาที!\n"
    )
    send_line_notify(
        msg,
        image        = frame,
        level        = cfg.ALERT_LEVEL_EMERGENCY,
        cooldown_key = cooldown_key,
        cooldown_sec = cfg.FALL_COOLDOWN_SECONDS,
    )
