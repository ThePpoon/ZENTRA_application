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
_yolo_fall_streak:    int   = 0       # consecutive inferred frames with a YOLO fall box


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

    # ── YOLO streak (shared by 'yolo' and 'hybrid') ─────────────
    n_falls = _count_yolo_falls(predictions)
    if n_falls > 0:
        _yolo_fall_streak += 1
    else:
        _yolo_fall_streak = 0

    need = max(1, getattr(cfg, "FALL_YOLO_CONFIRM_FRAMES", 4))

    # ── YOLO-only: temporal confirmation, ignore pose ──────────
    if mode == "yolo":
        if _yolo_fall_streak >= need:
            _trigger_alert(max(1, n_falls), "YOLO Fall", frame, predictions)
            _yolo_fall_streak = 0
        return

    # ── Hybrid (balanced): YOLO primary + pose cross-check ─────
    #   • YOLO confirmed over `need` frames → fire
    #   • OR YOLO half-confirmed AND pose agrees → fire sooner
    #   • pose alone never fires (kills pose-only false alarms)
    half = (need + 1) // 2
    fire = (_yolo_fall_streak >= need) or (_yolo_fall_streak >= half and _pose_risk_now)
    if fire:
        method = "Hybrid (YOLO+Pose)" if _pose_risk_now else "Hybrid (YOLO)"
        _trigger_alert(max(1, n_falls), method, frame, predictions)
        _yolo_fall_streak = 0
        _fall_confirm_counter = 0


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
