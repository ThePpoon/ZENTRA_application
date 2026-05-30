# modules/heat_stroke.py — ZENTRA Heat Stroke Risk Detection (v3)
# ================================================================
# แก้ไข v3:
#   1. MediaPipe dual-API — รองรับทั้ง < 0.10 (solutions) และ 0.10+ (tasks)
#   2. ลบ cv2.imshow ออกจาก module — imshow ทำที่ main.py เท่านั้น
#   3. on_frame ไม่ต้องการ window_title (ยังคง signature เพื่อ backward compat)
#   4. Sitting classifier คงไว้ครบ
#   5. Risk decay + confirmation buffer คงไว้
# ================================================================

from __future__ import annotations
import cv2, time, math, collections
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

import config as cfg
from alerts.line_notify import send_line_notify
from utils.collector    import get_collector


# ================================================================
# THRESHOLDS
# ================================================================
class TH:
    FALL_VELOCITY         = 0.55
    FALL_VELOCITY_CONFIRM = 3
    PRONE_RATIO           = 1.50
    PRONE_CONFIRM         = 5
    GAIT_ANOMALY          = 0.045
    GAIT_MIN_FRAMES       = 20
    MOTIONLESS_WARN_SEC   = 20.0
    MOTIONLESS_EMERG_SEC  = 40.0
    MOTIONLESS_MOVE_THRESH= 0.008
    SIT_KNEE_ANGLE_MIN    = 55.0
    SIT_HIP_ANGLE_MIN     = 65.0
    CROSS_VAL_DISCOUNT    = 0.5
    RISK_DECAY_RATE       = 0.15
    ALERT_MIN_SCORE       = 0.60


# ================================================================
# MediaPipe — dual API init (รองรับ 0.9.x และ 0.10+)
# ================================================================
_mp_pose_model  = None
_mp_process_fn  = None   # callable(bgr_frame) → mp_result | None
_mp_draw_fn     = None   # callable(frame, result)
MP_OK           = False

# ── shared draw references (ถ้า legacy API) ──────────────────────
_mp_pose   = None
_mp_draw   = None
_mp_styles = None


def _init_mediapipe():
    """ลอง legacy API ก่อน → ถ้าไม่ได้ลอง tasks API"""
    global _mp_pose_model, _mp_process_fn, _mp_draw_fn, MP_OK
    global _mp_pose, _mp_draw, _mp_styles

    # ── วิธีที่ 1: Legacy solutions API (mediapipe < 0.10) ───────
    try:
        import mediapipe as mp
        _mp_pose   = mp.solutions.pose
        _mp_draw   = mp.solutions.drawing_utils
        _mp_styles = mp.solutions.drawing_styles
        _mp_pose_model = _mp_pose.Pose(
            model_complexity         = cfg.MEDIAPIPE_MODEL_COMPLEXITY,
            enable_segmentation      = False,
            smooth_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )

        def _process_legacy(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return _mp_pose_model.process(rgb)

        def _draw_legacy(frame, result):
            if result and result.pose_landmarks:
                _mp_draw.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    _mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=_mp_styles.get_default_pose_landmarks_style(),
                )

        _mp_process_fn = _process_legacy
        _mp_draw_fn    = _draw_legacy
        MP_OK = True
        print("[HeatStroke] ✅ MediaPipe Pose loaded (legacy API)")
        return
    except AttributeError:
        pass  # solutions ไม่มี → ลอง tasks
    except Exception as e:
        print(f"[HeatStroke] ⚠️  MediaPipe legacy: {e}")

    # ── วิธีที่ 2: Tasks API (mediapipe 0.10+) ────────────────────
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # Download model ถ้ายังไม่มี
        from pathlib import Path as _Path
        import urllib.request as _req
        BASE = _Path(__file__).parent.parent
        model_path = BASE / "data" / "pose_landmarker_lite.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            print("[HeatStroke] 📥 Downloading MediaPipe pose model (~8MB)...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/1/"
                   "pose_landmarker_lite.task")
            _req.urlretrieve(url, str(model_path))
            print(f"[HeatStroke] ✅ Downloaded → {model_path.name}")

        base_opts  = mp_python.BaseOptions(model_asset_path=str(model_path))
        opts       = mp_vision.PoseLandmarkerOptions(
            base_options                  = base_opts,
            output_segmentation_masks     = False,
            num_poses                     = 4,
            min_pose_detection_confidence = 0.5,
            min_tracking_confidence       = 0.5,
        )
        _landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

        # Wrap เพื่อให้ interface เหมือน legacy (คืน object ที่มี .pose_landmarks)
        class _FakeResult:
            def __init__(self, lms): self.pose_landmarks = lms

        def _process_tasks(bgr):
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = _landmarker.detect(mp_img)
            if res.pose_landmarks:
                return _FakeResult(res.pose_landmarks[0])
            return _FakeResult(None)

        def _draw_tasks(frame, result):
            if result and result.pose_landmarks:
                h, w = frame.shape[:2]
                lms  = result.pose_landmarks
                for lm in lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 200, 200), -1)

        _mp_process_fn = _process_tasks
        _mp_draw_fn    = _draw_tasks
        _mp_pose_model = _landmarker
        MP_OK = True
        print("[HeatStroke] ✅ MediaPipe Pose loaded (tasks API 0.10+)")
        return
    except Exception as e:
        print(f"[HeatStroke] ⚠️  MediaPipe tasks API: {e}")

    print("[HeatStroke] ⚠️  MediaPipe ไม่พร้อม — ใช้ YOLOv8-pose อย่างเดียว")


_init_mediapipe()


# ================================================================
# Layer 1 — YOLOv8-pose
# ================================================================
_yolo_pose_model = None

def _load_yolo_pose():
    global _yolo_pose_model
    if _yolo_pose_model is not None:
        return _yolo_pose_model
    try:
        from ultralytics import YOLO
        from pathlib import Path as _P
        local = _P(cfg.MODELS_DIR) / "pose_finetuned.pt" \
                if hasattr(cfg, "MODELS_DIR") else None
        weights = str(local) if (local and local.exists()) else "yolov8m-pose.pt"
        _yolo_pose_model = YOLO(weights)
        print(f"[HeatStroke] ✅ YOLOv8-pose loaded: {weights}")
    except Exception as e:
        print(f"[HeatStroke] ⚠️  YOLOv8-pose unavailable: {e}")
        _yolo_pose_model = None
    return _yolo_pose_model


# ================================================================
# COCO Keypoint Indices (YOLO)
# ================================================================
KP = {
    "nose":0,"left_eye":1,"right_eye":2,"left_ear":3,"right_ear":4,
    "left_shoulder":5,"right_shoulder":6,
    "left_elbow":7,"right_elbow":8,
    "left_wrist":9,"right_wrist":10,
    "left_hip":11,"right_hip":12,
    "left_knee":13,"right_knee":14,
    "left_ankle":15,"right_ankle":16,
}


# ================================================================
# DATA CLASSES
# ================================================================
@dataclass
class PoseSnapshot:
    timestamp:  float
    bbox:       tuple
    keypoints:  np.ndarray
    bbox_ratio: float
    cog:        tuple
    is_mp:      bool  = False
    is_sitting: bool  = False
    knee_angle: float = 180.0
    hip_angle:  float = 180.0


@dataclass
class ConfirmBuffer:
    fall_frames:  int = 0
    prone_frames: int = 0
    gait_frames:  int = 0


@dataclass
class PersonState:
    person_id:        int
    snapshots:        collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=90))
    confirm:          ConfirmBuffer = field(default_factory=ConfirmBuffer)
    motionless_since: Optional[float] = None
    last_alert_type:  str   = ""
    smoothed_risk:    float = 0.0

    @property
    def kp_velocity(self) -> float:
        if len(self.snapshots) < 2:
            return 0.0
        a, b = self.snapshots[-2], self.snapshots[-1]
        if a.keypoints.shape != b.keypoints.shape:
            return 0.0
        vis  = (a.keypoints[:,2] > 0.35) & (b.keypoints[:,2] > 0.35)
        if vis.sum() < 4:
            return 0.0
        dt   = max(b.timestamp - a.timestamp, 0.033)
        diff = np.linalg.norm(b.keypoints[vis,:2] - a.keypoints[vis,:2], axis=1)
        return float(np.mean(diff) / dt)

    @property
    def cog_variance(self) -> float:
        if len(self.snapshots) < TH.GAIT_MIN_FRAMES:
            return 0.0
        pts = np.array([s.cog for s in list(self.snapshots)[-40:]])
        return float(np.var(pts[:,0]) + np.var(pts[:,1]))

    @property
    def motionless_seconds(self) -> float:
        return 0.0 if self.motionless_since is None \
               else time.time() - self.motionless_since

    @property
    def latest_bbox_ratio(self) -> float:
        return self.snapshots[-1].bbox_ratio if self.snapshots else 1.0

    @property
    def is_currently_sitting(self) -> bool:
        if not self.snapshots:
            return False
        recent = list(self.snapshots)[-5:]
        return sum(1 for s in recent if s.is_sitting) >= 3

    def update_motionless(self):
        if len(self.snapshots) < 5:
            return
        cogs     = np.array([s.cog for s in list(self.snapshots)[-5:]])
        movement = float(np.mean(np.std(cogs, axis=0)))
        if movement < TH.MOTIONLESS_MOVE_THRESH:
            if self.motionless_since is None:
                self.motionless_since = time.time()
        else:
            self.motionless_since = None

    def update_smoothed_risk(self, raw_risk: float):
        alpha = 0.35
        if raw_risk > self.smoothed_risk:
            self.smoothed_risk = alpha*raw_risk + (1-alpha)*self.smoothed_risk
        else:
            self.smoothed_risk = max(
                0.0,
                self.smoothed_risk - TH.RISK_DECAY_RATE*(self.smoothed_risk - raw_risk + 0.1)
            )
        self.smoothed_risk = min(1.0, self.smoothed_risk)


# ================================================================
# GEOMETRY HELPERS
# ================================================================
def _angle3(a, b, c) -> float:
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=np.float32)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=np.float32)
    n  = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-6:
        return 180.0
    return float(math.degrees(math.acos(np.clip(np.dot(ba,bc)/n, -1.0, 1.0))))


def _classify_posture_yolo(kp: np.ndarray) -> tuple[bool, float, float]:
    knee_angle = hip_angle = 180.0
    is_sitting = False
    angles_knee, angles_hip = [], []
    for hip_k, knee_k, ankle_k, shoulder_k in [
        ("left_hip","left_knee","left_ankle","left_shoulder"),
        ("right_hip","right_knee","right_ankle","right_shoulder"),
    ]:
        h, k, a, s = kp[KP[hip_k]], kp[KP[knee_k]], kp[KP[ankle_k]], kp[KP[shoulder_k]]
        if h[2]>0.35 and k[2]>0.35 and a[2]>0.35:
            angles_knee.append(_angle3(h[:2], k[:2], a[:2]))
        if s[2]>0.35 and h[2]>0.35 and k[2]>0.35:
            angles_hip.append(_angle3(s[:2], h[:2], k[:2]))
    if angles_knee: knee_angle = float(np.mean(angles_knee))
    if angles_hip:  hip_angle  = float(np.mean(angles_hip))
    if (TH.SIT_KNEE_ANGLE_MIN <= knee_angle <= 155.0 and
            TH.SIT_HIP_ANGLE_MIN <= hip_angle <= 155.0):
        is_sitting = True
    return is_sitting, knee_angle, hip_angle


def _classify_posture_mp(result) -> tuple[bool, float, float]:
    """รองรับทั้ง legacy result และ tasks result"""
    if not MP_OK or result is None or result.pose_landmarks is None:
        return False, 180.0, 180.0

    lms = result.pose_landmarks
    # landmark index (ใช้ได้ทั้ง legacy PoseLandmark list และ tasks NormalizedLandmark list)
    IDX = {"ls":11,"rs":12,"lh":23,"rh":24,"lk":25,"rk":26,"la":27,"ra":28}

    def pt(i):
        lm = lms[i]
        return np.array([lm.x, lm.y]), getattr(lm, "visibility", 1.0)

    angles_knee, angles_hip = [], []
    try:
        ls, ls_v = pt(IDX["ls"]); rs, rs_v = pt(IDX["rs"])
        lh, lh_v = pt(IDX["lh"]); rh, rh_v = pt(IDX["rh"])
        lk, lk_v = pt(IDX["lk"]); rk, rk_v = pt(IDX["rk"])
        la, la_v = pt(IDX["la"]); ra, ra_v = pt(IDX["ra"])
        if lh_v>0.5 and lk_v>0.5 and la_v>0.5: angles_knee.append(_angle3(lh,lk,la))
        if rh_v>0.5 and rk_v>0.5 and ra_v>0.5: angles_knee.append(_angle3(rh,rk,ra))
        if ls_v>0.5 and lh_v>0.5 and lk_v>0.5: angles_hip.append(_angle3(ls,lh,lk))
        if rs_v>0.5 and rh_v>0.5 and rk_v>0.5: angles_hip.append(_angle3(rs,rh,rk))
    except (IndexError, AttributeError):
        pass

    knee_angle = float(np.mean(angles_knee)) if angles_knee else 180.0
    hip_angle  = float(np.mean(angles_hip))  if angles_hip  else 180.0
    is_sitting = (TH.SIT_KNEE_ANGLE_MIN<=knee_angle<=155.0 and
                  TH.SIT_HIP_ANGLE_MIN<=hip_angle<=155.0)
    return is_sitting, knee_angle, hip_angle


def _cog_from_yolo_kp(kp: np.ndarray, w: int, h: int) -> tuple:
    lh, rh = kp[KP["left_hip"]], kp[KP["right_hip"]]
    if lh[2]>0.3 and rh[2]>0.3:
        return ((lh[0]+rh[0])/2/w, (lh[1]+rh[1])/2/h)
    vis = kp[kp[:,2]>0.3]
    if len(vis)==0: return (0.5, 0.5)
    return (float(vis[:,0].mean()/w), float(vis[:,1].mean()/h))


def _cog_from_mp(result, w: int, h: int) -> tuple:
    if not MP_OK or result is None or result.pose_landmarks is None:
        return (0.5, 0.5)
    try:
        lh = result.pose_landmarks[23]
        rh = result.pose_landmarks[24]
        return ((lh.x+rh.x)/2, (lh.y+rh.y)/2)
    except (IndexError, AttributeError):
        return (0.5, 0.5)


# ================================================================
# RULE ENGINE
# ================================================================
@dataclass
class DetectionResult:
    person_id:   int
    sudden_fall: bool  = False
    prone:       bool  = False
    gait_anomaly:bool  = False
    motionless:  bool  = False
    is_sitting:  bool  = False
    risk_score:  float = 0.0
    details:     str   = ""

    @property
    def is_emergency(self) -> bool:
        if self.is_sitting: return False
        return self.motionless or (self.sudden_fall and self.prone)

    @property
    def is_warning(self) -> bool:
        if self.is_sitting: return False
        return self.gait_anomaly or self.prone or self.sudden_fall


def _rule_engine(state: PersonState) -> DetectionResult:
    res = DetectionResult(person_id=state.person_id)
    res.is_sitting = state.is_currently_sitting

    # 1. Sudden Fall
    vel = state.kp_velocity
    if vel > TH.FALL_VELOCITY and not res.is_sitting:
        state.confirm.fall_frames += 1
    else:
        state.confirm.fall_frames = max(0, state.confirm.fall_frames - 2)
    if state.confirm.fall_frames >= TH.FALL_VELOCITY_CONFIRM:
        res.sudden_fall  = True
        res.details     += f"vel={vel:.3f}(×{state.confirm.fall_frames}) "
        res.risk_score  += 0.35

    # 2. Prone
    ratio = state.latest_bbox_ratio
    if ratio > TH.PRONE_RATIO and not res.is_sitting:
        state.confirm.prone_frames += 1
    else:
        state.confirm.prone_frames = max(0, state.confirm.prone_frames - 2)
    if state.confirm.prone_frames >= TH.PRONE_CONFIRM:
        res.prone       = True
        res.details    += f"ratio={ratio:.2f}(×{state.confirm.prone_frames}) "
        res.risk_score += 0.30

    # 3. Gait Anomaly
    var = state.cog_variance
    if var > TH.GAIT_ANOMALY and len(state.snapshots) >= TH.GAIT_MIN_FRAMES:
        state.confirm.gait_frames += 1
        if state.confirm.gait_frames >= 8:
            res.gait_anomaly = True
            res.details     += f"cog_var={var:.4f} "
            res.risk_score  += 0.20
    else:
        state.confirm.gait_frames = max(0, state.confirm.gait_frames - 1)

    # 4. Motionless
    still_sec = state.motionless_seconds
    warn_sec  = TH.MOTIONLESS_WARN_SEC  * (2.0 if res.is_sitting else 1.0)
    emerg_sec = TH.MOTIONLESS_EMERG_SEC * (2.0 if res.is_sitting else 1.0)
    if still_sec >= warn_sec:
        res.motionless  = True
        extra = min((still_sec - warn_sec) / emerg_sec, 1.0)
        res.risk_score += 0.15 + extra * 0.35
        res.details    += f"still={still_sec:.0f}s "

    if res.is_sitting and state.snapshots:
        snap = state.snapshots[-1]
        res.details += f"sit(K={snap.knee_angle:.0f}° H={snap.hip_angle:.0f}°) "

    res.risk_score = min(res.risk_score, 1.0)
    state.update_smoothed_risk(res.risk_score)
    res.risk_score = state.smoothed_risk
    return res


# ================================================================
# MODULE STATE
# ================================================================
_person_states: dict[int, PersonState] = {}
_global_state  = PersonState(person_id=0)

stats = {
    "falls":             0,
    "prone_events":      0,
    "gait_events":       0,
    "motionless_events": 0,
    "alerts_sent":       0,
    "frames_analyzed":   0,
    "sitting_filtered":  0,
}
_last_alert:  float = 0.0
_fall_confirm: int  = 0


# ================================================================
# DRAW HELPERS
# ================================================================
def _draw_status(frame: np.ndarray, states: list[PersonState],
                 results: list[DetectionResult]):
    h_f, w_f = frame.shape[:2]
    for res in results:
        state = _person_states.get(res.person_id, _global_state)
        if not state.snapshots:
            continue
        snap = state.snapshots[-1]
        x1, y1, x2, y2 = snap.bbox

        if res.is_sitting:
            box_color = (0, 200, 80);  label = f"SIT ID:{res.person_id}"
        elif res.is_emergency:
            box_color = (0, 0, 230);   label = f"EMERGENCY ID:{res.person_id}"
        elif res.is_warning:
            box_color = (0, 140, 230); label = f"WARNING ID:{res.person_id}"
        else:
            box_color = (0, 200, 80);  label = f"OK ID:{res.person_id}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2, cv2.LINE_AA)

        # Risk bar
        bar_w = int((x2-x1) * res.risk_score)
        bar_y = y2 + 4
        cv2.rectangle(frame, (x1, bar_y), (x1+bar_w, bar_y+6), box_color, -1)
        cv2.putText(frame, f"{res.risk_score*100:.0f}%",
                    (x1, bar_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    box_color, 1, cv2.LINE_AA)

        # Badges
        if not res.is_sitting:
            badge_y = y2 + 22
            bx = x1
            # for text, col in [
            #     ("FALL",  (0,0,220))   if res.sudden_fall  else None,
            #     ("PRONE", (0,80,220))  if res.prone        else None,
            #     ("GAIT",  (0,160,220)) if res.gait_anomaly else None,
            #     (f"STILL {state.motionless_seconds:.0f}s", (0,40,200))
            #         if res.motionless else None,
            # ]:
            for text, col in filter(None, [
                ("FALL",  (0,0,220))   if res.sudden_fall  else None,
                ("PRONE", (0,80,220))  if res.prone        else None,
                ("GAIT",  (0,160,220)) if res.gait_anomaly else None,
                (f"STILL {state.motionless_seconds:.0f}s", (0,40,200))
                    if res.motionless else None,
            ]):
                if text is None: continue
                (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(frame, (bx, badge_y-th-2), (bx+tw+6, badge_y+2), col, -1)
                cv2.putText(frame, text, (bx+3, badge_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1, cv2.LINE_AA)
                bx += tw + 8

        if snap.knee_angle < 170:
            cv2.putText(frame,
                f"K:{snap.knee_angle:.0f}° H:{snap.hip_angle:.0f}°",
                (x1, y2+42), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (160,160,160), 1, cv2.LINE_AA)

    # COG trail
    for state in states:
        if len(state.snapshots) < 3:
            continue
        trail = np.array([
            (int(s.cog[0]*w_f), int(s.cog[1]*h_f))
            for s in list(state.snapshots)[-20:]
        ], dtype=np.int32)
        cv2.polylines(frame, [trail], False, (0,200,200), 2)


def draw_fall_predictions(frame: np.ndarray,
                          predictions: list[dict]) -> np.ndarray:
    """วาด Roboflow fall model (fallback)"""
    for pred in predictions:
        cls  = pred.get("class","").lower()
        conf = pred.get("confidence", 0.0)
        x,y  = int(pred.get("x",0)), int(pred.get("y",0))
        w,h  = int(pred.get("width",0)), int(pred.get("height",0))
        x1,y1,x2,y2 = x-w//2, y-h//2, x+w//2, y+h//2
        color = (0,0,220) if "fall" in cls else (0,200,0)
        label = f"FALL {conf:.0%}" if "fall" in cls else f"OK {conf:.0%}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    return frame


# ================================================================
# ALERT LOGIC
# ================================================================
def _should_alert(res: DetectionResult,
                  state: PersonState) -> tuple[bool, str, str]:
    if res.risk_score < TH.ALERT_MIN_SCORE or res.is_sitting:
        return False, "", ""

    event_key = ""
    if res.is_emergency:
        event_key = "motionless" if res.motionless else "prone_fall"
    elif res.sudden_fall:   event_key = "sudden_fall"
    elif res.gait_anomaly:  event_key = "gait"
    elif res.prone:         event_key = "prone"
    if not event_key:
        return False, "", ""

    if time.time() - _last_alert < cfg.FALL_COOLDOWN_SECONDS:
        return False, "", ""

    still_sec = state.motionless_seconds
    if res.is_emergency:
        level = cfg.ALERT_LEVEL_EMERGENCY
        lines = ["🆘 ZENTRA Emergency Alert"]
        if res.motionless:
            lines.append(f"🚨 ตรวจพบคนไม่ขยับ {still_sec:.0f} วินาที!")
        if res.prone:
            lines.append("🚨 ตรวจพบท่าทางผิดปกติ (นอน/ล้ม)")
        if res.sudden_fall:
            lines.append("🚨 ตรวจพบการล้มกะทันหัน")
        lines.append("⏰ ต้องช่วยเหลือภายใน 30 นาที!")
    else:
        level = cfg.ALERT_LEVEL_WARNING
        lines = ["⚠️ ZENTRA Heat Stroke Warning"]
        if res.sudden_fall:
            lines.append("⚠️ ตรวจพบการเคลื่อนไหวผิดปกติ (เสี่ยงล้ม)")
        if res.prone:
            lines.append("⚠️ ตรวจพบท่าทางผิดปกติ (เสี่ยงหมดแรง)")
        if res.gait_anomaly:
            lines.append("⚠️ ตรวจพบการเดินโซเซผิดปกติ")

    return True, level, "\n".join(lines)


# ================================================================
# ON_FRAME — วาด skeleton + status overlay (ไม่ imshow)
# ================================================================
def on_frame(frame: np.ndarray, metadata, window_title: str = ""):
    """
    วาด skeleton และ status overlay ลงบน frame โดยตรง
    ✅ ไม่เรียก cv2.imshow — main.py จัดการ imshow เอง
    """
    global _fall_confirm
    stats["frames_analyzed"] += 1
    h_f, w_f = frame.shape[:2]

    # ── Layer 1: YOLOv8-pose ──────────────────────────────────
    yolo_states: list[PersonState] = []
    try:
        model = _load_yolo_pose()
        if model is not None:
            results = model.predict(frame, conf=0.40, iou=0.45,
                                    verbose=False)
            if results and results[0].keypoints is not None:
                boxes = results[0].boxes
                kps   = results[0].keypoints.data.cpu().numpy()
                for i in range(len(boxes)):
                    pid = int(boxes[i].id.item()) \
                          if boxes[i].id is not None else i
                    x1,y1,x2,y2 = boxes[i].xyxy[0].cpu().numpy().astype(int)
                    bw,bh = max(x2-x1,1), max(y2-y1,1)
                    ratio = bw/bh
                    kp    = kps[i] if i<len(kps) else np.zeros((17,3))
                    cog   = _cog_from_yolo_kp(kp, w_f, h_f)
                    is_sitting, knee_a, hip_a = _classify_posture_yolo(kp)

                    snap = PoseSnapshot(
                        timestamp=time.time(), bbox=(x1,y1,x2,y2),
                        keypoints=kp, bbox_ratio=ratio, cog=cog,
                        is_mp=False, is_sitting=is_sitting,
                        knee_angle=knee_a, hip_angle=hip_a,
                    )
                    if pid not in _person_states:
                        _person_states[pid] = PersonState(person_id=pid)
                    _person_states[pid].snapshots.append(snap)
                    _person_states[pid].update_motionless()
                    yolo_states.append(_person_states[pid])

                # วาด skeleton จาก YOLO
                try:
                    ann = results[0].plot(
                        boxes=True, labels=True, conf=True, img=frame)
                    frame[:] = ann
                except Exception:
                    pass

            # ลบ stale states
            now    = time.time()
            active = {s.person_id for s in yolo_states}
            for pid in [p for p,st in _person_states.items()
                        if p not in active and st.snapshots
                        and now - st.snapshots[-1].timestamp > 5.0]:
                del _person_states[pid]

    except Exception as e:
        print(f"[HeatStroke] YOLO frame error: {e}")

    # ── Layer 2: MediaPipe ────────────────────────────────────
    mp_result = None
    if MP_OK and _mp_process_fn:
        run_mp = (not yolo_states) or any(
            not s.is_currently_sitting and
            (s.confirm.fall_frames>1 or s.confirm.prone_frames>1
             or s.motionless_since is not None)
            for s in yolo_states
        )
        if run_mp:
            try:
                mp_result = _mp_process_fn(frame)
            except Exception as e:
                print(f"[HeatStroke] MediaPipe error: {e}")

            if mp_result and mp_result.pose_landmarks is not None:
                # วาด landmarks
                if _mp_draw_fn:
                    _mp_draw_fn(frame, mp_result)

                if not yolo_states:
                    # ใช้ MediaPipe เป็นหลัก
                    arr  = np.array(
                        [[lm.x, lm.y, getattr(lm,"visibility",1.0)]
                         for lm in mp_result.pose_landmarks],
                        dtype=np.float32)
                    vis  = arr[:,2]>0.40
                    xs,ys = arr[vis,0], arr[vis,1]
                    if vis.sum() >= 6:
                        bw    = xs.max()-xs.min()
                        bh    = ys.max()-ys.min()
                        ratio = bw/bh if bh>1e-3 else 1.0
                        cog   = _cog_from_mp(mp_result, w_f, h_f)
                        is_sitting,ka,ha = _classify_posture_mp(mp_result)
                        snap = PoseSnapshot(
                            timestamp=time.time(),
                            bbox=(int(xs.min()*w_f), int(ys.min()*h_f),
                                  int(xs.max()*w_f), int(ys.max()*h_f)),
                            keypoints=arr, bbox_ratio=ratio, cog=cog,
                            is_mp=True, is_sitting=is_sitting,
                            knee_angle=ka, hip_angle=ha,
                        )
                        _global_state.snapshots.append(snap)
                        _global_state.update_motionless()
                        yolo_states = [_global_state]
                else:
                    # Cross-validate
                    mp_sit,_,_ = _classify_posture_mp(mp_result)
                    if mp_sit:
                        for s in yolo_states:
                            s.smoothed_risk *= TH.CROSS_VAL_DISCOUNT
                            stats["sitting_filtered"] += 1

    # ── Layer 3: Rule Engine + Draw ───────────────────────────
    det_results = [_rule_engine(s) for s in yolo_states]
    _draw_status(frame, yolo_states, det_results)

    # ── MediaPipe-only fallback counter ───────────────────────
    if not yolo_states and MP_OK:
        gs = _rule_engine(_global_state)
        if gs.sudden_fall or gs.prone:
            _fall_confirm += 1
        else:
            _fall_confirm = max(0, _fall_confirm-1)
        if _fall_confirm > 0:
            cv2.putText(frame,
                f"⚠ FALL RISK [{_fall_confirm}/{cfg.FALL_CONFIRM_FRAMES}]",
                (10,95), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (0,40,210), 2, cv2.LINE_AA)


# ================================================================
# ON_DATA — alert dispatch
# ================================================================
def on_data(data: dict, metadata, frame: Optional[np.ndarray] = None, raw_frame: Optional[np.ndarray] = None):
    global _last_alert, _fall_confirm

    all_states = list(_person_states.values()) or \
                 ([_global_state] if MP_OK else [])

    for state in all_states:
        if not state.snapshots:
            continue
        dr = _rule_engine(state)

        if dr.is_sitting and dr.risk_score > 0.1:
            stats["sitting_filtered"] += 1
        if not (dr.is_warning or dr.is_emergency):
            continue

        should, level, msg = _should_alert(dr, state)
        if not should:
            continue

        if dr.sudden_fall:  stats["falls"] += 1
        if dr.prone:        stats["prone_events"] += 1
        if dr.gait_anomaly: stats["gait_events"] += 1
        if dr.motionless:   stats["motionless_events"] += 1
        stats["alerts_sent"] += 1
        _last_alert = time.time()

        # ✅ frame ที่ส่งมาเป็น annotated frame แล้ว (มี bbox)
        if frame is not None:
            get_collector().collect(frame, [], "fall_events", force=True)

        print(f"[HeatStroke] 🆘 ALERT ID:{state.person_id} "
              f"level={level} score={dr.risk_score:.2f}")
        print(f"             {dr.details}")
        alert_img = raw_frame if raw_frame is not None else frame
        send_line_notify(
            msg, image=alert_img, level=level,
            cooldown_key=f"fall_{state.person_id}",
            cooldown_sec=cfg.FALL_COOLDOWN_SECONDS,
        )

    # ── Roboflow fallback ─────────────────────────────────────
    if not _person_states and not MP_OK:
        predictions = data.get("predictions") or []
        falls = [p for p in predictions
                 if "fall" in p.get("class","").lower()]
        if falls:
            now = time.time()
            if now - _last_alert >= cfg.FALL_COOLDOWN_SECONDS:
                _last_alert = now
                stats["falls"] += len(falls)
                stats["alerts_sent"] += 1
                if frame is not None:
                    get_collector().collect(frame, predictions,
                                            "fall_events", force=True)
                alert_img = raw_frame if raw_frame is not None else frame
                send_line_notify(
                    f"[ZENTRA Emergency]\nFall detected: {len(falls)} person(s)\nAssist within 30 minutes!",
                    image=alert_img,
                    level=cfg.ALERT_LEVEL_EMERGENCY,
                    cooldown_key="fall_roboflow",
                    cooldown_sec=cfg.FALL_COOLDOWN_SECONDS,
                )

    # ── MediaPipe confirm path ────────────────────────────────
    if not _person_states and MP_OK and _fall_confirm >= cfg.FALL_CONFIRM_FRAMES:
        gs = _rule_engine(_global_state)
        should, level, msg = _should_alert(gs, _global_state)
        if should:
            stats["alerts_sent"] += 1
            _last_alert = time.time()
            if frame is not None:
                get_collector().collect(frame, [], "fall_events", force=True)
            alert_img = raw_frame if raw_frame is not None else frame
            send_line_notify(msg, image=alert_img, level=level,
                cooldown_key="fall_mp",
                cooldown_sec=cfg.FALL_COOLDOWN_SECONDS)
        _fall_confirm = 0
