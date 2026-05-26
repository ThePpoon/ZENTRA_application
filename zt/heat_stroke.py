# modules/heat_stroke.py — ZENTRA Heat Stroke Risk Detection
# ================================================================
# Hybrid 3-Layer System:
#   Layer 1: YOLOv8 Pose (RTMPose/YOLOv8-pose) — ทนทานกับกล้องมุมสูง/แสงน้อย
#   Layer 2: MediaPipe Pose — 33 keypoints สำหรับ joint angle analysis
#   Layer 3: Rule Engine — 4 พฤติกรรม + temporal context
#
# ตรวจจับ 4 พฤติกรรม:
#   1. Sudden Fall      — velocity spike + bbox ratio change
#   2. Prone / Slumped  — นอนหรือนั่งหมดแรงบนพื้น (abnormal posture)
#   3. Gait Anomaly     — เดินโซเซ COG trajectory ผิดปกติ
#   4. Motionless       — ไม่ขยับนานผิดปกติ (heat stroke จริงๆ)
#
# ทำงานได้แม้: กล้องมุมสูง, แสงน้อย, คนหลายคน, MediaPipe ไม่ available
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
# Layer 1 — YOLOv8-pose (Primary, ทนทานกว่า MediaPipe)
# ================================================================
_yolo_pose_model = None

def _load_yolo_pose():
    global _yolo_pose_model
    if _yolo_pose_model is not None:
        return _yolo_pose_model
    try:
        from ultralytics import YOLO
        # ใช้ local fine-tuned ถ้ามี, ไม่งั้น yolov8m-pose.pt
        local = cfg.MODELS_DIR / "pose_finetuned.pt"  if hasattr(cfg,"MODELS_DIR") else None
        weights = str(local) if (local and local.exists()) else "yolov8m-pose.pt"
        _yolo_pose_model = YOLO(weights)
        print(f"[HeatStroke] ✅ YOLOv8-pose loaded: {weights}")
    except Exception as e:
        print(f"[HeatStroke] ⚠️  YOLOv8-pose unavailable: {e}")
        _yolo_pose_model = None
    return _yolo_pose_model

# ================================================================
# Layer 2 — MediaPipe Pose (Secondary)
# ================================================================
_mp_pose_model = None
MP_OK = False

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
    MP_OK = True
    print("[HeatStroke] ✅ MediaPipe Pose loaded")
except Exception as _e:
    print(f"[HeatStroke] ⚠️  MediaPipe unavailable ({_e}) → YOLOv8-pose only")

# ================================================================
# YOLO Keypoint Indices (COCO 17 keypoints)
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
    """Snapshot ของ pose ณ เวลาหนึ่ง"""
    timestamp:   float
    bbox:        tuple           # (x1,y1,x2,y2)
    keypoints:   np.ndarray     # (17,3) x,y,conf หรือ (33,3) MediaPipe
    bbox_ratio:  float          # width/height
    cog:         tuple          # (cx_norm, cy_norm)
    is_mp:       bool = False   # True=MediaPipe, False=YOLO

@dataclass
class PersonState:
    """State ต่อ person (tracked by ID)"""
    person_id:    int
    snapshots:    collections.deque = field(default_factory=lambda: collections.deque(maxlen=90))
    motionless_since: Optional[float] = None
    fall_risk_frames: int = 0
    last_alert_type:  str = ""

    # ── Computed properties ───────────────────────────────
    @property
    def kp_velocity(self) -> float:
        """Keypoint velocity ระหว่าง 2 snapshots ล่าสุด"""
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
        """COG trajectory variance — Gait Anomaly"""
        if len(self.snapshots) < 15:
            return 0.0
        pts = np.array([s.cog for s in list(self.snapshots)[-30:]])
        return float(np.var(pts[:,0]) + np.var(pts[:,1]))

    @property
    def motionless_seconds(self) -> float:
        """วินาทีที่ไม่ขยับ"""
        if self.motionless_since is None:
            return 0.0
        return time.time() - self.motionless_since

    @property
    def latest_bbox_ratio(self) -> float:
        return self.snapshots[-1].bbox_ratio if self.snapshots else 1.0

    @property
    def is_prone(self) -> bool:
        """นอนหรือเอียงมากผิดปกติ (ratio สูง)"""
        return self.latest_bbox_ratio > cfg.FALL_BBOX_RATIO_THRESH

    def update_motionless(self, threshold_px: float = 0.015):
        """อัพเดต motionless timer"""
        if len(self.snapshots) < 5:
            return
        recent = list(self.snapshots)[-5:]
        cogs   = np.array([s.cog for s in recent])
        movement = float(np.mean(np.std(cogs, axis=0)))
        if movement < threshold_px:
            if self.motionless_since is None:
                self.motionless_since = time.time()
        else:
            self.motionless_since = None


# ================================================================
# GEOMETRY HELPERS
# ================================================================
def _angle3(a, b, c) -> float:
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=np.float32)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=np.float32)
    n  = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-6: return 180.0
    return float(math.degrees(math.acos(np.clip(np.dot(ba,bc)/n, -1.0, 1.0))))

def _cog_from_yolo_kp(kp: np.ndarray, w: int, h: int) -> tuple:
    """COG จาก hip keypoints (COCO)"""
    lh, rh = kp[KP["left_hip"]], kp[KP["right_hip"]]
    if lh[2] > 0.3 and rh[2] > 0.3:
        return ((lh[0]+rh[0])/2/w, (lh[1]+rh[1])/2/h)
    # fallback: centroid ของ keypoints ที่มองเห็น
    vis = kp[kp[:,2]>0.3]
    if len(vis) == 0:
        return (0.5, 0.5)
    return (float(vis[:,0].mean()/w), float(vis[:,1].mean()/h))

def _cog_from_mp_lm(lm_list, w: int, h: int) -> tuple:
    if not MP_OK: return (0.5, 0.5)
    PL = _mp_pose.PoseLandmark
    lh = lm_list[PL.LEFT_HIP.value]
    rh = lm_list[PL.RIGHT_HIP.value]
    return ((lh.x+rh.x)/2, (lh.y+rh.y)/2)


# ================================================================
# Layer 3 — Rule Engine
# ================================================================
@dataclass
class DetectionResult:
    person_id:    int
    sudden_fall:  bool = False
    prone:        bool = False
    gait_anomaly: bool = False
    motionless:   bool = False
    risk_score:   float = 0.0   # 0.0-1.0
    details:      str = ""

    @property
    def is_emergency(self) -> bool:
        """ฉุกเฉิน = motionless + prone หรือ sudden fall ยืนยันแล้ว"""
        return self.motionless or (self.sudden_fall and self.prone)

    @property
    def is_warning(self) -> bool:
        return self.gait_anomaly or self.prone or self.sudden_fall


def _rule_engine(state: PersonState) -> DetectionResult:
    res = DetectionResult(person_id=state.person_id)

    # 1. Sudden Fall — velocity spike + ratio เปลี่ยนเร็ว
    vel = state.kp_velocity
    if vel > cfg.FALL_KEYPOINT_VELOCITY_THRESH:
        res.sudden_fall = True
        res.details += f"vel={vel:.3f} "
        res.risk_score += 0.35

    # 2. Prone / Slumped — bbox ratio
    ratio = state.latest_bbox_ratio
    if ratio > cfg.FALL_BBOX_RATIO_THRESH:
        res.prone = True
        res.details += f"ratio={ratio:.2f} "
        res.risk_score += 0.30

    # 3. Gait Anomaly — COG variance
    var = state.cog_variance
    if var > cfg.GAIT_ANOMALY_THRESH:
        res.gait_anomaly = True
        res.details += f"cog_var={var:.3f} "
        res.risk_score += 0.20

    # 4. Motionless — ไม่ขยับ > N วินาที
    motionless_sec = state.motionless_seconds
    MOTIONLESS_WARN      = getattr(cfg, "MOTIONLESS_WARN_SEC",   15.0)
    MOTIONLESS_EMERGENCY = getattr(cfg, "MOTIONLESS_EMERGENCY_SEC", 30.0)
    if motionless_sec >= MOTIONLESS_WARN:
        res.motionless  = True
        extra = min((motionless_sec - MOTIONLESS_WARN) / MOTIONLESS_EMERGENCY, 1.0)
        res.risk_score += 0.15 + extra * 0.35
        res.details    += f"still={motionless_sec:.0f}s "

    res.risk_score = min(res.risk_score, 1.0)
    return res


# ================================================================
# MODULE STATE
# ================================================================
_person_states: dict[int, PersonState] = {}
_global_state   = PersonState(person_id=0)   # fallback ถ้าไม่มี ID

stats = {
    "falls":          0,
    "prone_events":   0,
    "gait_events":    0,
    "motionless_events": 0,
    "alerts_sent":    0,
    "frames_analyzed": 0,
}
_last_alert: float = 0.0
_fall_confirm:  int = 0   # สำหรับ MediaPipe path


# ================================================================
# YOLO-POSE PROCESSING (Layer 1)
# ================================================================
def _process_yolo_pose(frame: np.ndarray) -> list[PersonState]:
    """รัน YOLOv8-pose → อัพเดต PersonState หลายคน"""
    model = _load_yolo_pose()
    if model is None:
        return []

    h, w  = frame.shape[:2]
    try:
        results = model.predict(frame, conf=0.40, iou=0.45,
                                verbose=False, stream=False)
    except Exception as e:
        print(f"[HeatStroke] YOLO predict error: {e}")
        return []

    updated = []
    if not results or results[0].keypoints is None:
        return []

    boxes = results[0].boxes
    kps   = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

    for i in range(len(boxes)):
        pid = int(boxes[i].id.item()) if boxes[i].id is not None else i
        x1,y1,x2,y2 = boxes[i].xyxy[0].cpu().numpy().astype(int)
        bw, bh = max(x2-x1, 1), max(y2-y1, 1)
        ratio  = bw / bh

        kp     = kps[i] if i < len(kps) else np.zeros((17,3))
        cog    = _cog_from_yolo_kp(kp, w, h)
        snap   = PoseSnapshot(
            timestamp  = time.time(),
            bbox       = (x1,y1,x2,y2),
            keypoints  = kp,
            bbox_ratio = ratio,
            cog        = cog,
            is_mp      = False,
        )

        if pid not in _person_states:
            _person_states[pid] = PersonState(person_id=pid)
        state = _person_states[pid]
        state.snapshots.append(snap)
        state.update_motionless()
        updated.append(state)

    # ลบ state ที่ไม่มีใน frame นี้นานเกิน 5 วินาที
    now    = time.time()
    active = {s.person_id for s in updated}
    stale  = [pid for pid, st in _person_states.items()
              if pid not in active
              and st.snapshots
              and now - st.snapshots[-1].timestamp > 5.0]
    for pid in stale:
        del _person_states[pid]

    return updated


# ================================================================
# MEDIAPIPE PROCESSING (Layer 2 — supplement)
# ================================================================
def _process_mediapipe(frame: np.ndarray):
    """รัน MediaPipe → อัพเดต global state (single person)"""
    if not MP_OK or _mp_pose_model is None:
        return None

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        res = _mp_pose_model.process(rgb)
    except Exception as e:
        print(f"[HeatStroke] MediaPipe error: {e}")
        return None

    if not res.pose_landmarks:
        return None

    lm  = res.pose_landmarks.landmark
    arr = np.array([[l.x, l.y, l.visibility] for l in lm], dtype=np.float32)

    vis    = arr[:,2] > 0.40
    xs, ys = arr[vis,0], arr[vis,1]
    if vis.sum() < 6:
        return res

    bw  = xs.max()-xs.min()
    bh  = ys.max()-ys.min()
    ratio = bw/bh if bh > 1e-3 else 1.0
    cog   = _cog_from_mp_lm(lm, w, h)

    snap = PoseSnapshot(
        timestamp  = time.time(),
        bbox       = (int(xs.min()*w), int(ys.min()*h),
                      int(xs.max()*w), int(ys.max()*h)),
        keypoints  = arr,
        bbox_ratio = ratio,
        cog        = cog,
        is_mp      = True,
    )
    _global_state.snapshots.append(snap)
    _global_state.update_motionless()
    return res


# ================================================================
# DRAW
# ================================================================
def _draw_yolo_pose(frame: np.ndarray, states: list[PersonState],
                    results_list) -> np.ndarray:
    """วาด skeleton จาก YOLO-pose"""
    if not results_list:
        return frame
    try:
        frame = results_list[0].plot(
            boxes=False, labels=False,
            conf=False, img=frame,
        )
    except Exception:
        pass
    return frame


def _draw_mp_landmarks(frame: np.ndarray, mp_result):
    if not MP_OK or mp_result is None or not mp_result.pose_landmarks:
        return
    _mp_draw.draw_landmarks(
        frame,
        mp_result.pose_landmarks,
        _mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec = _mp_styles.get_default_pose_landmarks_style(),
    )


def _draw_status(frame: np.ndarray, states: list[PersonState],
                 results: list[DetectionResult]):
    """วาดสถานะแต่ละคนบน frame"""
    h_f, w_f = frame.shape[:2]

    for res in results:
        state = _person_states.get(res.person_id, _global_state)
        if not state.snapshots:
            continue

        snap   = state.snapshots[-1]
        x1,y1,x2,y2 = snap.bbox

        # สี box ตาม risk
        if res.is_emergency:
            box_color = (0, 0, 230)
            label     = f"EMERGENCY ID:{res.person_id}"
        elif res.is_warning:
            box_color = (0, 140, 230)
            label     = f"WARNING ID:{res.person_id}"
        else:
            box_color = (0, 200, 80)
            label     = f"OK ID:{res.person_id}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2, cv2.LINE_AA)

        # Risk score bar
        bar_w  = int((x2-x1) * res.risk_score)
        bar_y  = y2 + 4
        cv2.rectangle(frame, (x1, bar_y), (x1+bar_w, bar_y+6), box_color, -1)

        # Status badges
        badge_y = y2 + 18
        badges  = []
        if res.sudden_fall:   badges.append(("FALL",      (0,0,220)))
        if res.prone:         badges.append(("PRONE",     (0,80,220)))
        if res.gait_anomaly:  badges.append(("GAIT",      (0,160,220)))
        if res.motionless:
            sec = state.motionless_seconds
            badges.append((f"STILL {sec:.0f}s", (0,40,200)))
        bx = x1
        for text, col in badges:
            (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (bx, badge_y-th-2), (bx+tw+6, badge_y+2), col, -1)
            cv2.putText(frame, text, (bx+3, badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            bx += tw + 10

    # COG trail สำหรับทุกคน
    for state in states:
        if len(state.snapshots) < 3:
            continue
        trail = np.array([
            (int(s.cog[0]*w_f), int(s.cog[1]*h_f))
            for s in list(state.snapshots)[-20:]
        ], dtype=np.int32)
        cv2.polylines(frame, [trail], False, (0, 200, 200), 2)


def draw_fall_predictions(frame: np.ndarray, predictions: list[dict]) -> np.ndarray:
    """วาด Roboflow fall model fallback"""
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return frame


# ================================================================
# ALERT LOGIC
# ================================================================
def _should_alert(res: DetectionResult, state: PersonState) -> tuple[bool, str, str]:
    """
    คืน (should_alert, level, msg)
    level: 'emergency' | 'warning'
    """
    # de-duplicate: อย่า alert event เดิมซ้ำๆ
    event_key = ""
    if res.is_emergency:
        if res.motionless:  event_key = "motionless"
        elif res.prone:     event_key = "prone_fall"
        else:               event_key = "emergency"
    elif res.sudden_fall:   event_key = "sudden_fall"
    elif res.gait_anomaly:  event_key = "gait"
    elif res.prone:         event_key = "prone"

    if not event_key:
        return False, "", ""

    # cooldown ต่อ person
    now      = time.time()
    cooldown = cfg.FALL_COOLDOWN_SECONDS
    if now - _last_alert < cooldown:
        return False, "", ""

    # สร้าง message
    motionless_sec = state.motionless_seconds
    if res.is_emergency:
        level = cfg.ALERT_LEVEL_EMERGENCY
        lines = ["🆘 ZENTRA Emergency Alert"]
        if res.motionless:
            lines.append(f"🚨 ตรวจพบคนไม่ขยับ {motionless_sec:.0f} วินาที!")
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
    lines.append(f"🆔 Person ID: {res.person_id}")
    lines.append(f"📊 Risk Score: {res.risk_score*100:.0f}%")

    return True, level, "\n".join(lines)


# ================================================================
# ON_FRAME — main entry point
# ================================================================
_last_yolo_results = []   # เก็บ results ล่าสุดสำหรับ draw

def on_frame(frame: np.ndarray, metadata, window_title: str):
    global _fall_confirm, _last_yolo_results

    stats["frames_analyzed"] += 1
    h_f, w_f = frame.shape[:2]

    # ── Layer 1: YOLOv8-pose ────────────────────────────────
    yolo_states: list[PersonState] = []
    try:
        model = _load_yolo_pose()
        if model is not None:
            results = model.predict(frame, conf=0.40, iou=0.45, verbose=False)
            _last_yolo_results = results
            # อัพเดต state แต่ละคน
            if results and results[0].keypoints is not None:
                boxes = results[0].boxes
                kps   = results[0].keypoints.data.cpu().numpy()
                for i in range(len(boxes)):
                    pid   = int(boxes[i].id.item()) if boxes[i].id is not None else i
                    x1,y1,x2,y2 = boxes[i].xyxy[0].cpu().numpy().astype(int)
                    bw,bh = max(x2-x1,1), max(y2-y1,1)
                    ratio = bw/bh
                    kp    = kps[i] if i < len(kps) else np.zeros((17,3))
                    cog   = _cog_from_yolo_kp(kp, w_f, h_f)
                    snap  = PoseSnapshot(
                        timestamp=time.time(), bbox=(x1,y1,x2,y2),
                        keypoints=kp, bbox_ratio=ratio, cog=cog, is_mp=False,
                    )
                    if pid not in _person_states:
                        _person_states[pid] = PersonState(person_id=pid)
                    _person_states[pid].snapshots.append(snap)
                    _person_states[pid].update_motionless()
                    yolo_states.append(_person_states[pid])

                # วาด skeleton จาก YOLO-pose
                try:
                    annotated = results[0].plot(boxes=True, labels=True,
                                                conf=True, img=frame)
                    frame[:] = annotated
                except Exception:
                    pass
    except Exception as e:
        print(f"[HeatStroke] YOLO frame error: {e}")

    # ── Layer 2: MediaPipe (supplement, single-person) ──────
    mp_result = None
    if MP_OK and not yolo_states:   # ใช้ MediaPipe เฉพาะถ้า YOLO ไม่เจอใคร
        mp_result = _process_mediapipe(frame)
        if mp_result and mp_result.pose_landmarks:
            _draw_mp_landmarks(frame, mp_result)
            # ใช้ global_state แทน person_states
            if not yolo_states:
                yolo_states = [_global_state]
    elif MP_OK and yolo_states:
        # รัน MediaPipe เพื่อ cross-validate เฉพาะ กรณี suspicion สูง
        high_risk = any(s.is_prone or s.motionless_since is not None for s in yolo_states)
        if high_risk:
            mp_result = _process_mediapipe(frame)
            if mp_result and mp_result.pose_landmarks:
                _draw_mp_landmarks(frame, mp_result)

    # ── Layer 3: Rule Engine ─────────────────────────────────
    det_results: list[DetectionResult] = []
    for state in yolo_states:
        dr = _rule_engine(state)
        if dr.risk_score > 0.05:
            det_results.append(dr)

    # วาด status overlay
    _draw_status(frame, yolo_states, det_results)

    # ── Confirm counter (MediaPipe fallback path) ─────────────
    if not yolo_states and MP_OK:
        gs_result = _rule_engine(_global_state)
        if gs_result.sudden_fall or gs_result.prone:
            _fall_confirm += 1
        else:
            _fall_confirm = max(0, _fall_confirm - 1)
        if _fall_confirm > 0:
            cv2.putText(frame,
                f"⚠ FALL RISK [{_fall_confirm}/{cfg.FALL_CONFIRM_FRAMES}]",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,40,210), 2, cv2.LINE_AA)


# ================================================================
# ON_DATA — alert dispatch
# ================================================================
def on_data(data: dict, metadata, frame: Optional[np.ndarray] = None):
    global _last_alert, _fall_confirm

    # ── Path A: YOLO-pose states ─────────────────────────────
    all_states = list(_person_states.values())
    if not all_states and MP_OK:
        all_states = [_global_state]

    for state in all_states:
        if not state.snapshots:
            continue
        dr = _rule_engine(state)
        if not dr.is_warning and not dr.is_emergency:
            continue

        should, level, msg = _should_alert(dr, state)
        if not should:
            continue

        # Update stats
        if dr.sudden_fall:  stats["falls"] += 1
        if dr.prone:        stats["prone_events"] += 1
        if dr.gait_anomaly: stats["gait_events"] += 1
        if dr.motionless:   stats["motionless_events"] += 1
        stats["alerts_sent"] += 1
        _last_alert = time.time()

        # Auto-collect
        if frame is not None:
            get_collector().collect(frame, [], "fall_events", force=True)

        print(f"[HeatStroke] 🆘 ALERT ID:{state.person_id} level={level}")
        print(f"             {dr.details}")
        send_line_notify(
            msg, image=frame, level=level,
            cooldown_key=f"fall_{state.person_id}",
            cooldown_sec=cfg.FALL_COOLDOWN_SECONDS,
        )

    # ── Path B: Roboflow Fall model (fallback ถ้าไม่มี YOLO-pose) ──
    if not _person_states and not MP_OK:
        predictions = data.get("predictions") or []
        falls = [p for p in predictions if "fall" in p.get("class","").lower()]
        if falls:
            now = time.time()
            if now - _last_alert >= cfg.FALL_COOLDOWN_SECONDS:
                _last_alert = now
                stats["falls"] += len(falls)
                stats["alerts_sent"] += 1
                if frame is not None:
                    get_collector().collect(frame, predictions, "fall_events", force=True)
                msg = (
                    f"🆘 ZENTRA Emergency Alert\n"
                    f"🚨 ตรวจพบการล้ม {len(falls)} ราย\n"
                    f"⏰ ต้องช่วยเหลือภายใน 30 นาที!\n"
                )
                send_line_notify(msg, image=frame,
                    level=cfg.ALERT_LEVEL_EMERGENCY,
                    cooldown_key="fall_roboflow",
                    cooldown_sec=cfg.FALL_COOLDOWN_SECONDS,
                )

    # ── MediaPipe confirm path ───────────────────────────────
    if not _person_states and MP_OK and _fall_confirm >= cfg.FALL_CONFIRM_FRAMES:
        gs = _rule_engine(_global_state)
        should, level, msg = _should_alert(gs, _global_state)
        if should:
            stats["alerts_sent"] += 1
            _last_alert = time.time()
            if frame is not None:
                get_collector().collect(frame, [], "fall_events", force=True)
            send_line_notify(msg, image=frame, level=level,
                cooldown_key="fall_mp",
                cooldown_sec=cfg.FALL_COOLDOWN_SECONDS,
            )
        _fall_confirm = 0
