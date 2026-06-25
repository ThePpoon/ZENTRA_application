"""
pipeline/pipeline.py — ZENTRA AI Pipeline Wrapper
Wraps the existing ZENTRA AI backend into a controllable Pipeline
class for use by the FastAPI server (no cv2.imshow, no keyboard).
"""
from __future__ import annotations

import sys
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

# ── Add ZENTRA backend root to sys.path ──────────────────────
_ZENTRA_ROOT = Path(__file__).parent.parent.parent / "ZENTRA"
if str(_ZENTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZENTRA_ROOT))

# ── App data dir (zones.json lives here) ─────────────────────
_APP_DATA = Path(__file__).parent.parent / "data"
_APP_DATA.mkdir(exist_ok=True)


# ================================================================
# LIGHTWEIGHT FRAME READER (mirrors main.py's FrameReader)
# ================================================================
class _FrameReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name="FrameReader")
        self.cap  = cap
        self.q    = queue.Queue(maxsize=4)
        self._stop = threading.Event()

    def run(self):
        errors = 0
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                errors += 1
                if errors > 30:
                    break
                time.sleep(0.05)
                continue
            errors = 0
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return True, self.q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop.set()


# ================================================================
# LIGHTWEIGHT INFERENCE WORKER (mirrors main.py's InferenceWorker)
# ================================================================
def _yolo_to_roboflow(result, conf_min: float = 0.0) -> list[dict]:
    """Convert an ultralytics Results object → Roboflow-style prediction dicts
    (class/confidence/x/y/width/height) so the existing modules work unchanged.
    Class names are lower-cased to match config.PPE_CLASSES keys."""
    preds = []
    names = getattr(result, "names", {})
    for b in result.boxes:
        c = float(b.conf[0])
        if c < conf_min:
            continue
        cid = int(b.cls[0])
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        preds.append({
            "class": str(names.get(cid, cid)).lower(),
            "confidence": c,
            "x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0,
            "width": x2 - x1, "height": y2 - y1,
        })
    return preds


class _BoxSmoother:
    """Temporal EMA on detection boxes (DISPLAY only) so they glide instead of
    jittering frame-to-frame. Matches boxes across frames by class + IoU; also
    holds a box for `ttl` seconds to bridge per-class flicker (e.g. gloves)."""

    def __init__(self):
        self._items: list[dict] = []   # class,confidence,x,y,width,height,ts

    @staticmethod
    def _xyxy(b: dict):
        return (b["x"] - b["width"] / 2, b["y"] - b["height"] / 2,
                b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
        return inter / union

    def update(self, preds, now: float, alpha: float, iou_thr: float, ttl: float,
               deadband: float = 0.0):
        used = set()
        for p in preds:
            pb, cls = self._xyxy(p), p.get("class", "")
            best, bj = iou_thr, None
            for j, it in enumerate(self._items):
                if j in used or it["class"] != cls:
                    continue
                i = self._iou(pb, self._xyxy(it))
                if i >= best:
                    best, bj = i, j
            if bj is None:                       # new object
                self._items.append({
                    "class": cls, "confidence": p.get("confidence", 0.0),
                    "x": p.get("x", 0), "y": p.get("y", 0),
                    "width": p.get("width", 0), "height": p.get("height", 0), "ts": now})
                used.add(len(self._items) - 1)
            else:                                # matched existing object
                it = self._items[bj]
                # Deadband: only move if the box changed more than `deadband` px
                # (frozen → rock-steady while a person stands still).
                moved = any(abs(p.get(k, 0) - it[k]) > deadband
                            for k in ("x", "y", "width", "height"))
                if moved:
                    for k in ("x", "y", "width", "height"):
                        it[k] = alpha * p.get(k, 0) + (1 - alpha) * it[k]
                it["confidence"] = p.get("confidence", it["confidence"])
                it["ts"] = now
                used.add(bj)
        self._items = [it for it in self._items if now - it["ts"] <= ttl]
        return [{"class": it["class"], "confidence": it["confidence"],
                 "x": it["x"], "y": it["y"], "width": it["width"], "height": it["height"]}
                for it in self._items]


def _box_xyxy(p: dict):
    return (p["x"] - p["width"] / 2, p["y"] - p["height"] / 2,
            p["x"] + p["width"] / 2, p["y"] + p["height"] / 2)


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / union


def _nms_dedup(preds: list, iou_thr: float) -> list:
    """Class-aware NMS: keep the highest-confidence box, drop same-class boxes
    that overlap it above iou_thr (removes the duplicate 'ซ้อน' boxes)."""
    out: list = []
    for p in sorted(preds, key=lambda x: x.get("confidence", 0.0), reverse=True):
        pb, cls = _box_xyxy(p), p.get("class")
        if any(q.get("class") == cls and _iou_xyxy(pb, _box_xyxy(q)) > iou_thr for q in out):
            continue
        out.append(p)
    return out


def _ppe_conf_ok(pred: dict, cfg) -> bool:
    """Per-class confidence gate: small/hard classes (e.g. 'no gloves') use a
    lower threshold from PPE_CLASS_CONF; others fall back to INFERENCE_CONFIDENCE."""
    cls = str(pred.get("class", "")).lower()
    thr = getattr(cfg, "PPE_CLASS_CONF", {}).get(
        cls, float(getattr(cfg, "INFERENCE_CONFIDENCE", 0.4)))
    return float(pred.get("confidence", 0.0)) >= thr


class _InferenceWorker(threading.Thread):
    def __init__(self, client, ppe_id: str, fall_id: str,
                 local_ppe=None, conf: float = 0.4):
        super().__init__(daemon=True, name="InferenceWorker")
        self.client    = client
        self.ppe_id    = ppe_id
        self.fall_id   = fall_id
        self.local_ppe = local_ppe        # ultralytics YOLO model, or None
        self.conf      = conf
        self.in_q      = queue.Queue(maxsize=2)
        self.out_q     = queue.Queue(maxsize=4)
        self._stop     = threading.Event()
        # Cloud fall inference (~300 ms) runs on its OWN thread so the fast PPE
        # result (~10 ms on GPU) is NEVER delayed waiting for it (fixes the
        # laggy PPE boxes). They share only the latest frame + latest fall preds.
        self._fall_lock  = threading.Lock()
        self._fall_frame: Optional[np.ndarray] = None
        self._fall_preds: list = []
        self._fall_thr = None

    def run(self):
        import config as cfg
        # Spin up the independent fall-inference thread (only if a cloud client)
        if self.client:
            self._fall_thr = threading.Thread(target=self._fall_loop, daemon=True,
                                              name="FallInfer")
            self._fall_thr.start()

        while not self._stop.is_set():
            try:
                fid, frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # Read PPE confidence fresh each loop so the Settings slider applies
                # live (local model also honours it now).
                # Predict at a low floor and filter PER CLASS (so small classes
                # like 'no gloves' can use a lower threshold and stop flickering).
                floor = float(getattr(cfg, "INFERENCE_SERVER_FLOOR", 0.20))
                imgsz = int(getattr(cfg, "PPE_IMGSZ", 640))
                ppe_preds = []
                if self.local_ppe is not None:
                    r = self.local_ppe.predict(frame, conf=floor, imgsz=imgsz, verbose=False)
                    if r:
                        ppe_preds = [p for p in _yolo_to_roboflow(r[0]) if _ppe_conf_ok(p, cfg)]
                elif self.client:
                    r1 = self.client.infer(frame, model_id=self.ppe_id)
                    ppe_preds = [p for p in r1.get("predictions", []) if _ppe_conf_ok(p, cfg)]
                # Hand the latest frame to the fall thread; read its latest result
                # without blocking on the slow round-trip.
                with self._fall_lock:
                    self._fall_frame = frame
                    fall_preds = list(self._fall_preds)
                self.out_q.put((fid, ppe_preds, fall_preds))
            except Exception as e:
                print(f"[Inference] {e}")

    def _fall_loop(self):
        """Run cloud fall inference continuously on the MOST RECENT frame only,
        decoupled from PPE so it can be slow without making the view lag."""
        while not self._stop.is_set():
            with self._fall_lock:
                frame = self._fall_frame
                self._fall_frame = None
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                r = self.client.infer(frame, model_id=self.fall_id)
                preds = r.get("predictions", [])
            except Exception:
                preds = []
            with self._fall_lock:
                self._fall_preds = preds

    def submit(self, fid: int, frame: np.ndarray):
        if not self.in_q.full():
            self.in_q.put((fid, frame.copy()))

    def get_result(self):
        try:
            return self.out_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()


class _Meta:
    __slots__ = ("frame_id", "tracks")
    def __init__(self, fid: int, tracks=None):
        self.frame_id = fid
        # None  = caller did not run tracking (module falls back to its own)
        # list  = shared single-pass tracks (possibly empty = no persons)
        self.tracks = tracks


# ================================================================
# PIPELINE
# ================================================================
class Pipeline:
    """Controllable ZENTRA AI pipeline for desktop application."""

    def __init__(self):
        self._lock       = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop_evt   = threading.Event()
        self._running    = False

        self._latest_frame: Optional[np.ndarray] = None
        self._cap        = None
        self._reader     = None
        self._inf_wkr    = None
        self._proc_thr   = None
        self._start_time: Optional[float] = None
        self._modules_ok = False
        self._flip_override: Optional[bool] = None   # None = auto (mirror webcam)
        self._inf_client = None                       # InferenceHTTPClient ref

        # Called on every real alert: (msg: str, level: str, line_sent: bool) → None
        # line_sent reflects whether the LINE push was actually dispatched (a
        # recipient exists and it wasn't suppressed by cooldown).
        self.on_alert: Optional[Callable[[str, str, bool], None]] = None
        # Called whenever pipeline status changes: (status: dict) → None
        self.on_status: Optional[Callable[[dict], None]] = None

        self._source_config: dict = {}

        self.status: dict = {
            "running":        False,
            "source":         None,
            "camera":         "disconnected",   # connected | reconnecting | disconnected
            "modules":        {"ppe": "error", "zone": "error", "fall": "error"},
            "alerts":         {"total": 0, "warning": 0, "emergency": 0},
            "uptime_seconds": 0,
            "last_emergency": None,
            "ppe_model":      "cloud",   # cloud (Roboflow) | local (fine-tuned)
        }

    # ── Public API ────────────────────────────────────────────

    def start(self, source_config: dict) -> bool:
        """Open camera + start AI threads. Returns True on success."""
        if self._running:
            self.stop()
        self._stop_evt.clear()
        self._source_config = dict(source_config)
        try:
            self._apply_config(source_config)
            self._import_modules()
            self._cap = self._open_camera(source_config)
        except Exception as e:
            print(f"[Pipeline] ❌ start failed: {e}")
            traceback.print_exc()
            self._set_camera_state("disconnected")
            return False

        self._start_time = time.time()
        self._running    = True

        with self._lock:
            self.status["running"] = True
            self.status["source"]  = source_config.get("source", "webcam")
        self._set_camera_state("connected")

        self._proc_thr = threading.Thread(
            target=self._process_loop, daemon=True, name="PipelineLoop"
        )
        self._proc_thr.start()

        print(f"[Pipeline] ✅ Started — {source_config.get('source', 'webcam')}")
        return True

    def _set_camera_state(self, state: str):
        """Update camera connection state and notify listeners (if changed)."""
        changed = False
        with self._lock:
            if self.status.get("camera") != state:
                self.status["camera"] = state
                changed = True
            snapshot = dict(self.status)
        if changed and self.on_status:
            try:
                self.on_status(snapshot)
            except Exception as e:
                print(f"[Pipeline] on_status callback: {e}")

    def stop(self):
        """Stop all pipeline threads gracefully."""
        if not self._running:
            return
        self._running = False
        self._stop_evt.set()

        for obj in (self._reader, self._inf_wkr):
            try:
                if obj:
                    obj.stop()
            except Exception:
                pass

        if self._proc_thr and self._proc_thr.is_alive():
            self._proc_thr.join(timeout=3.0)

        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        with self._lock:
            self.status["running"] = False
        self._set_camera_state("disconnected")
        print("[Pipeline] ⏹️  Stopped")

    def is_running(self) -> bool:
        return self._running

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_snapshot(self) -> Optional[bytes]:
        frame = self.get_latest_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def get_uptime(self) -> int:
        if self._start_time is None:
            return 0
        return int(time.time() - self._start_time)

    def reload_zones(self):
        """Reload zones.json into safety_zone module (call after zone CRUD)."""
        if not self._modules_ok:
            return
        try:
            import modules.safety_zone as zm
            zm._load_zones()
            print(f"[Pipeline] Zones reloaded: {len(zm.zones)}")
        except Exception as e:
            print(f"[Pipeline] reload_zones: {e}")

    def apply_settings(self, settings: dict):
        """Apply settings at runtime (no restart needed)."""
        try:
            import config as cfg
            ai = settings.get("ai", {})
            if "ppe_confidence" in ai:
                cfg.INFERENCE_CONFIDENCE = float(ai["ppe_confidence"])
                self._apply_infer_config()   # push new threshold to the client live
            if "fall_bbox_ratio" in ai:
                cfg.FALL_BBOX_RATIO_THRESH = float(ai["fall_bbox_ratio"])
            if "fall_confirm_frames" in ai:
                cfg.FALL_CONFIRM_FRAMES = int(ai["fall_confirm_frames"])
            if "use_local_model" in ai:
                cfg.USE_LOCAL_MODEL = bool(ai["use_local_model"])  # applies on next (re)connect
            if "fall_mode" in ai:
                cfg.FALL_MODE = str(ai["fall_mode"]).lower()       # hybrid | yolo | pose (live)
            if "fall_yolo_confidence" in ai:
                cfg.FALL_YOLO_CONFIDENCE = float(ai["fall_yolo_confidence"])
            if "fall_yolo_confirm_frames" in ai:
                cfg.FALL_YOLO_CONFIRM_FRAMES = int(ai["fall_yolo_confirm_frames"])
            alr = settings.get("alerts", {})
            if "violation_cooldown_seconds" in alr:
                cfg.VIOLATION_COOLDOWN_SECONDS = int(alr["violation_cooldown_seconds"])
            if "zone_cooldown_seconds" in alr:
                cfg.ZONE_COOLDOWN_SECONDS = int(alr["zone_cooldown_seconds"])
            if "fall_cooldown_seconds" in alr:
                cfg.FALL_COOLDOWN_SECONDS = int(alr["fall_cooldown_seconds"])
            if "upload_images" in alr:
                cfg.LINE_UPLOAD_IMAGES = bool(alr["upload_images"])  # PDPA toggle
            line = settings.get("line", {})
            if "channel_access_token" in line:
                cfg.LINE_OA_CHANNEL_ACCESS_TOKEN = line["channel_access_token"]
            # Group IDs: ALERT_RECIPIENTS is built once at config import time,
            # so updating the groups requires rebuilding the recipients map too.
            sup = line.get("group_supervisor", cfg.LINE_OA_GROUP_SUPERVISOR)
            saf = line.get("group_safety",     cfg.LINE_OA_GROUP_SAFETY)
            emg = line.get("group_emergency",  cfg.LINE_OA_GROUP_EMERGENCY)
            if any(k in line for k in ("group_supervisor", "group_safety", "group_emergency")):
                cfg.LINE_OA_GROUP_SUPERVISOR = sup
                cfg.LINE_OA_GROUP_SAFETY     = saf
                cfg.LINE_OA_GROUP_EMERGENCY  = emg
                cfg.ALERT_RECIPIENTS = {
                    cfg.ALERT_LEVEL_WARNING:   [sup],
                    cfg.ALERT_LEVEL_ALERT:     [saf, sup],
                    cfg.ALERT_LEVEL_EMERGENCY: [emg, saf, sup],
                }
            cam = settings.get("camera", {})
            if "flip_horizontal" in cam:
                self._flip_override = bool(cam["flip_horizontal"])
            print("[Pipeline] ⚙️  Settings applied")
        except Exception as e:
            print(f"[Pipeline] apply_settings: {e}")

    # ── Private helpers ───────────────────────────────────────

    def _apply_config(self, src_cfg: dict):
        import config as cfg
        cfg.CAMERA_SOURCE   = src_cfg.get("source", "webcam")
        cfg.WEBCAM_INDEX    = int(src_cfg.get("webcam_index", 0))
        cfg.RTSP_URL        = src_cfg.get("rtsp_url", cfg.RTSP_URL)
        cfg.VIDEO_FILE_PATH = src_cfg.get("video_file_path", "")
        cfg.USE_DSHOW       = True
        # Point zones at app's data dir
        cfg.ZONE_POLYGON_FILE = str(_APP_DATA / "zones.json")
        print(f"[Pipeline] Config: source={cfg.CAMERA_SOURCE}")

    def _import_modules(self):
        try:
            import modules.ppe         as ppe
            import modules.safety_zone as zm
            import modules.heat_stroke as fall
            from alerts.line_notify import start_sender

            zm._load_zones()
            start_sender()
            self._modules_ok = True
            self._setup_monkey_patch()

            with self._lock:
                self.status["modules"] = {"ppe": "ok", "zone": "ok", "fall": "ok"}
            print("[Pipeline] AI modules loaded ✅")
        except Exception as e:
            print(f"[Pipeline] module import error: {e}")
            raise

    def _setup_monkey_patch(self):
        """Patch send_line_notify in all modules to fire self.on_alert callback."""
        try:
            import alerts.line_notify  as ln_mod
            import modules.ppe         as ppe_mod
            import modules.safety_zone as zone_mod
            import modules.heat_stroke as fall_mod

            original = ln_mod.send_line_notify
            pipeline = self

            def _patched(msg, image=None, level="warning", **kwargs):
                result = original(msg, image=image, level=level, **kwargs)
                # Update internal counters
                with pipeline._lock:
                    pipeline.status["alerts"]["total"] += 1
                    if level == "emergency":
                        pipeline.status["alerts"]["emergency"] += 1
                    else:
                        pipeline.status["alerts"]["warning"] += 1
                    if level == "emergency":
                        pipeline.status["last_emergency"] = msg
                # Fire external callback (for WebSocket broadcast + local history).
                # Pass the real LINE dispatch result so history records line_sent
                # accurately (the event is always logged locally regardless — PDPA).
                if pipeline.on_alert:
                    try:
                        pipeline.on_alert(msg, level, bool(result))
                    except Exception as cb_e:
                        print(f"[Pipeline] on_alert callback: {cb_e}")
                return result

            # Patch all local references
            ln_mod.send_line_notify  = _patched
            ppe_mod.send_line_notify  = _patched
            zone_mod.send_line_notify = _patched
            fall_mod.send_line_notify = _patched
            print("[Pipeline] Monkey-patch applied ✅")
        except Exception as e:
            print(f"[Pipeline] monkey-patch error: {e}")

    def _open_camera(self, src_cfg: dict) -> cv2.VideoCapture:
        src = src_cfg.get("source", "webcam")
        if src == "webcam":
            cap = cv2.VideoCapture(int(src_cfg.get("webcam_index", 0)), cv2.CAP_DSHOW)
        elif src == "rtsp":
            cap = cv2.VideoCapture(src_cfg.get("rtsp_url", ""), cv2.CAP_FFMPEG)
        elif src == "file":
            cap = cv2.VideoCapture(src_cfg.get("video_file_path", ""))
        else:
            raise ValueError(f"Unknown source: {src}")

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera (source={src})")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _apply_infer_config(self):
        """Push confidence / IoU NMS thresholds to the inference client so the
        Settings PPE-confidence slider actually affects detections."""
        if not self._inf_client:
            return
        try:
            from inference_sdk import InferenceConfiguration
            import config as cfg
            # Use a low server-side floor so BOTH models return candidates; the real
            # PPE threshold is applied in code (_InferenceWorker) and the fall
            # threshold in heat_stroke. This stops the PPE slider from silently
            # raising the fall model's threshold (they share one client).
            self._inf_client.configure(InferenceConfiguration(
                confidence_threshold=float(getattr(cfg, "INFERENCE_SERVER_FLOOR", 0.20)),
                iou_threshold=float(cfg.INFERENCE_IOU),
                class_agnostic_nms=True,
            ))
        except Exception as e:
            print(f"[Pipeline] infer config skipped: {e}")

    def _load_local_ppe(self):
        """Load the locally fine-tuned PPE model when the user opted for it.
        Returns an ultralytics YOLO model, or None to use the Roboflow server."""
        import config as cfg
        if not getattr(cfg, "USE_LOCAL_MODEL", False):
            return None
        path = Path(cfg.PPE_LOCAL_MODEL)
        if not path.exists():
            print(f"[Pipeline] USE_LOCAL_MODEL on but {path} missing → using Roboflow model")
            return None
        try:
            from ultralytics import YOLO
            model = YOLO(str(path))
            print(f"[Pipeline] 🧠 PPE = local fine-tuned model: {path.name}")
            # Warn loudly if this model lacks the classes PPE/Zone depend on,
            # so swapping in a bad model fails visibly instead of silently.
            self._validate_ppe_classes(getattr(model, "names", {}).values()
                                       if hasattr(model, "names") else [],
                                       "local fine-tuned")
            return model
        except Exception as e:
            print(f"[Pipeline] local model load failed ({e}) → using Roboflow model")
            return None

    @staticmethod
    def _filter_excluded(tracks, zone_module):
        """Remove tracks whose foot point lies inside any exclusion zone."""
        try:
            polys = zone_module.get_exclusion_polygons()
        except Exception:
            polys = []
        if not polys:
            return tracks
        kept = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            fx, fy = (x1 + x2) / 2.0, float(y2)   # foot point (bottom-centre)
            inside = any(
                cv2.pointPolygonTest(p, (float(fx), float(fy)), False) >= 0
                for p in polys
            )
            if not inside:
                kept.append(t)
        return kept

    @staticmethod
    def _validate_ppe_classes(names, source: str):
        """Log whether the active PPE model exposes the classes downstream modules
        rely on: 'person' (Safety Zone tracking) and at least one violation class
        (PPE alerts). Prevents silent breakage when a model is swapped."""
        import config as cfg
        norm = {str(n).lower() for n in names}
        if not norm:
            return
        has_person    = any("person" in n for n in norm)
        has_violation = any(cfg.PPE_CLASSES.get(n, {}).get("violation") for n in norm)
        print(f"[Pipeline] PPE model ({source}) classes: {sorted(norm)}")
        if not has_person:
            print("[Pipeline] ⚠️  PPE model has NO 'person' class → Safety Zone will not detect intruders")
        if not has_violation:
            print("[Pipeline] ⚠️  PPE model has NO violation class (no_helmet/no_vest/...) → PPE alerts will not fire")
        if has_person and has_violation:
            print("[Pipeline] ✅ PPE class check OK (person + violation classes present)")

    def _make_client(self):
        try:
            from inference_sdk import InferenceHTTPClient
            import config as cfg
            ppe_id  = (cfg.PPE_LOCAL_MODEL if cfg.USE_LOCAL_MODEL
                       and Path(cfg.PPE_LOCAL_MODEL).exists() else cfg.PPE_MODEL_ID)
            fall_id = (cfg.FALL_LOCAL_MODEL if cfg.USE_LOCAL_MODEL
                       and Path(cfg.FALL_LOCAL_MODEL).exists() else cfg.FALL_MODEL_ID)
            client  = InferenceHTTPClient(
                api_url=cfg.INFERENCE_SERVER_URL, api_key=cfg.ROBOFLOW_API_KEY
            )
            self._inf_client = client
            self._apply_infer_config()
            print(f"[Pipeline] Inference server: {cfg.INFERENCE_SERVER_URL} "
                  f"(conf={cfg.INFERENCE_CONFIDENCE}, iou={cfg.INFERENCE_IOU})")
            return client, ppe_id, fall_id
        except Exception as e:
            print(f"[Pipeline] inference_sdk unavailable: {e} — running without YOLO")
            import config as cfg
            return None, cfg.PPE_MODEL_ID, cfg.FALL_MODEL_ID

    def _reconnect_camera(self) -> bool:
        """Try to reopen the camera with backoff. Returns True on success."""
        self._set_camera_state("reconnecting")
        # Stop the old reader and release the dead capture
        if self._reader:
            try:
                self._reader.stop()
            except Exception:
                pass
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        delays  = [1.0, 2.0, 3.0, 5.0]
        attempt = 0
        while not self._stop_evt.is_set() and self._running:
            try:
                self._cap = self._open_camera(self._source_config)
                reader    = _FrameReader(self._cap)
                reader.start()
                self._reader = reader
                self._set_camera_state("connected")
                print("[Pipeline] 🔌 Camera reconnected")
                return True
            except Exception as e:
                wait = delays[min(attempt, len(delays) - 1)]
                attempt += 1
                print(f"[Pipeline] reconnect attempt {attempt} failed ({e}); retry in {wait}s")
                # Sleep in small slices so stop() stays responsive
                slept = 0.0
                while slept < wait and not self._stop_evt.is_set() and self._running:
                    time.sleep(0.2)
                    slept += 0.2
        return False

    def _process_loop(self):
        try:
            import config as cfg
            import modules.ppe         as ppe_module
            import modules.safety_zone as zone_module
            import modules.heat_stroke as fall_module

            client, ppe_id, fall_id = self._make_client()

            # PPE source: locally fine-tuned model (toggle) vs Roboflow server
            local_ppe = self._load_local_ppe()
            with self._lock:
                self.status["ppe_model"] = "local" if local_ppe is not None else "cloud"

            reader     = _FrameReader(self._cap)
            reader.start()
            self._reader = reader

            inf_worker = _InferenceWorker(
                client, ppe_id, fall_id,
                local_ppe=local_ppe,
                conf=float(getattr(cfg, "INFERENCE_CONFIDENCE", 0.4)),
            )
            inf_worker.start()
            self._inf_wkr = inf_worker

            frame_id         = 0
            last_ppe_preds   = []
            last_fall_preds  = []
            last_ppe_ts      = 0.0     # when last_ppe_preds was last refreshed (anti-flicker hold)
            ppe_hold         = float(getattr(cfg, "PPE_HOLD_SEC", 0.5))
            read_failures    = 0
            is_file          = (cfg.CAMERA_SOURCE == "file")
            is_webcam        = (cfg.CAMERA_SOURCE == "webcam")
            # Cloud model class names aren't known up-front; log the classes it
            # actually emits once, for debugging model swaps (local is validated
            # at load time instead).
            seen_ppe_classes: set[str] = set()
            ppe_validated    = (local_ppe is not None)

            # Single-pass person tracker shared by ALL modules so PPE / Zone /
            # Heat reference the SAME persistent track IDs (consistency + perf).
            from utils.tracker import ByteTracker
            person_tracker = ByteTracker(
                track_thresh=getattr(cfg, "BYTETRACK_TRACK_THRESH", 0.5),
                track_buffer=getattr(cfg, "BYTETRACK_TRACK_BUFFER", 30),
                match_thresh=getattr(cfg, "BYTETRACK_MATCH_THRESH", 0.8),
            )
            ppe_smoother = _BoxSmoother()   # display-only EMA (anti-jitter)

            print("[Pipeline] ▶️  Process loop running")

            while not self._stop_evt.is_set() and self._running:
                reader   = self._reader
                ret, raw = reader.read() if reader else (False, None)
                if not ret or raw is None:
                    read_failures += 1
                    # A live camera that stops yielding frames → reconnect.
                    # (A finished video file is expected to stop; just idle.)
                    if read_failures > 40 and not is_file:
                        print("[Pipeline] ⚠️  Camera signal lost — reconnecting")
                        if not self._reconnect_camera():
                            break
                        read_failures = 0
                    continue
                read_failures = 0

                frame_id += 1
                # Flip: explicit override wins; otherwise mirror webcam only
                flip = self._flip_override if self._flip_override is not None else is_webcam
                if flip:
                    raw = cv2.flip(raw, 1)

                if frame_id % cfg.INFER_EVERY_N_FRAMES == 0:
                    inf_worker.submit(frame_id, raw)

                res = inf_worker.get_result()
                if res:
                    _, new_ppe, last_fall_preds = res
                    # Merge duplicate overlapping boxes (one object → one box) so
                    # tracking doesn't double-count and the view isn't cluttered.
                    new_ppe = _nms_dedup(new_ppe, getattr(cfg, "PPE_NMS_IOU", 0.70))
                    # Anti-flicker: refresh on a real detection; on a momentary
                    # empty result keep the previous boxes until the hold expires
                    # (stops boxes blinking while a person stands still).
                    nowt = time.time()
                    if new_ppe:
                        last_ppe_preds, last_ppe_ts = new_ppe, nowt
                    elif (nowt - last_ppe_ts) > ppe_hold:
                        last_ppe_preds = []
                    if not ppe_validated:
                        for p in last_ppe_preds:
                            seen_ppe_classes.add(str(p.get("class", "")).lower())
                        if frame_id >= 250:
                            ppe_validated = True
                            if seen_ppe_classes:
                                print(f"[Pipeline] PPE model (cloud) classes seen: {sorted(seen_ppe_classes)}")

                # Shared person tracker FIRST (needed for the clean per-person
                # display + shared meta). Drop anyone in an exclusion zone so NO
                # module fires there.
                person_dets = [p for p in last_ppe_preds
                               if str(p.get("class", "")).lower() == "person"]
                tracks = person_tracker.update(person_dets)
                tracks = self._filter_excluded(tracks, zone_module)
                meta = _Meta(frame_id, tracks)

                annotated = raw.copy()
                # Smooth boxes for DISPLAY only (logic/tracking still use raw preds)
                disp_ppe = last_ppe_preds
                if getattr(cfg, "PPE_SMOOTH", True):
                    disp_ppe = ppe_smoother.update(
                        last_ppe_preds, time.time(),
                        getattr(cfg, "PPE_SMOOTH_ALPHA", 0.25),
                        getattr(cfg, "PPE_SMOOTH_IOU", 0.30),
                        getattr(cfg, "PPE_HOLD_SEC", 0.5),
                        getattr(cfg, "PPE_SMOOTH_DEADBAND", 3.0))
                    # Final guarantee: no overlapping duplicate boxes get drawn
                    disp_ppe = _nms_dedup(disp_ppe, getattr(cfg, "PPE_NMS_IOU", 0.70))
                if getattr(cfg, "PPE_CLEAN_DISPLAY", True):
                    # One clean box per person + PPE status label
                    annotated = ppe_module.draw_person_status(annotated, disp_ppe, tracks)
                else:
                    annotated = ppe_module.draw_predictions(annotated, disp_ppe)
                annotated = fall_module.draw_fall_predictions(annotated, last_fall_preds)

                # Pass empty window_title — modules skip cv2.imshow().
                # MediaPipe Pose (fall_module.on_frame) is heavy, so run it only in
                # modes that use pose and only every Nth frame — otherwise it caps
                # the live FPS and freezes the Live view (H1). 'yolo' mode skips it.
                fall_mode  = getattr(cfg, "FALL_MODE", "hybrid").lower()
                pose_every = max(1, getattr(cfg, "FALL_POSE_EVERY_N", 3))
                if (getattr(fall_module, "MP_OK", False)
                        and fall_mode in ("hybrid", "pose")
                        and frame_id % pose_every == 0):
                    fall_module.on_frame(annotated, meta, "")
                zone_module.on_frame(annotated, meta, "")
                ppe_module.on_frame(annotated, meta, "")

                try:
                    ppe_module.on_data({"predictions": last_ppe_preds},  meta, frame=raw)
                    zone_module.on_data({"predictions": last_ppe_preds},  meta, frame=raw)
                    fall_module.on_data({"predictions": last_fall_preds}, meta, frame=raw)
                except Exception as e:
                    print(f"[Pipeline] on_data error: {e}")

                # Store annotated frame
                with self._frame_lock:
                    self._latest_frame = annotated

                # Update uptime every 150 frames
                if frame_id % 150 == 0:
                    with self._lock:
                        self.status["uptime_seconds"] = self.get_uptime()

        except Exception:
            print("[Pipeline] ❌ Process loop crashed:")
            traceback.print_exc()
        finally:
            self._running = False
            with self._lock:
                self.status["running"] = False
            self._set_camera_state("disconnected")
            print("[Pipeline] Process loop ended")
