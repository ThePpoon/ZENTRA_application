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

    def run(self):
        import config as cfg
        while not self._stop.is_set():
            try:
                fid, frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # Read PPE confidence fresh each loop so the Settings slider applies
                # live (local model also honours it now). Fall is filtered separately
                # by FALL_YOLO_CONFIDENCE in heat_stroke, so PPE conf never throttles it.
                ppe_conf = float(getattr(cfg, "INFERENCE_CONFIDENCE", 0.4))
                ppe_preds, fall_preds = [], []
                if self.local_ppe is not None:
                    # PPE from the locally fine-tuned YOLOv8 model
                    r = self.local_ppe.predict(frame, conf=ppe_conf, verbose=False)
                    if r:
                        ppe_preds = _yolo_to_roboflow(r[0])
                elif self.client:
                    # Server returns low-floor candidates; filter PPE in code by the
                    # live PPE threshold (keeps fall detections independent).
                    r1 = self.client.infer(frame, model_id=self.ppe_id)
                    ppe_preds = [p for p in r1.get("predictions", [])
                                 if p.get("confidence", 0.0) >= ppe_conf]
                # Fall always via the Roboflow server (MediaPipe pose also covers it)
                if self.client:
                    try:
                        r2 = self.client.infer(frame, model_id=self.fall_id)
                        fall_preds = r2.get("predictions", [])
                    except Exception:
                        fall_preds = []
                self.out_q.put((fid, ppe_preds, fall_preds))
            except Exception as e:
                print(f"[Inference] {e}")

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
                    _, last_ppe_preds, last_fall_preds = res
                    if not ppe_validated:
                        for p in last_ppe_preds:
                            seen_ppe_classes.add(str(p.get("class", "")).lower())
                        if frame_id >= 250:
                            ppe_validated = True
                            if seen_ppe_classes:
                                print(f"[Pipeline] PPE model (cloud) classes seen: {sorted(seen_ppe_classes)}")

                annotated = raw.copy()
                annotated = ppe_module.draw_predictions(annotated, last_ppe_preds)
                annotated = fall_module.draw_fall_predictions(annotated, last_fall_preds)

                # Run the shared person tracker once per frame and expose the
                # tracks via meta so every module uses the same IDs.
                person_dets = [p for p in last_ppe_preds
                               if str(p.get("class", "")).lower() == "person"]
                tracks = person_tracker.update(person_dets)
                # Drop anyone standing in an exclusion (ignore) zone so NO module
                # fires there — reduces false positives in static/irrelevant areas.
                tracks = self._filter_excluded(tracks, zone_module)
                meta = _Meta(frame_id, tracks)

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
