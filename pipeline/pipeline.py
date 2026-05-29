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
class _InferenceWorker(threading.Thread):
    def __init__(self, client, ppe_id: str, fall_id: str):
        super().__init__(daemon=True, name="InferenceWorker")
        self.client  = client
        self.ppe_id  = ppe_id
        self.fall_id = fall_id
        self.in_q    = queue.Queue(maxsize=2)
        self.out_q   = queue.Queue(maxsize=4)
        self._stop   = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                fid, frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                ppe_preds, fall_preds = [], []
                if self.client:
                    r1 = self.client.infer(frame, model_id=self.ppe_id)
                    ppe_preds = r1.get("predictions", [])
                    r2 = self.client.infer(frame, model_id=self.fall_id)
                    fall_preds = r2.get("predictions", [])
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
    __slots__ = ("frame_id",)
    def __init__(self, fid: int):
        self.frame_id = fid


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

        # Called on every real alert: (msg: str, level: str) → None
        self.on_alert: Optional[Callable[[str, str], None]] = None

        self.status: dict = {
            "running":        False,
            "source":         None,
            "modules":        {"ppe": "error", "zone": "error", "fall": "error"},
            "alerts":         {"total": 0, "warning": 0, "emergency": 0},
            "uptime_seconds": 0,
            "last_emergency": None,
        }

    # ── Public API ────────────────────────────────────────────

    def start(self, source_config: dict) -> bool:
        """Open camera + start AI threads. Returns True on success."""
        if self._running:
            self.stop()
        self._stop_evt.clear()
        try:
            self._apply_config(source_config)
            self._import_modules()
            self._cap = self._open_camera(source_config)
        except Exception as e:
            print(f"[Pipeline] ❌ start failed: {e}")
            traceback.print_exc()
            return False

        self._start_time = time.time()
        self._running    = True

        self._proc_thr = threading.Thread(
            target=self._process_loop, daemon=True, name="PipelineLoop"
        )
        self._proc_thr.start()

        with self._lock:
            self.status["running"] = True
            self.status["source"]  = source_config.get("source", "webcam")

        print(f"[Pipeline] ✅ Started — {source_config.get('source', 'webcam')}")
        return True

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
            if "fall_bbox_ratio" in ai:
                cfg.FALL_BBOX_RATIO_THRESH = float(ai["fall_bbox_ratio"])
            if "fall_confirm_frames" in ai:
                cfg.FALL_CONFIRM_FRAMES = int(ai["fall_confirm_frames"])
            alr = settings.get("alerts", {})
            if "violation_cooldown_seconds" in alr:
                cfg.VIOLATION_COOLDOWN_SECONDS = int(alr["violation_cooldown_seconds"])
            if "zone_cooldown_seconds" in alr:
                cfg.ZONE_COOLDOWN_SECONDS = int(alr["zone_cooldown_seconds"])
            if "fall_cooldown_seconds" in alr:
                cfg.FALL_COOLDOWN_SECONDS = int(alr["fall_cooldown_seconds"])
            line = settings.get("line", {})
            if "channel_access_token" in line:
                cfg.LINE_OA_CHANNEL_ACCESS_TOKEN = line["channel_access_token"]
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
                # Fire external callback (for WebSocket broadcast)
                if pipeline.on_alert:
                    try:
                        pipeline.on_alert(msg, level)
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
            print(f"[Pipeline] Inference server: {cfg.INFERENCE_SERVER_URL}")
            return client, ppe_id, fall_id
        except Exception as e:
            print(f"[Pipeline] inference_sdk unavailable: {e} — running without YOLO")
            import config as cfg
            return None, cfg.PPE_MODEL_ID, cfg.FALL_MODEL_ID

    def _process_loop(self):
        try:
            import config as cfg
            import modules.ppe         as ppe_module
            import modules.safety_zone as zone_module
            import modules.heat_stroke as fall_module

            client, ppe_id, fall_id = self._make_client()

            reader     = _FrameReader(self._cap)
            reader.start()
            self._reader = reader

            inf_worker = _InferenceWorker(client, ppe_id, fall_id)
            inf_worker.start()
            self._inf_wkr = inf_worker

            frame_id         = 0
            last_ppe_preds   = []
            last_fall_preds  = []
            flip_webcam      = (cfg.CAMERA_SOURCE == "webcam")

            print("[Pipeline] ▶️  Process loop running")

            while not self._stop_evt.is_set() and self._running:
                ret, raw = reader.read()
                if not ret or raw is None:
                    continue

                frame_id += 1
                if flip_webcam:
                    raw = cv2.flip(raw, 1)

                if frame_id % cfg.INFER_EVERY_N_FRAMES == 0:
                    inf_worker.submit(frame_id, raw)

                res = inf_worker.get_result()
                if res:
                    _, last_ppe_preds, last_fall_preds = res

                annotated = raw.copy()
                annotated = ppe_module.draw_predictions(annotated, last_ppe_preds)
                annotated = fall_module.draw_fall_predictions(annotated, last_fall_preds)

                meta = _Meta(frame_id)

                # Pass empty window_title — modules skip cv2.imshow()
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
            print("[Pipeline] Process loop ended")
