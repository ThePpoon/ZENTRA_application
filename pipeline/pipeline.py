"""
pipeline/pipeline.py — ZENTRA Camera Pipeline (passthrough, no detection)
Reads frames from a camera source, annotates nothing, and exposes them
for WebSocket broadcast.  Detection modules will be added one by one.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

_APP_DATA = Path(__file__).parent.parent / "data"
_APP_DATA.mkdir(exist_ok=True)


# ================================================================
# FRAME READER
# ================================================================
class _FrameReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name="FrameReader")
        self.cap   = cap
        self.q     = queue.Queue(maxsize=4)
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
# PIPELINE
# ================================================================
class Pipeline:
    """Camera capture pipeline — passthrough (no AI detection yet)."""

    def __init__(self):
        self._lock        = threading.Lock()
        self._frame_lock  = threading.Lock()
        self._stop_evt    = threading.Event()
        self._running     = False

        self._latest_frame: Optional[np.ndarray] = None
        self._cap         = None
        self._reader      = None
        self._proc_thr    = None
        self._start_time: Optional[float] = None
        self._flip_override: Optional[bool] = None

        self.on_alert:  Optional[Callable[[str, str, bool], None]] = None
        self.on_status: Optional[Callable[[dict], None]] = None

        self._source_config: dict = {}

        self.status: dict = {
            "running":        False,
            "source":         None,
            "camera":         "disconnected",
            "modules":        {"ppe": "standby", "zone": "standby", "fall": "standby"},
            "alerts":         {"total": 0, "warning": 0, "emergency": 0},
            "uptime_seconds": 0,
            "last_emergency": None,
        }

    # ── Public API ────────────────────────────────────────────

    def start(self, source_config: dict) -> bool:
        if self._running:
            self.stop()
        self._stop_evt.clear()
        self._source_config = dict(source_config)
        try:
            self._apply_config(source_config)
            self._cap = self._open_camera(source_config)
        except Exception as e:
            print(f"[Pipeline] ❌ start failed: {e}")
            self._set_camera_state("disconnected")
            return False

        self._start_time = time.time()
        self._running    = True
        with self._lock:
            self.status["running"] = True
            self.status["source"]  = source_config.get("source", "webcam")
            self.status["modules"] = {"ppe": "standby", "zone": "standby", "fall": "standby"}
        self._set_camera_state("connected")

        self._proc_thr = threading.Thread(
            target=self._process_loop, daemon=True, name="PipelineLoop"
        )
        self._proc_thr.start()
        print(f"[Pipeline] ✅ Started (passthrough) — {source_config.get('source', 'webcam')}")
        return True

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_evt.set()
        if self._reader:
            try:
                self._reader.stop()
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
        pass   # no-op until Zone module is added

    def apply_settings(self, settings: dict):
        try:
            cam = settings.get("camera", {})
            if "flip_horizontal" in cam:
                self._flip_override = bool(cam["flip_horizontal"])
            line = settings.get("line", {})
            try:
                import config as cfg
                if "channel_access_token" in line:
                    cfg.LINE_OA_CHANNEL_ACCESS_TOKEN = line["channel_access_token"]
                sup = line.get("group_supervisor", getattr(cfg, "LINE_OA_GROUP_SUPERVISOR", ""))
                saf = line.get("group_safety",     getattr(cfg, "LINE_OA_GROUP_SAFETY", ""))
                emg = line.get("group_emergency",  getattr(cfg, "LINE_OA_GROUP_EMERGENCY", ""))
                if any(k in line for k in ("group_supervisor", "group_safety", "group_emergency")):
                    cfg.LINE_OA_GROUP_SUPERVISOR = sup
                    cfg.LINE_OA_GROUP_SAFETY     = saf
                    cfg.LINE_OA_GROUP_EMERGENCY  = emg
            except ImportError:
                pass
            print("[Pipeline] ⚙️  Settings applied")
        except Exception as e:
            print(f"[Pipeline] apply_settings: {e}")

    # ── Private helpers ───────────────────────────────────────

    def _apply_config(self, src_cfg: dict):
        try:
            import config as cfg
            cfg.CAMERA_SOURCE   = src_cfg.get("source", "webcam")
            cfg.WEBCAM_INDEX    = int(src_cfg.get("webcam_index", 0))
            cfg.RTSP_URL        = src_cfg.get("rtsp_url", getattr(cfg, "RTSP_URL", ""))
            cfg.VIDEO_FILE_PATH = src_cfg.get("video_file_path", "")
            cfg.ZONE_POLYGON_FILE = str(_APP_DATA / "zones.json")
        except ImportError:
            pass

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

    def _set_camera_state(self, state: str):
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

    def _reconnect_camera(self) -> bool:
        self._set_camera_state("reconnecting")
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
                slept = 0.0
                while slept < wait and not self._stop_evt.is_set() and self._running:
                    time.sleep(0.2)
                    slept += 0.2
        return False

    def _process_loop(self):
        try:
            try:
                import config as cfg
                is_file   = (cfg.CAMERA_SOURCE == "file")
                is_webcam = (cfg.CAMERA_SOURCE == "webcam")
            except ImportError:
                is_file   = False
                is_webcam = True

            reader = _FrameReader(self._cap)
            reader.start()
            self._reader = reader

            frame_id      = 0
            read_failures = 0
            print("[Pipeline] ▶️  Process loop running (passthrough — no detection)")

            while not self._stop_evt.is_set() and self._running:
                ret, raw = (self._reader.read() if self._reader else (False, None))
                if not ret or raw is None:
                    read_failures += 1
                    if read_failures > 40 and not is_file:
                        print("[Pipeline] ⚠️  Camera signal lost — reconnecting")
                        if not self._reconnect_camera():
                            break
                        read_failures = 0
                    continue
                read_failures = 0

                frame_id += 1
                flip = self._flip_override if self._flip_override is not None else is_webcam
                if flip:
                    raw = cv2.flip(raw, 1)

                with self._frame_lock:
                    self._latest_frame = raw

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
