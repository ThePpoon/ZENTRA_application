#!/usr/bin/env python3
# main.py -- ZENTRA System Entry Point
# Windows 11 | Python 3.11
# Run: python main.py
# ================================================================
# FIXES v4:
#   - All terminal output changed to English (fixes ??? on Windows)
#   - UTF-8 stdout encoding forced
#   - Detection pipeline verified
#   - LINE alert integrated correctly
# ================================================================

from __future__ import annotations
import sys
import io
import os

# ── Force UTF-8 output on Windows (fixes ???) ────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")

import cv2
import time
import queue
import threading
import traceback
import numpy as np
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit("Python 3.10+ required (recommend 3.11)")

import config as cfg
from alerts.line_notify   import start_sender, stop_sender
from reports.daily_report import get_logger, get_scheduler
from utils.collector      import get_collector
import modules.ppe         as ppe_module
import modules.safety_zone as zone_module
import modules.heat_stroke as fall_module


# ================================================================
# INFERENCE CLIENT
# ================================================================
def _make_client():
    try:
        from inference_sdk import InferenceHTTPClient
        ppe_id  = (cfg.PPE_LOCAL_MODEL
                   if cfg.USE_LOCAL_MODEL and Path(cfg.PPE_LOCAL_MODEL).exists()
                   else cfg.PPE_MODEL_ID)
        fall_id = (cfg.FALL_LOCAL_MODEL
                   if cfg.USE_LOCAL_MODEL and Path(cfg.FALL_LOCAL_MODEL).exists()
                   else cfg.FALL_MODEL_ID)
        client = InferenceHTTPClient(
            api_url=cfg.INFERENCE_SERVER_URL,
            api_key=cfg.ROBOFLOW_API_KEY,
        )
        print(f"[Main] Inference server : {cfg.INFERENCE_SERVER_URL}")
        print(f"[Main] PPE  model       : {ppe_id}")
        print(f"[Main] Fall model       : {fall_id}")
        return client, ppe_id, fall_id
    except Exception as e:
        print(f"[Main] WARNING - inference_sdk unavailable: {e}")
        print("[Main] Running without cloud inference (local model only)")
        return None, cfg.PPE_MODEL_ID, cfg.FALL_MODEL_ID


# ================================================================
# CAMERA
# ================================================================
def open_camera() -> cv2.VideoCapture:
    src = cfg.CAMERA_SOURCE
    if src == "webcam":
        backend = cv2.CAP_DSHOW if cfg.USE_DSHOW else cv2.CAP_ANY
        cap = cv2.VideoCapture(cfg.WEBCAM_INDEX, backend)
        print(f"[Main] Camera: Webcam index={cfg.WEBCAM_INDEX}")
    elif src == "rtsp":
        cap = cv2.VideoCapture(cfg.RTSP_URL, cv2.CAP_FFMPEG)
        print(f"[Main] Camera: RTSP {cfg.RTSP_URL}")
    elif src == "file":
        cap = cv2.VideoCapture(cfg.VIDEO_FILE_PATH)
        print(f"[Main] Camera: File {cfg.VIDEO_FILE_PATH}")
    else:
        raise ValueError(f"Unknown CAMERA_SOURCE: {src}")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (source={src})")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.FRAME_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS, cfg.TARGET_FPS)
    return cap


# ================================================================
# ASYNC FRAME READER
# ================================================================
class FrameReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name="FrameReader")
        self.cap         = cap
        self.q           = queue.Queue(maxsize=cfg.FRAME_BUFFER_SIZE)
        self._stop       = threading.Event()
        self.error_count = 0

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.error_count += 1
                if self.error_count > 30:
                    print("[FrameReader] WARNING: Too many read errors")
                time.sleep(0.05)
                continue
            self.error_count = 0
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
# ASYNC INFERENCE WORKER
# ================================================================
class InferenceWorker(threading.Thread):
    def __init__(self, client, ppe_id: str, fall_id: str):
        super().__init__(daemon=True, name="InferenceWorker")
        self.client  = client
        self.ppe_id  = ppe_id
        self.fall_id = fall_id
        self.in_q    = queue.Queue(maxsize=2)
        self.out_q   = queue.Queue(maxsize=4)
        self._stop   = threading.Event()
        self.error_count = 0

    def run(self):
        while not self._stop.is_set():
            try:
                fid, frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                ppe_preds = fall_preds = []
                if self.client:
                    r1 = self.client.infer(frame, model_id=self.ppe_id)
                    ppe_preds  = r1.get("predictions", [])
                    r2 = self.client.infer(frame, model_id=self.fall_id)
                    fall_preds = r2.get("predictions", [])
                self.out_q.put((fid, ppe_preds, fall_preds))
                self.error_count = 0
            except Exception as e:
                self.error_count += 1
                if self.error_count <= 3 or self.error_count % 30 == 0:
                    print(f"[Inference] Error: {e}")

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


# ================================================================
# KEYBOARD HANDLER
# ================================================================
def handle_key(key: int) -> bool:
    """Return True = should stop"""
    if key in (ord("q"), ord("Q")):
        print("\n[Main] Q pressed - stopping...")
        return True
    elif key in (ord("z"), ord("Z")):
        zone_module.toggle_draw_mode()
    elif key in (ord("c"), ord("C")):
        zone_module.clear_all_zones()
    elif key in (ord("t"), ord("T")):
        _bg_train()
    elif key in (ord("s"), ord("S")):
        _show_stats()
    elif key in (ord("h"), ord("H")):
        _show_help()
    return False


def _show_help():
    print("\n[Main] ------- Controls -------------------------")
    print("  Q  Quit system")
    print("  Z  Start drawing Safety Zone")
    print("  C  Clear all zones")
    print("  T  Start model training in background")
    print("  S  Show statistics")
    print("  H  Show this help")
    print("-------------------------------------------------\n")


def _show_stats():
    col = get_collector()
    print("\n[Main] ------- Statistics -----------------------")
    print(f"  PPE Violations  : {ppe_module.stats['violations']}")
    print(f"  Zone Intrusions : {zone_module.stats['intrusions']}")
    print(f"  Fall Events     : {fall_module.stats['falls']}")
    print(f"  Frames          : {ppe_module.stats['frames']:,}")
    print(f"  Avg FPS         : {ppe_module.get_fps():.1f}")
    print(f"  Collected       : {col.get_stats()}")
    print("-------------------------------------------------\n")


def _bg_train():
    def _run():
        print("\n[Training] Starting background training...")
        try:
            from training.trainer import run_training_pipeline
            run_training_pipeline(task="ppe", augment=True)
        except Exception as e:
            print(f"[Training] ERROR: {e}")
    threading.Thread(target=_run, daemon=True, name="Trainer").start()
    print("[Main] Training started in background (press S to check progress)")


# ================================================================
# METADATA
# ================================================================
class _Meta:
    __slots__ = ("frame_id",)
    def __init__(self, fid: int):
        self.frame_id = fid


# ================================================================
# OSD
# ================================================================
def _draw_osd(frame: np.ndarray):
    """Draw status bar overlay on frame"""
    fps = ppe_module.get_fps()

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 38), cfg.OSD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    osd = (
        f"ZENTRA | FPS:{fps:.1f} "
        f"Frames:{ppe_module.stats['frames']:,} "
        f"Violations:{ppe_module.stats['violations']} "
        f"Alerts:{ppe_module.stats['alerts_sent']}"
    )
    cv2.putText(frame, osd, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
                cfg.OSD_COLOR, cfg.FONT_THICKNESS, cv2.LINE_AA)

    try:
        from ppe_config import PPE_PROFILE
        profile_str = f"[PPE: {PPE_PROFILE}]"
        (pw, _), _ = cv2.getTextSize(profile_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, profile_str, (frame.shape[1] - pw - 8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 120), 1, cv2.LINE_AA)
    except Exception:
        pass


# ================================================================
# MAIN LOOP
# ================================================================
def main():
    print("=" * 60)
    print("  ZENTRA -- Zone Environment Network Thermal Risk Analysis")
    print(f"  Platform : Windows 11")
    print(f"  Camera   : {cfg.CAMERA_SOURCE}")
    print(f"  Server   : {cfg.INFERENCE_SERVER_URL}")
    print(f"  PPE Model: {cfg.PPE_MODEL_ID}")
    print(f"  Fall Model:{cfg.FALL_MODEL_ID}")
    print("=" * 60)

    # ── Subsystems ───────────────────────────────────────────────
    start_sender()
    scheduler = get_scheduler()
    scheduler.start()
    logger    = get_logger()
    collector = get_collector()

    # ── Inference client ─────────────────────────────────────────
    client, ppe_id, fall_id = _make_client()

    # ── Camera ──────────────────────────────────────────────────
    cap    = open_camera()
    reader = FrameReader(cap)
    if cfg.ENABLE_THREADING:
        reader.start()

    # ── Inference worker ─────────────────────────────────────────
    inf_worker = InferenceWorker(client, ppe_id, fall_id)
    if cfg.ENABLE_THREADING:
        inf_worker.start()

    # ── OpenCV window ────────────────────────────────────────────
    cv2.namedWindow(cfg.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(cfg.WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(cfg.WINDOW_TITLE, cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT)
    cv2.setMouseCallback(cfg.WINDOW_TITLE, zone_module.mouse_callback)

    # ── Loop state ───────────────────────────────────────────────
    frame_id         = 0
    last_ppe_preds:  list = []
    last_fall_preds: list = []
    last_annotated:  np.ndarray | None = None
    last_raw:        np.ndarray | None = None  # raw frame ไม่มี overlay ส่ง LINE

    _show_help()
    print("[Main] System ready!\n")

    try:
        while True:
            # 1. Read frame
            if cfg.ENABLE_THREADING:
                ret, raw = reader.read()
            else:
                ret, raw = cap.read()

            if not ret or raw is None:
                time.sleep(0.04)
                continue

            frame_id += 1
            if cfg.CAMERA_SOURCE == "webcam":
                raw = cv2.flip(raw, 1)

            # เก็บ raw frame ก่อน annotate ใช้ส่ง LINE (ไม่มี UI overlay)
            last_raw = raw.copy()

            # 2. Submit inference every N frames
            if frame_id % cfg.INFER_EVERY_N_FRAMES == 0:
                if cfg.ENABLE_THREADING:
                    inf_worker.submit(frame_id, raw)
                elif client:
                    try:
                        r1 = client.infer(raw, model_id=ppe_id)
                        last_ppe_preds  = r1.get("predictions", [])
                        r2 = client.infer(raw, model_id=fall_id)
                        last_fall_preds = r2.get("predictions", [])
                    except Exception as e:
                        print(f"[Inference] {e}")

            # 3. Collect inference results
            if cfg.ENABLE_THREADING:
                res = inf_worker.get_result()
                if res:
                    _, last_ppe_preds, last_fall_preds = res

            # 4. Build annotated frame
            annotated = raw.copy()
            annotated = ppe_module.draw_predictions(annotated, last_ppe_preds)
            annotated = fall_module.draw_fall_predictions(annotated, last_fall_preds)

            meta = _Meta(frame_id)
            fall_module.on_frame(annotated, meta, cfg.WINDOW_TITLE)
            zone_module.on_frame(annotated, meta, cfg.WINDOW_TITLE)

            ppe_module.stats["frames"] += 1
            _draw_osd(annotated)

            last_annotated = annotated.copy()

            # 5. Data callbacks
            # frame=last_annotated → debug/collect (มี bbox)
            # raw_frame=last_raw   → LINE alert (ภาพสะอาด ไม่มี UI)
            ppe_module.on_data(
                {"predictions": last_ppe_preds}, meta,
                frame=last_annotated, raw_frame=last_raw)
            zone_module.on_data(
                {"predictions": last_ppe_preds}, meta,
                frame=last_annotated, raw_frame=last_raw)
            fall_module.on_data(
                {"predictions": last_fall_preds}, meta,
                frame=last_annotated, raw_frame=last_raw)

            # 6. Daily logger
            if frame_id % 150 == 0:
                logger.update_frames(ppe_module.stats["frames"])

            # 7. Show
            cv2.imshow(cfg.WINDOW_TITLE, annotated)

            # 8. Keyboard
            key = cv2.waitKey(1) & 0xFF
            if handle_key(key):
                break

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C pressed")
    except Exception:
        print("\n[Main] Unhandled error:")
        traceback.print_exc()
    finally:
        if cfg.ENABLE_THREADING:
            reader.stop()
            inf_worker.stop()
        stop_sender()
        scheduler.stop()
        cap.release()
        cv2.destroyAllWindows()
        _show_stats()
        print("=" * 60)
        print("  ZENTRA shutdown complete")
        print("=" * 60)


if __name__ == "__main__":
    main()
