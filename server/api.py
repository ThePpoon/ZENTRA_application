"""
server/api.py — ZENTRA FastAPI Server (Stage B — Real Pipeline)
REST + SSE + WebSocket endpoints backed by the real AI pipeline.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
UI_DIR   = BASE_DIR / "ui"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Backend (AI) project + its auto-collected data
ZENTRA_BACKEND = BASE_DIR.parent / "ZENTRA"
COLLECTED_DIR  = ZENTRA_BACKEND / "data" / "collected"
_DATA_CATEGORIES = ["ppe_violations", "zone_intrusions", "fall_events", "normal"]

# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(title="ZENTRA API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# ── Globals (set on startup) ─────────────────────────────────
_loop:       asyncio.AbstractEventLoop | None = None
_broadcaster = None
pipeline     = None   # Pipeline singleton


# ================================================================
# WEBSOCKET MANAGER
# ================================================================
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ================================================================
# IN-MEMORY EVENT HISTORY
# ================================================================
_event_history: deque[dict] = deque(maxlen=500)
_event_id_counter = 0


def _add_event(msg: str, level: str) -> dict:
    global _event_id_counter
    _event_id_counter += 1
    # Infer type from level
    type_map = {"warning": "ppe", "alert": "zone", "emergency": "fall"}
    event = {
        "id":        _event_id_counter,
        "type":      type_map.get(level, "ppe"),
        "level":     level,
        "message":   msg.split("\n")[0] if msg else level,
        "time":      datetime.now().strftime("%H:%M:%S"),
        "camera":    "Cam 1",
        "track":     None,
        "line_sent": True,
    }
    _event_history.appendleft(event)
    return event


# ================================================================
# STARTUP / SHUTDOWN
# ================================================================
@app.on_event("startup")
async def _startup():
    global _loop, _broadcaster, pipeline

    _loop = asyncio.get_running_loop()

    try:
        # Import Pipeline (adds ZENTRA backend to sys.path, imports cv2/numpy)
        from pipeline.pipeline          import Pipeline
        from pipeline.frame_broadcaster import FrameBroadcaster

        pipeline = Pipeline()

        # Wire alert callback → WebSocket broadcast + history
        def _on_alert(msg: str, level: str):
            event = _add_event(msg, level)
            # Authoritative counts from the pipeline (avoids client drift)
            with pipeline._lock:
                alerts = dict(pipeline.status.get("alerts", {}))
            broadcast_msg = {
                "type":      "event",
                "event":     "alert",
                "level":     level,
                "message":   event["message"],
                "timestamp": event["time"],
                "camera":    "Cam 1",
                "alerts":    alerts,
            }
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(broadcast_msg), _loop
            )

        pipeline.on_alert = _on_alert

        # Wire status changes (camera connect/reconnect/disconnect) → WebSocket
        def _on_status(status: dict):
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "type":    "event",
                    "event":   "status",
                    "modules": status.get("modules", {}),
                    "alerts":  status.get("alerts", {}),
                    "camera":  status.get("camera", "disconnected"),
                    "running": status.get("running", False),
                }),
                _loop,
            )

        pipeline.on_status = _on_status

        # Apply any saved settings to config before first run
        try:
            pipeline.apply_settings(_load_settings())
        except Exception as e:
            print(f"[API] settings preload skipped: {e}")

        # Start frame broadcaster (frames → WebSocket at 10 fps)
        _broadcaster = FrameBroadcaster(pipeline, manager, _loop, fps=10)
        _broadcaster.start()
        print("[API] Startup complete ✅")

    except Exception as e:
        print(f"[API] ⚠️  Startup warning (pipeline not loaded): {e}")
        print("[API] Server running in UI-only mode")


@app.on_event("shutdown")
async def _shutdown():
    global _broadcaster
    if _broadcaster:
        _broadcaster.stop()
    if pipeline:
        pipeline.stop()
        # Stop LINE sender if loaded
        try:
            from alerts.line_notify import stop_sender
            stop_sender()
        except Exception:
            pass
    print("[API] Shutdown complete")


# ================================================================
# STATIC / ROOT
# ================================================================
@app.get("/")
async def root():
    return FileResponse(str(UI_DIR / "index.html"))


# ================================================================
# SPLASH — SSE init progress
# ================================================================
@app.get("/api/init")
async def init_stream():
    steps = [
        (15,  "กำลังโหลดการตั้งค่า..."),
        (35,  "ตรวจสอบ Inference Server..."),
        (55,  "เตรียม AI โมดูล..."),
        (75,  "โหลดข้อมูลโซนความปลอดภัย..."),
        (90,  "เริ่มต้นระบบ..."),
        (100, "พร้อมใช้งาน"),
    ]

    async def _gen():
        for pct, msg in steps:
            yield f"data: {json.dumps({'percent': pct, 'message': msg})}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ================================================================
# STATUS
# ================================================================
@app.get("/api/status")
async def status():
    if pipeline is None:
        return JSONResponse({
            "running": False, "source": None,
            "modules": {"ppe": "error", "zone": "error", "fall": "error"},
            "alerts":  {"total": 0, "warning": 0, "emergency": 0},
            "uptime":  0, "last_emergency": None,
        })
    with pipeline._lock:
        s = dict(pipeline.status)
    s["uptime"] = pipeline.get_uptime()
    return JSONResponse(s)


# ================================================================
# PIPELINE  start / stop
# ================================================================
@app.post("/api/pipeline/start")
async def pipeline_start(body: dict[str, Any]):
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "pipeline not initialised"}, status_code=503)

    source  = body.get("source", "webcam")
    src_cfg = {
        "source":          source,
        "webcam_index":    int(body.get("webcam_index", 0)),
        "rtsp_url":        body.get("rtsp_url", ""),
        "video_file_path": body.get("video_file_path", ""),
    }

    # Run blocking start() in thread pool so we don't block the event loop
    loop   = asyncio.get_running_loop()
    ok     = await loop.run_in_executor(None, pipeline.start, src_cfg)
    if not ok:
        return JSONResponse({"ok": False, "error": "カメラを開けません"}, status_code=400)

    return JSONResponse({"ok": True, "source": source})


@app.post("/api/pipeline/stop")
async def pipeline_stop():
    if pipeline:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline.stop)
    return JSONResponse({"ok": True})


# ================================================================
# ZONES
# ================================================================
ZONES_FILE = DATA_DIR / "zones.json"


def _load_zones() -> list:
    if ZONES_FILE.exists():
        return json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    return []


def _save_zones(zones: list) -> None:
    ZONES_FILE.write_text(json.dumps(zones, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/zones")
async def get_zones():
    return JSONResponse(_load_zones())


@app.post("/api/zones")
async def create_zone(body: dict[str, Any]):
    zones = _load_zones()
    zone  = {
        "id":      max((z.get("id", 0) for z in zones), default=0) + 1,
        "name":    body.get("name", f"Zone {len(zones) + 1}"),
        "color":   body.get("color", "#ef4444"),
        "points":  body.get("points", []),
        "enabled": True,
    }
    zones.append(zone)
    _save_zones(zones)
    if pipeline:
        pipeline.reload_zones()
    return JSONResponse(zone)


@app.put("/api/zones/{zone_id}")
async def update_zone(zone_id: int, body: dict[str, Any]):
    zones = _load_zones()
    for z in zones:
        if z.get("id") == zone_id:
            z.update({k: v for k, v in body.items() if k != "id"})
            _save_zones(zones)
            if pipeline:
                pipeline.reload_zones()
            return JSONResponse(z)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: int):
    zones = [z for z in _load_zones() if z.get("id") != zone_id]
    _save_zones(zones)
    if pipeline:
        pipeline.reload_zones()
    return JSONResponse({"ok": True})


# ================================================================
# SETTINGS
# ================================================================
SETTINGS_FILE = DATA_DIR / "settings.json"

SETTINGS_DEFAULTS: dict[str, Any] = {
    "line": {
        "channel_access_token": "",
        "group_supervisor": "",
        "group_safety": "",
        "group_emergency": "",
    },
    "ai": {
        "ppe_confidence": 0.70,
        "fall_bbox_ratio": 0.72,
        "fall_confirm_frames": 6,
    },
    "alerts": {
        "violation_cooldown_seconds": 30,
        "zone_cooldown_seconds": 20,
        "fall_cooldown_seconds": 15,
        "warning_enabled": True,
        "alert_enabled":   True,
        "emergency_enabled": True,
    },
    "camera": {
        "source": "webcam",
        "webcam_index": 0,
        "rtsp_url": "",
        "video_file_path": "",
        "flip_horizontal": True,
    },
    "display": {
        "stream_fps": 10,
        "stream_jpeg_quality": 70,
    },
}


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        saved  = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = {**SETTINGS_DEFAULTS}
        for k, v in saved.items():
            if isinstance(v, dict) and k in merged:
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    return dict(SETTINGS_DEFAULTS)


def _save_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/settings")
async def get_settings():
    return JSONResponse(_load_settings())


@app.post("/api/settings")
async def save_settings(body: dict[str, Any]):
    _save_settings(body)
    if pipeline:
        pipeline.apply_settings(body)
    return JSONResponse({"ok": True})


# ================================================================
# SNAPSHOT  (for Zone Editor canvas background)
# ================================================================

# 1×1 dark pixel PNG fallback (no camera)
_DARK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@app.get("/api/frame/snapshot")
async def frame_snapshot():
    if pipeline and pipeline.is_running():
        snap = pipeline.get_snapshot()
        if snap:
            return StreamingResponse(iter([snap]), media_type="image/jpeg")
    return StreamingResponse(iter([_DARK_PNG]), media_type="image/png")


# ================================================================
# HISTORY
# ================================================================
@app.get("/api/history/today")
async def history_today():
    events = list(_event_history)
    total     = len(events)
    emergency = sum(1 for e in events if e["level"] == "emergency")
    ppe_v     = sum(1 for e in events if e["type"] == "ppe")
    zone_i    = sum(1 for e in events if e["type"] == "zone")
    falls     = sum(1 for e in events if e["type"] == "fall")
    uptime    = pipeline.get_uptime() if pipeline else 0
    return JSONResponse({
        "total":          total,
        "emergency":      emergency,
        "ppe_violations": ppe_v,
        "zone_intrusions":zone_i,
        "falls":          falls,
        "uptime_seconds": uptime,
    })


@app.get("/api/history/hourly")
async def history_hourly():
    hourly = {str(h).zfill(2): 0 for h in range(24)}
    for e in _event_history:
        hh = e["time"][:2]
        if hh in hourly:
            hourly[hh] += 1
    return JSONResponse(hourly)


@app.get("/api/history/events")
async def history_events(limit: int = 20, offset: int = 0):
    events = list(_event_history)
    page   = events[offset: offset + limit]
    return JSONResponse({
        "events":   page,
        "total":    len(events),
        "has_more": (offset + limit) < len(events),
    })


@app.get("/api/history/export.csv")
async def history_export():
    lines = ["id,type,level,message,time,camera,track,line_sent"]
    for e in _event_history:
        msg = e["message"].replace(",", " ")
        lines.append(
            f"{e['id']},{e['type']},{e['level']},{msg},"
            f"{e['time']},{e['camera']},{e.get('track','')},{e['line_sent']}"
        )
    return StreamingResponse(
        iter(["\n".join(lines)]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=zentra_history.csv"},
    )


# ================================================================
# DATA COLLECTION (training dataset) + JOBS (train / upload)
# ================================================================
def _dir_size_mb(path: Path) -> float:
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    return round(total / (1024 * 1024), 1)


@app.get("/api/data/stats")
async def data_stats():
    cats = {}
    for cat in _DATA_CATEGORIES:
        d = COLLECTED_DIR / cat
        cats[cat] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return JSONResponse({
        "categories":   cats,
        "total_images": sum(cats.values()),
        "size_mb":      _dir_size_mb(COLLECTED_DIR),
        "labeled":      sum(
            1 for cat in _DATA_CATEGORIES
            for j in (COLLECTED_DIR / cat).glob("*.jpg")
            if j.with_suffix(".txt").exists()
        ) if COLLECTED_DIR.exists() else 0,
    })


@app.post("/api/data/clear")
async def data_clear(body: dict[str, Any] | None = None):
    body = body or {}
    cats = [body["category"]] if body.get("category") in _DATA_CATEGORIES else _DATA_CATEGORIES
    removed = 0
    for cat in cats:
        d = COLLECTED_DIR / cat
        if not d.exists():
            continue
        for f in list(d.glob("*.jpg")) + list(d.glob("*.txt")):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return JSONResponse({"ok": True, "removed_files": removed})


@app.post("/api/jobs/train")
async def jobs_train(body: dict[str, Any]):
    from server.jobs import manager as jobs
    task    = body.get("task", "ppe")
    if task not in ("ppe", "fall"):
        return JSONResponse({"ok": False, "error": "task ต้องเป็น ppe หรือ fall"}, status_code=400)
    args = ["training.trainer", "--task", task, "--export"]
    project = body.get("project")
    if project:
        args += ["--project", str(project)]
    ok, msg = jobs.start(args, label=f"เทรน {task.upper()}")
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.post("/api/jobs/upload")
async def jobs_upload(body: dict[str, Any]):
    from server.jobs import manager as jobs
    task = body.get("task", "ppe")
    if task not in ("ppe", "fall", "zone"):
        return JSONResponse({"ok": False, "error": "task ไม่ถูกต้อง"}, status_code=400)
    args = ["training.upload", "--task", task]
    project = body.get("project")
    if project:
        args += ["--project", str(project)]
    ok, msg = jobs.start(args, label=f"อัปโหลด {task.upper()} → Roboflow")
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.get("/api/jobs/status")
async def jobs_status():
    from server.jobs import manager as jobs
    return JSONResponse(jobs.status())


@app.post("/api/jobs/stop")
async def jobs_stop():
    from server.jobs import manager as jobs
    return JSONResponse({"ok": jobs.stop()})


# ================================================================
# WEBSOCKET
# ================================================================
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status on connect (real values, not placeholders)
        modules = {"ppe": "error", "zone": "error", "fall": "error"}
        alerts  = {"total": 0, "warning": 0, "emergency": 0}
        camera  = "disconnected"
        if pipeline:
            with pipeline._lock:
                modules = dict(pipeline.status.get("modules", modules))
                alerts  = dict(pipeline.status.get("alerts", alerts))
                camera  = pipeline.status.get("camera", camera)
        await websocket.send_json({
            "type":    "event",
            "event":   "status",
            "modules": modules,
            "alerts":  alerts,
            "camera":  camera,
        })
        # Keep connection alive; frames arrive via FrameBroadcaster
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
