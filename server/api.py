import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent.parent
UI_DIR = BASE_DIR / "ui"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ZENTRA API", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


# ─── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(UI_DIR / "index.html"))


# ─── Init SSE (Splash progress) ───────────────────────────────────────────────

@app.get("/api/init")
async def init_stream():
    steps = [
        (20,  "กำลังโหลดการตั้งค่า..."),
        (40,  "ตรวจสอบการเชื่อมต่อ Inference Server..."),
        (60,  "เตรียม AI โมดูล..."),
        (80,  "โหลดข้อมูลโซนความปลอดภัย..."),
        (100, "พร้อมใช้งาน"),
    ]

    async def generate():
        for percent, message in steps:
            yield f"data: {json.dumps({'percent': percent, 'message': message})}\n\n"
            await asyncio.sleep(0.55)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Status ───────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return JSONResponse({
        "running": False,
        "source": None,
        "modules": {"ppe": "ok", "zone": "ok", "fall": "ok"},
        "alerts": {"total": 0, "warning": 0, "emergency": 0},
        "uptime": 0,
        "last_emergency": None,
    })


# ─── Pipeline (mock for Stage A) ─────────────────────────────────────────────

@app.post("/api/pipeline/start")
async def pipeline_start(body: dict[str, Any]):
    source = body.get("source", "webcam")
    return JSONResponse({"ok": True, "source": source})


@app.post("/api/pipeline/stop")
async def pipeline_stop():
    return JSONResponse({"ok": True})


# ─── Zones ────────────────────────────────────────────────────────────────────

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
    zone = {
        "id": max((z.get("id", 0) for z in zones), default=0) + 1,
        "name": body.get("name", f"Zone {len(zones) + 1}"),
        "color": body.get("color", "#ef4444"),
        "points": body.get("points", []),
        "enabled": True,
    }
    zones.append(zone)
    _save_zones(zones)
    return JSONResponse(zone)


@app.put("/api/zones/{zone_id}")
async def update_zone(zone_id: int, body: dict[str, Any]):
    zones = _load_zones()
    for z in zones:
        if z.get("id") == zone_id:
            z.update({k: v for k, v in body.items() if k != "id"})
            _save_zones(zones)
            return JSONResponse(z)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: int):
    zones = _load_zones()
    zones = [z for z in zones if z.get("id") != zone_id]
    _save_zones(zones)
    return JSONResponse({"ok": True})


# ─── Settings ─────────────────────────────────────────────────────────────────

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
        "alert_enabled": True,
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
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
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
    return JSONResponse({"ok": True})


# ─── History (mock data for Stage A) ─────────────────────────────────────────

MOCK_EVENTS = [
    {"id": 1, "type": "fall",    "level": "emergency", "message": "Fall — Zone A",        "time": "14:32:07", "camera": "Cam 1", "track": None,      "line_sent": True},
    {"id": 2, "type": "ppe",     "level": "warning",   "message": "PPE — ไม่สวมหมวก",    "time": "13:15:42", "camera": "Cam 1", "track": "Track #04","line_sent": True},
    {"id": 3, "type": "ppe",     "level": "warning",   "message": "PPE — ไม่ใส่เสื้อกั๊ก","time": "11:08:19", "camera": "Cam 1", "track": "Track #02","line_sent": True},
    {"id": 4, "type": "ppe",     "level": "warning",   "message": "PPE — ไม่สวมหมวก",    "time": "08:54:03", "camera": "Cam 1", "track": "Track #01","line_sent": True},
]

MOCK_HOURLY = {str(h).zfill(2): 0 for h in range(24)}
MOCK_HOURLY.update({"08": 1, "09": 0, "10": 0, "11": 1, "12": 2, "13": 1, "14": 1})


@app.get("/api/history/today")
async def history_today():
    return JSONResponse({
        "total": 4,
        "emergency": 1,
        "ppe_violations": 3,
        "zone_intrusions": 0,
        "falls": 1,
        "uptime_seconds": 30840,
    })


@app.get("/api/history/hourly")
async def history_hourly():
    return JSONResponse(MOCK_HOURLY)


@app.get("/api/history/events")
async def history_events(limit: int = 20, offset: int = 0):
    page = MOCK_EVENTS[offset: offset + limit]
    return JSONResponse({"events": page, "total": len(MOCK_EVENTS), "has_more": False})


@app.get("/api/history/export.csv")
async def history_export():
    lines = ["id,type,level,message,time,camera,track,line_sent"]
    for e in MOCK_EVENTS:
        lines.append(
            f"{e['id']},{e['type']},{e['level']},{e['message']},"
            f"{e['time']},{e['camera']},{e.get('track','')},{e['line_sent']}"
        )
    return StreamingResponse(
        iter(["\n".join(lines)]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=zentra_history.csv"},
    )


# ─── WebSocket (mock frames + events for Stage A) ────────────────────────────

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


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send mock status event immediately
        await websocket.send_json({
            "type": "event",
            "event": "status",
            "modules": {"ppe": "ok", "zone": "ok", "fall": "ok"},
            "alerts": {"total": 0, "warning": 0, "emergency": 0},
        })
        # Keep connection alive; real frames come in Stage B
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ─── Snapshot (mock: 1x1 dark pixel for Stage A) ─────────────────────────────

@app.get("/api/frame/snapshot")
async def frame_snapshot():
    import base64
    # 1×1 dark pixel PNG
    DARK_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return StreamingResponse(iter([DARK_PNG]), media_type="image/png")
