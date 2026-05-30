# ZENTRA — Safety AI Desktop Application

Native Windows desktop app (PyWebView + FastAPI) for the ZENTRA factory
safety system. Wraps the AI backend (`../ZENTRA`) in a 6-screen GUI:
Splash → Source → Live Dashboard → Zone Editor → History → Settings.

## Layout

```
c:\ZENTRA\
├─ ZENTRA\              ← AI backend (PPE / Safety Zone / Heat-Stroke)
│   ├─ modules\         ppe.py, safety_zone.py, heat_stroke.py
│   ├─ alerts\          line_notify.py
│   ├─ config.py        reads .env
│   ├─ .env             local config (git-ignored)
│   ├─ docs\            proposals, slides, notebooks
│   └─ archive\         old backups / dead code
└─ ZENTRA_application\  ← THIS app (desktop UI + pipeline wrapper)
    ├─ app.py           entry point (PyWebView window + uvicorn)
    ├─ run_zentra.ps1   one-click launcher (starts inference server too)
    ├─ pipeline\        Pipeline + FrameBroadcaster
    ├─ server\          FastAPI (api.py)
    └─ ui\              screens + assets
```

## The 3 AI modules and what each needs

| Module | Tech | Needs |
|--------|------|-------|
| **PPE Detection** | YOLO via Roboflow | inference server on `localhost:9001` |
| **Safety Zone** | ByteTrack + polygons | inference server (uses `person` detections) |
| **Heat-Stroke / Fall** | MediaPipe Pose (+ Roboflow fall fallback) | `mediapipe` (local, no server) |

> PPE and Zone **do not work** unless the inference server is running.
> The launcher starts it automatically.

## Run it (recommended)

```powershell
cd c:\ZENTRA\ZENTRA_application
.\run_zentra.ps1
```

or double-click **`run_zentra.bat`**.

The launcher:
1. Checks `http://localhost:9001` — if not up, starts Docker Desktop and the
   `roboflow/roboflow-inference-server-cpu` container (pulls image on first run).
2. Waits until the server responds.
3. Launches `python app.py`.

## Run it (manual)

```powershell
# 1) inference server (once; stays up via --restart)
docker run -d --name zentra-inference --restart unless-stopped -p 9001:9001 ^
  roboflow/roboflow-inference-server-cpu:latest

# 2) the app
cd c:\ZENTRA\ZENTRA_application
python app.py
```

## First-time setup

```powershell
# Backend deps (AI)
cd c:\ZENTRA\ZENTRA
copy .env.example .env        # then edit if needed (LINE tokens, camera)
pip install -r requirements.txt

# App deps (UI)
cd c:\ZENTRA\ZENTRA_application
pip install -r requirements.txt
```

> **MediaPipe note:** mediapipe is pinned to `0.10.14` and protobuf to
> `>=4.25.3,<5`. Newer protobuf (5/7) breaks MediaPipe pose with
> `FieldDescriptor ... has no attribute 'label'`. Keep these pins.

## GPU (optional, faster)

The CPU inference image works everywhere. For GPU (RTX), once Docker Desktop
has WSL2 GPU support configured:

```powershell
docker run -d --name zentra-inference --gpus all --restart unless-stopped ^
  -p 9001:9001 roboflow/roboflow-inference-server-gpu:latest
```

## LINE alerts (optional)

Leave LINE tokens blank to run without delivery — events still show in the
app's History and Dashboard. To enable, set the tokens in
`..\ZENTRA\.env` or via the app's **Settings → การแจ้งเตือน Line**.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| PPE/Zone never detect anything | inference server down → run launcher or `docker start zentra-inference` |
| Heat-stroke pose not drawn | mediapipe broken → `pip install --force-reinstall --no-deps mediapipe==0.10.14` + `pip install "protobuf>=4.25.3,<5"` |
| App stuck on splash | hard-refresh; see `.claude/skills/zentra-dev` (WebView2 script-injection notes) |
| Emoji/Thai console crash | app.py forces UTF-8 stdout; run via the launcher |
