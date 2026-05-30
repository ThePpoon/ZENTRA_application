---
name: zentra-dev
description: >
  Use when running, debugging, or extending the ZENTRA Safety AI desktop
  application (the PyWebView + FastAPI app in c:\ZENTRA\ZENTRA_application
  that wraps the ZENTRA AI backend in c:\ZENTRA\ZENTRA). Covers the
  architecture, how to launch it, the Edge WebView2 gotchas that cause
  blank/stuck screens, the SPA router contract, and the Pipeline
  integration. Read this before touching app.py, server/api.py,
  pipeline/*, or ui/* so you don't re-discover the same traps.
---

# ZENTRA Desktop Application — Developer Skill

A native Windows desktop app that wraps the **existing, unchanged** ZENTRA
AI backend (`c:\ZENTRA\ZENTRA`) in a 6-screen dark-themed GUI.

- **App layer** (this repo): `c:\ZENTRA\ZENTRA_application` — repo `ZENTRA_application`
- **AI backend** (separate): `c:\ZENTRA\ZENTRA` — PPE / Safety Zone / Fall modules

## Run it

```
cd c:\ZENTRA\ZENTRA_application
python app.py
```

`app.py` starts **uvicorn in a daemon thread** (`127.0.0.1:7788`), waits ~2s,
then opens a **PyWebView** window (Edge WebView2) pointed at the local server.
The flow is: Splash → Source Selection → Live Dashboard → (Zone Editor /
History / Settings tabs).

It runs **without** a Roboflow inference server or a LINE token — you just lose
YOLO boxes (MediaPipe pose still runs) and LINE delivery (events still log).

## Architecture

```
app.py
├─ daemon thread: uvicorn → server/api.py (FastAPI)
└─ main thread:   PyWebView window → http://127.0.0.1:7788/

server/api.py (FastAPI)
├─ @startup: build Pipeline(), wire on_alert → WS broadcast + event history,
│            start FrameBroadcaster (frames → WS @10fps)
├─ REST: /api/status /api/pipeline/{start,stop} /api/zones(CRUD)
│        /api/settings /api/frame/snapshot /api/history/*
└─ WS  : /ws/stream  (frames + events, multiplexed by msg.type)

pipeline/pipeline.py — Pipeline class (wraps the backend's main.py loop)
├─ start(src_cfg) / stop() / is_running()
├─ _process_loop (daemon): read → infer(worker) → annotate → store frame
│   passes window_title="" to modules so cv2.imshow() is skipped
├─ get_latest_frame() / get_snapshot() / get_uptime()
├─ reload_zones() / apply_settings()
└─ monkey-patches send_line_notify in ALL module namespaces → on_alert cb

pipeline/frame_broadcaster.py — daemon: latest frame → resize → JPEG →
                                base64 → manager.broadcast() via the loop
```

## ⚠️ Edge WebView2 traps (these cost real debugging time — don't repeat)

1. **`element.innerHTML = html` does NOT execute `<script>` tags.**
   The SPA router (`ZENTRA.navigate`) injects screen HTML, so each screen's
   `init_<screen>()` would never run. Fix already in `app.js`: after setting
   innerHTML, **re-create** every `<script>` as a fresh element and append it
   so the browser executes it.

2. **Use script-element injection, NOT `eval()`, for inline scripts.**
   Direct `eval(text)` runs in the *local* scope of `navigate()`, so
   `function init_x(){}` is defined locally and `window['init_x']` stays
   undefined → screen looks stuck. A re-created `<script>` element executes in
   **global** scope, which is what the router needs.

3. **External CDN scripts must be awaited.** Chart.js (history page) is loaded
   by appending a `<script src>` to `<head>`; `navigate()` awaits its `onload`
   before calling `init_history()`, else `Chart` is undefined. Loaded once,
   tracked via `data-cdn` attribute.

4. **Top-level `let`/`const` in screen scripts → use `var`.** Classic scripts
   share one global lexical environment. Re-navigating to a screen re-runs its
   script; a second top-level `let X`/`const X` throws
   `SyntaxError: already declared`, killing that screen. `var` redeclaration is
   allowed. (Variables *inside* `init_*()` functions are fine.)

5. **SSE (`EventSource`) is unreliable inside WebView2** — it buffers and may
   never fire `onmessage`. The splash progress bar was switched from an SSE
   stream to a local `setTimeout` animation. Avoid SSE for anything the UI
   depends on; prefer WebSocket or polling.

## SPA router contract (`ui/assets/app.js`)

- `ZENTRA.navigate(screenId)` fetches `/ui/screens/<id>.html`, injects it,
  re-executes scripts, then calls `window['init_<id>']()`.
- Every screen file is an HTML fragment + a `<script>` that defines
  `function init_<id>() { ... }`. Main screens call
  `renderNavbar('<id>')` into `#navbar-mount`.
- Global state lives on `ZENTRA.state` (`pipeline`, `modules`, `alerts`,
  `uptime`, `last_emergency`, `camera_label`).
- WebSocket messages: `{type:'frame', data:<base64 jpeg>}` updates
  `#video-feed`; `{type:'event', ...}` updates module dots / alert counters /
  emergency banner.

## Pipeline integration rules

- **Never edit the AI backend's logic.** The only allowed change is the
  `if window_title:` guard before `cv2.imshow()` in `modules/ppe.py` and
  `modules/safety_zone.py` (so the headless pipeline doesn't pop OpenCV
  windows from a background thread).
- The pipeline mirrors `main.py`'s threading model (`FrameReader`,
  `InferenceWorker`, process loop) but stores the annotated frame in
  `self._latest_frame` instead of `cv2.imshow()`.
- **asyncio/threading bridge:** the FrameBroadcaster and the alert callback
  run on threads but must talk to async WebSockets — they use
  `asyncio.run_coroutine_threadsafe(manager.broadcast(...), _loop)` where
  `_loop` is captured in the FastAPI `@startup` handler.
- **Alerts:** `send_line_notify` is monkey-patched in every module namespace
  (`ppe`, `safety_zone`, `heat_stroke`, and `alerts.line_notify`) because each
  module did `from alerts.line_notify import send_line_notify` (binds a local
  name). Patch must hit all of them or some alerts won't reach the UI.

## Data contracts

- `data/zones.json`: list of `{id, name, color, points:[[x,y],...], enabled}`.
  Points are **raw pixel coords** at camera resolution. The backend's
  `safety_zone._load_zones()` reads `points`/`name` and ignores the rest, so
  the format is compatible. After any zone CRUD, the API calls
  `pipeline.reload_zones()` → `safety_zone._load_zones()` (no restart).
- `data/settings.json`: merged over `SETTINGS_DEFAULTS` in `api.py`. Saving
  calls `pipeline.apply_settings()` to push values into `config` at runtime.

## Where to look when X breaks

| Symptom | Likely cause / file |
|---|---|
| Screen blank / stuck on splash | script not executing — `app.js navigate()` script re-injection |
| Screen breaks on 2nd visit | top-level `let`/`const` in that screen → `var` |
| No video on dashboard | pipeline not running, or FrameBroadcaster/WS down — check `/api/status`, server console |
| Chart missing on history | Chart.js `onload` not awaited, or CDN blocked |
| Alerts not in UI | monkey-patch missed a module namespace |
| RTSP/file won't connect | body field names must be `rtsp_url`/`video_file_path`/`webcam_index` (match `api.py`) |
| OpenCV window pops up | a module's `cv2.imshow()` missing the `if window_title:` guard |

## Conventions

- Commit per phase, push to `origin/main` (`https://github.com/ThePpoon/ZENTRA_application.git`).
- Thai UI copy; font is Sarabun (Google Fonts CDN). Online CDN is allowed.
- Design tokens in `ui/assets/style.css`: `--bg:#0d1b2a`, `--accent:#7ecfff`,
  `--green:#4ade80`, `--red:#f87171`, `--emergency:#ef4444`.
