"""
server/store.py — ZENTRA local event store (PDPA: on-device only)

A tiny SQLite-backed store for safety events + evidence snapshots.
Everything lives under data/ on the local machine — no cloud, no external
service. Used by the History page and daily reports.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional

_DATA_DIR  = Path(__file__).parent.parent / "data"
_DB_PATH   = _DATA_DIR / "zentra.db"
_SNAP_DIR  = _DATA_DIR / "snapshots"

# level → module type (PPE=warning, Zone=alert, Fall=emergency)
_TYPE_BY_LEVEL = {"warning": "ppe", "alert": "zone", "emergency": "fall"}

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DATA_DIR.mkdir(exist_ok=True)
        _SNAP_DIR.mkdir(exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT NOT NULL,
                day       TEXT NOT NULL,
                hour      TEXT NOT NULL,
                type      TEXT NOT NULL,
                level     TEXT NOT NULL,
                message   TEXT,
                camera    TEXT,
                snapshot  TEXT,
                line_sent INTEGER DEFAULT 0
            )
        """)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_events_day ON events(day)")
        _conn.commit()
    return _conn


# ── Write ─────────────────────────────────────────────────────
def add_event(level: str, message: str, camera: str = "Cam 1",
              frame_jpeg: Optional[bytes] = None, line_sent: bool = True,
              type_: Optional[str] = None) -> dict:
    """Insert an event, optionally saving a local evidence snapshot."""
    now  = datetime.now()
    typ  = type_ or _TYPE_BY_LEVEL.get(level, "ppe")
    msg  = (message or "").split("\n")[0][:300]
    with _lock:
        cur = _db().execute(
            "INSERT INTO events (ts, day, hour, type, level, message, camera, snapshot, line_sent)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.strftime("%Y-%m-%d"),
             now.strftime("%H"), typ, level, msg, camera, None, 1 if line_sent else 0),
        )
        eid = cur.lastrowid
        snap_name = None
        if frame_jpeg:
            snap_name = f"{eid}.jpg"
            try:
                (_SNAP_DIR / snap_name).write_bytes(frame_jpeg)
                _db().execute("UPDATE events SET snapshot=? WHERE id=?", (snap_name, eid))
            except OSError:
                snap_name = None
        _db().commit()
    return {
        "id": eid, "type": typ, "level": level, "message": msg,
        "time": now.strftime("%H:%M:%S"), "camera": camera,
        "has_snapshot": snap_name is not None, "line_sent": line_sent,
    }


# ── Read ──────────────────────────────────────────────────────
def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def today_stats(day: Optional[str] = None) -> dict:
    day = day or _today()
    with _lock:
        rows = _db().execute(
            "SELECT type, level, COUNT(*) c FROM events WHERE day=? GROUP BY type, level", (day,)
        ).fetchall()
    total = emergency = ppe = zone = fall = 0
    for r in rows:
        total += r["c"]
        if r["level"] == "emergency": emergency += r["c"]
        if r["type"] == "ppe":  ppe  += r["c"]
        if r["type"] == "zone": zone += r["c"]
        if r["type"] == "fall": fall += r["c"]
    return {"total": total, "emergency": emergency, "ppe_violations": ppe,
            "zone_intrusions": zone, "falls": fall, "day": day}


def hourly(day: Optional[str] = None) -> dict:
    day = day or _today()
    out = {f"{h:02d}": 0 for h in range(24)}
    with _lock:
        rows = _db().execute(
            "SELECT hour, COUNT(*) c FROM events WHERE day=? GROUP BY hour", (day,)
        ).fetchall()
    for r in rows:
        if r["hour"] in out:
            out[r["hour"]] = r["c"]
    return out


def list_events(limit: int = 20, offset: int = 0, day: Optional[str] = None) -> dict:
    with _lock:
        where = "WHERE day=?" if day else ""
        params: tuple = (day,) if day else ()
        total = _db().execute(f"SELECT COUNT(*) c FROM events {where}", params).fetchone()["c"]
        rows = _db().execute(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    events = [{
        "id": r["id"], "type": r["type"], "level": r["level"], "message": r["message"],
        "time": r["ts"][11:19] if r["ts"] else "", "date": r["day"],
        "camera": r["camera"], "has_snapshot": bool(r["snapshot"]),
        "line_sent": bool(r["line_sent"]),
    } for r in rows]
    return {"events": events, "total": total, "has_more": (offset + limit) < total}


def available_days() -> list[str]:
    with _lock:
        rows = _db().execute("SELECT DISTINCT day FROM events ORDER BY day DESC").fetchall()
    return [r["day"] for r in rows]


def snapshot_path(event_id: int) -> Optional[Path]:
    with _lock:
        row = _db().execute("SELECT snapshot FROM events WHERE id=?", (event_id,)).fetchone()
    if row and row["snapshot"]:
        p = _SNAP_DIR / row["snapshot"]
        return p if p.exists() else None
    return None


def export_csv(day: Optional[str] = None) -> str:
    data = list_events(limit=100000, offset=0, day=day)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "date", "time", "type", "level", "message", "camera", "snapshot", "line_sent"])
    for e in data["events"]:
        w.writerow([e["id"], e["date"], e["time"], e["type"], e["level"],
                    e["message"], e["camera"], "yes" if e["has_snapshot"] else "",
                    "yes" if e["line_sent"] else ""])
    return buf.getvalue()


# ── PDPA: retention / erasure (local only) ───────────────────
def purge_all() -> int:
    with _lock:
        n = _db().execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        _db().execute("DELETE FROM events")
        _db().commit()
    for f in _SNAP_DIR.glob("*.jpg"):
        try: f.unlink()
        except OSError: pass
    return n


def purge_before(days: int) -> int:
    """Delete events older than `days` (data minimisation)."""
    from datetime import timedelta
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    with _lock:
        rows = _db().execute("SELECT id, snapshot FROM events WHERE day < ?", (cutoff,)).fetchall()
        for r in rows:
            if r["snapshot"]:
                try: (_SNAP_DIR / r["snapshot"]).unlink()
                except OSError: pass
        _db().execute("DELETE FROM events WHERE day < ?", (cutoff,))
        _db().commit()
    return len(rows)
