"""
server/report.py — ZENTRA daily safety report (local PDF via matplotlib)

Builds a one-page A4 PDF from the local event store. No external service,
no new dependency (matplotlib ships with the project). Thai text renders
with a Thai-capable Windows font (Tahoma / Leelawadee).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")                      # headless backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from server import store

_REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"

_FONT_READY = False


def _ensure_thai_font():
    global _FONT_READY
    if _FONT_READY:
        return
    for name in ("Tahoma", "Leelawadee UI", "Leelawadee", "TH Sarabun New", "Angsana New"):
        try:
            path = fm.findfont(name, fallback_to_default=False)
            if path and Path(path).exists():
                fm.fontManager.addfont(path)
                matplotlib.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def build_daily_pdf(day: Optional[str] = None) -> Path:
    """Render the daily report PDF and return its local path."""
    _ensure_thai_font()
    day = day or date.today().strftime("%Y-%m-%d")
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stats  = store.today_stats(day)
    hourly = store.hourly(day)
    events = store.list_events(limit=12, offset=0, day=day)["events"]

    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
    fig.patch.set_facecolor("white")

    # Header
    fig.text(0.06, 0.955, "ZENTRA — รายงานความปลอดภัยประจำวัน",
             fontsize=20, fontweight="bold", color="#0f172a")
    fig.text(0.06, 0.932, f"Industrial Safety AI System   ·   วันที่ {day}",
             fontsize=11, color="#475569")
    fig.text(0.06, 0.915, "NSC 2026", fontsize=9, color="#94a3b8")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.905, 0.905], color="#cdd6e3", lw=1))

    # KPI row
    kpis = [
        ("เหตุการณ์รวม", stats["total"],          "#2563eb"),
        ("ฉุกเฉิน",       stats["emergency"],      "#dc2626"),
        ("PPE",          stats["ppe_violations"], "#ea580c"),
        ("เข้าเขต",       stats["zone_intrusions"],"#d97706"),
        ("การล้ม",        stats["falls"],          "#dc2626"),
    ]
    n = len(kpis)
    for i, (label, val, color) in enumerate(kpis):
        x = 0.06 + i * (0.88 / n)
        fig.text(x + 0.02, 0.865, str(val), fontsize=26, fontweight="bold", color=color)
        fig.text(x + 0.02, 0.845, label, fontsize=10, color="#475569")

    # Hourly chart
    ax = fig.add_axes([0.08, 0.50, 0.86, 0.27])
    hours  = [f"{h:02d}" for h in range(24)]
    values = [hourly.get(h, 0) for h in hours]
    ax.bar(range(24), values, color="#2563eb", width=0.7)
    ax.set_title("จำนวน Alert รายชั่วโมง", fontsize=12, color="#0f172a", loc="left")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=8, color="#64748b")
    ax.tick_params(axis="y", labelsize=8, colors="#64748b")
    if max(values, default=0) <= 5:
        ax.set_ylim(0, 5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cdd6e3")
    ax.spines["bottom"].set_color("#cdd6e3")
    ax.grid(axis="y", color="#e2e8f0", lw=0.6)

    # Event list
    fig.text(0.06, 0.45, "บันทึกเหตุการณ์ล่าสุด", fontsize=12, fontweight="bold", color="#0f172a")
    y = 0.42
    if not events:
        fig.text(0.08, y, "— ไม่มีเหตุการณ์ในวันนี้ —", fontsize=10, color="#94a3b8")
    else:
        lvl_color = {"warning": "#d97706", "alert": "#ea580c", "emergency": "#dc2626"}
        for e in events:
            c = lvl_color.get(e["level"], "#475569")
            fig.text(0.08, y, f"{e['time']}", fontsize=9, color="#64748b")
            fig.text(0.20, y, e["type"].upper(), fontsize=9, fontweight="bold", color=c)
            fig.text(0.30, y, (e["message"] or "")[:60], fontsize=9, color="#0f172a")
            fig.text(0.88, y, "[ภาพ]" if e["has_snapshot"] else "", fontsize=8, color="#475569")
            y -= 0.028

    fig.text(0.06, 0.04,
             "ข้อมูลทั้งหมดจัดเก็บภายในเครื่อง (on-device) ตามหลัก PDPA · สร้างโดย ZENTRA",
             fontsize=8, color="#94a3b8")

    out = _REPORTS_DIR / f"zentra_report_{day}.pdf"
    fig.savefig(str(out), format="pdf")
    plt.close(fig)
    return out


def daily_stats_for_line(day: Optional[str] = None) -> dict:
    """Stats dict shaped for alerts.line_notify.send_daily_report (text only)."""
    s = store.today_stats(day)
    return {
        "ppe_violations":  s["ppe_violations"],
        "zone_intrusions": s["zone_intrusions"],
        "fall_events":     s["falls"],
    }
