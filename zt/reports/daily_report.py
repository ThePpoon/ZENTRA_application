# reports/daily_report.py — ZENTRA Daily Safety Report
# ================================================================
# ส่งรายงานประจำวันผ่าน LINE OA ทุกวัน เวลา 20:00 น.
# ================================================================

from __future__ import annotations
import json
import time
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import schedule

import config as cfg
from alerts.line_notify import send_line_notify


# ================================================================
# DAILY LOGGER — เก็บสถิติรายวัน
# ================================================================
class DailyLogger:
    def __init__(self):
        self._lock = threading.Lock()
        self._reset()
        self._log_path = cfg.LOGS_DIR / "daily_stats.json"
        self._load()

    def _reset(self):
        self._data = {
            "date":             str(date.today()),
            "ppe_violations":   0,
            "zone_intrusions":  0,
            "fall_events":      0,
            "frames_processed": 0,
            "alerts_sent":      0,
            "uptime_start":     time.time(),
        }

    def _load(self):
        """โหลดข้อมูลวันนี้ถ้ามีอยู่แล้ว"""
        try:
            if self._log_path.exists():
                saved = json.loads(self._log_path.read_text())
                if saved.get("date") == str(date.today()):
                    with self._lock:
                        self._data.update(saved)
        except Exception:
            pass

    def _save(self):
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text(json.dumps(self._data, indent=2))
        except Exception as e:
            print(f"[Report] Save error: {e}")

    def log_ppe_violation(self):
        with self._lock:
            self._data["ppe_violations"] += 1
            self._data["alerts_sent"] += 1
        self._save()

    def log_zone_intrusion(self):
        with self._lock:
            self._data["zone_intrusions"] += 1
            self._data["alerts_sent"] += 1
        self._save()

    def log_fall(self):
        with self._lock:
            self._data["fall_events"] += 1
            self._data["alerts_sent"] += 1
        self._save()

    def update_frames(self, frames: int):
        with self._lock:
            self._data["frames_processed"] = frames
        self._save()

    def build_report(self) -> str:
        with self._lock:
            d = self._data.copy()
        uptime_sec  = time.time() - d.get("uptime_start", time.time())
        uptime_hr   = uptime_sec / 3600

        total       = d["ppe_violations"] + d["zone_intrusions"] + d["fall_events"]
        safety_pct  = max(0, 100 - (total * 5))   # rough estimate

        lines = [
            f"📊 ZENTRA รายงานความปลอดภัยประจำวัน",
            f"📅 {d['date']}  ⏰ {datetime.now().strftime('%H:%M')} น.",
            f"─" * 35,
            f"🪖 PPE Violations   : {d['ppe_violations']} ครั้ง",
            f"⛔ Zone Intrusions  : {d['zone_intrusions']} ครั้ง",
            f"🆘 Fall / Heat Stroke: {d['fall_events']} ครั้ง",
            f"─" * 35,
            f"📈 รวมเหตุการณ์      : {total} ครั้ง",
            f"✅ ดัชนีความปลอดภัย : {safety_pct:.0f}%",
            f"🎥 Frame ที่ประมวลผล : {d['frames_processed']:,}",
            f"⏱️  ทำงาน            : {uptime_hr:.1f} ชม.",
        ]

        if total == 0:
            lines.append(f"\n🎉 ไม่พบเหตุการณ์อันตรายในวันนี้ !")
        elif d["fall_events"] > 0:
            lines.append(f"\n⚠️  พบเหตุฉุกเฉิน {d['fall_events']} ครั้ง กรุณาตรวจสอบบันทึก")
        else:
            lines.append(f"\n💡 กรุณาดำเนินมาตรการป้องกันต่อเนื่อง")

        return "\n".join(lines)

    def send_daily_report(self):
        """ส่งรายงานผ่าน LINE และ reset สำหรับวันถัดไป"""
        report = self.build_report()
        print(f"\n[Report] 📨 Sending daily report...\n{report}\n")
        send_line_notify(
            report,
            image=None,
            level=cfg.ALERT_LEVEL_WARNING,
            cooldown_key="daily_report",
            cooldown_sec=3600,
        )
        # Reset สำหรับวันถัดไป
        self._reset()
        self._save()


# ================================================================
# SCHEDULER
# ================================================================
class ReportScheduler:
    def __init__(self, logger: DailyLogger):
        self._logger = logger
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()

    def start(self):
        report_time = cfg.DAILY_REPORT_TIME   # "20:00"
        schedule.every().day.at(report_time).do(self._logger.send_daily_report)
        print(f"[Report] ⏰ Daily report scheduled at {report_time}")

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ReportScheduler",
        )
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            schedule.run_pending()
            time.sleep(30)

    def stop(self):
        self._stop.set()
        schedule.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)


# ── Singletons ────────────────────────────────────────────────────
_logger:    Optional[DailyLogger]     = None
_scheduler: Optional[ReportScheduler] = None


def get_logger() -> DailyLogger:
    global _logger
    if _logger is None:
        _logger = DailyLogger()
    return _logger


def get_scheduler() -> ReportScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ReportScheduler(get_logger())
    return _scheduler
