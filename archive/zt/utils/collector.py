# utils/collector.py — ZENTRA Frame Data Collector
# ================================================================
# เก็บ frame ที่น่าสนใจ (violation / normal) ไว้สำหรับเทรนโมเดล
# ================================================================

from __future__ import annotations
import cv2
import time
import threading
from pathlib import Path
from typing import Optional
import numpy as np

import config as cfg


class FrameCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._normal_counter   = 0
        self._total_collected  = 0

    # ── Collect violation frame ────────────────────────────────────
    def collect(
        self,
        frame: np.ndarray,
        predictions: list,
        category: str,
        force: bool = False,
    ):
        """บันทึก frame ของ violation event"""
        if not cfg.COLLECT_VIOLATION_FRAMES and not force:
            return
        dest = cfg.COLLECTED_DIR / category
        dest.mkdir(parents=True, exist_ok=True)
        # จำกัดจำนวนไฟล์ใน folder
        existing = list(dest.glob("*.jpg"))
        if len(existing) >= cfg.COLLECT_MAX_PER_CLASS:
            # ลบไฟล์เก่าที่สุด
            oldest = min(existing, key=lambda p: p.stat().st_mtime)
            oldest.unlink(missing_ok=True)

        fname = dest / f"{category}_{int(time.time()*1000)}.jpg"
        try:
            cv2.imwrite(
                str(fname),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, cfg.COLLECT_JPEG_QUALITY],
            )
            with self._lock:
                self._total_collected += 1
        except Exception as e:
            print(f"[Collector] Save error: {e}")

    # ── Collect normal frame ───────────────────────────────────────
    def collect_normal(
        self,
        frame: np.ndarray,
        predictions: list,
        frame_id: int,
    ):
        """บันทึก frame ปกติ ทุก N frame (เพื่อสร้าง dataset สมดุล)"""
        if not cfg.AUTO_COLLECT_FRAMES:
            return
        if frame_id % cfg.COLLECT_NORMAL_INTERVAL != 0:
            return
        self.collect(frame, predictions, "normal")

    def get_stats(self) -> dict:
        with self._lock:
            stats = {"total_collected": self._total_collected}
        # นับไฟล์แต่ละ category
        for cat_dir in cfg.COLLECTED_DIR.iterdir():
            if cat_dir.is_dir():
                stats[cat_dir.name] = len(list(cat_dir.glob("*.jpg")))
        return stats


# ── Singleton ─────────────────────────────────────────────────────
_collector: Optional[FrameCollector] = None


def get_collector() -> FrameCollector:
    global _collector
    if _collector is None:
        _collector = FrameCollector()
    return _collector
