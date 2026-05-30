# utils/tracker.py — ZENTRA ByteTrack Multi-Object Tracker
# ================================================================
# Simplified ByteTrack implementation
# ไม่ต้องติดตั้ง library เพิ่มเติม — ใช้แค่ numpy + scipy
# ================================================================

from __future__ import annotations
import time
import numpy as np
from collections import deque
from typing import List, Optional

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[Tracker] scipy ไม่พบ — ใช้ greedy matching")


# ================================================================
# TRACK STATE
# ================================================================
class TrackState:
    NEW       = 0
    TRACKED   = 1
    LOST      = 2
    REMOVED   = 3


# ================================================================
# SINGLE TRACK
# ================================================================
class Track:
    _id_counter = 0

    def __init__(self, bbox: np.ndarray, score: float):
        Track._id_counter += 1
        self.track_id  = Track._id_counter
        self.bbox      = bbox.copy()       # [x1,y1,x2,y2]
        self.score     = score
        self.state     = TrackState.NEW
        self.age       = 1
        self.hits      = 1
        self.time_since_update = 0
        self.history: deque = deque(maxlen=30)  # trail of centers
        self._update_history()

    def _update_history(self):
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        self.history.append((cx, cy))

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def update(self, bbox: np.ndarray, score: float):
        self.bbox  = bbox.copy()
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED
        self._update_history()

    def predict(self):
        """Simple linear motion model"""
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.state = TrackState.LOST

    def is_confirmed(self) -> bool:
        return self.hits >= 2

    def is_deleted(self) -> bool:
        return self.state == TrackState.REMOVED


# ================================================================
# IoU HELPER
# ================================================================
def _iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """คำนวณ IoU matrix [N x M]"""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    ax1, ay1, ax2, ay2 = bboxes_a[:,0], bboxes_a[:,1], bboxes_a[:,2], bboxes_a[:,3]
    bx1, by1, bx2, by2 = bboxes_b[:,0], bboxes_b[:,1], bboxes_b[:,2], bboxes_b[:,3]

    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:,None] + area_b[None,:] - inter

    return np.where(union > 0, inter / union, 0.0)


def _match(tracks: List[Track],
           dets: np.ndarray,
           iou_thresh: float) -> tuple:
    """จับคู่ track กับ detection — คืน (matched_pairs, unmatched_tracks, unmatched_dets)"""
    if not tracks or len(dets) == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))

    t_bboxes = np.array([t.bbox for t in tracks])
    iou_mat  = _iou_matrix(t_bboxes, dets)
    cost_mat = 1 - iou_mat

    if _HAS_SCIPY:
        t_idx, d_idx = linear_sum_assignment(cost_mat)
    else:
        # Greedy fallback
        t_idx, d_idx = [], []
        used_d = set()
        for ti in range(len(tracks)):
            best_d = -1
            best_v = 1.0
            for di in range(len(dets)):
                if di not in used_d and cost_mat[ti, di] < best_v:
                    best_v, best_d = cost_mat[ti, di], di
            if best_d >= 0:
                t_idx.append(ti)
                d_idx.append(best_d)
                used_d.add(best_d)

    matched, unmatched_t, unmatched_d = [], [], []
    for ti, di in zip(t_idx, d_idx):
        if iou_mat[ti, di] >= iou_thresh:
            matched.append((ti, di))
        else:
            unmatched_t.append(ti)
            unmatched_d.append(di)

    all_t = set(range(len(tracks)))
    all_d = set(range(len(dets)))
    for ti, _ in matched:
        all_t.discard(ti)
    for _, di in matched:
        all_d.discard(di)
    unmatched_t += list(all_t)
    unmatched_d += list(all_d)

    return matched, unmatched_t, unmatched_d


# ================================================================
# BYTETRACKER
# ================================================================
class ByteTracker:
    """
    Simplified ByteTrack Multi-Object Tracker
    ใช้สำหรับ Safety Zone monitoring
    """

    def __init__(
        self,
        track_thresh: float = 0.50,
        track_buffer: int   = 30,
        match_thresh: float = 0.80,
    ):
        self.track_thresh  = track_thresh
        self.track_buffer  = track_buffer
        self.match_thresh  = match_thresh

        self._tracked:   List[Track] = []
        self._lost:      List[Track] = []
        self._frame_id   = 0

    def reset(self):
        self._tracked  = []
        self._lost     = []
        self._frame_id = 0
        Track._id_counter = 0

    # ── Main update ────────────────────────────────────────────────
    def update(self, detections: list[dict]) -> List[Track]:
        """
        รับ list ของ predictions จาก Roboflow
        คืน list ของ Track ที่ active ในเฟรมนี้
        """
        self._frame_id += 1

        # Parse detections → np.ndarray [N, 5] (x1,y1,x2,y2,score)
        dets_high, dets_low = [], []
        for d in detections:
            score = d.get("confidence", 0.0)
            x = d.get("x", 0); y = d.get("y", 0)
            w = d.get("width", 0); h = d.get("height", 0)
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            bbox = np.array([x1, y1, x2, y2])
            if score >= self.track_thresh:
                dets_high.append((*bbox, score))
            elif score >= 0.1:
                dets_low.append((*bbox, score))

        dets_high = np.array(dets_high) if dets_high else np.zeros((0, 5))
        dets_low  = np.array(dets_low)  if dets_low  else np.zeros((0, 5))

        # Predict all existing tracks
        for t in self._tracked + self._lost:
            t.predict()

        # Stage 1: match tracked tracks with high-score detections
        matched1, unmatched_t1, unmatched_d1 = _match(
            self._tracked,
            dets_high[:, :4] if len(dets_high) else np.zeros((0,4)),
            iou_thresh=self.match_thresh,
        )
        for ti, di in matched1:
            self._tracked[ti].update(dets_high[di, :4], dets_high[di, 4])

        # Stage 2: match lost tracks with high-score detections
        unmatched_tracked = [self._tracked[i] for i in unmatched_t1]
        matched2, _, unmatched_d2_idx = _match(
            self._lost,
            dets_high[unmatched_d1, :4] if unmatched_d1 else np.zeros((0,4)),
            iou_thresh=self.match_thresh * 0.7,
        )
        for ti, di in matched2:
            self._lost[ti].update(dets_high[unmatched_d1[di], :4],
                                  dets_high[unmatched_d1[di], 4])
            self._lost[ti].state = TrackState.TRACKED

        # Stage 3: match unmatched tracked with low-score detections
        matched3, unmatched_t3, _ = _match(
            unmatched_tracked,
            dets_low[:, :4] if len(dets_low) else np.zeros((0,4)),
            iou_thresh=self.match_thresh * 0.5,
        )
        for ti, di in matched3:
            unmatched_tracked[ti].update(dets_low[di, :4], dets_low[di, 4])

        # Create new tracks for unmatched high-score detections
        truly_unmatched = set(unmatched_d1) - {unmatched_d1[di] for _, di in matched2}
        for di in truly_unmatched:
            new_t = Track(dets_high[di, :4], dets_high[di, 4])
            self._tracked.append(new_t)

        # Move to lost if not updated
        still_tracked = []
        for t in self._tracked:
            if t.time_since_update == 0:
                still_tracked.append(t)
            elif t.time_since_update <= self.track_buffer:
                self._lost.append(t)
        self._tracked = still_tracked

        # Move recovered lost tracks back
        for t in self._lost[:]:
            if t.state == TrackState.TRACKED:
                self._lost.remove(t)
                self._tracked.append(t)

        # Remove expired lost tracks
        self._lost = [t for t in self._lost
                      if t.time_since_update <= self.track_buffer]

        # Return confirmed active tracks
        return [t for t in self._tracked if t.is_confirmed()]
