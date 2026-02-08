import time
from collections import deque
import numpy as np

class OnnxLandmarkEngine:
    def __init__(self, inferencer):
        self.infer = inferencer
        self.timestamps = deque()
        self.state = "OPEN"
        self.closed_start = None
        self.cooldown_until = 0.0
        self.ear_thresh = 0.2
        self.lar_history = deque(maxlen=30)
        self.last_result = None

    def _dist(self, a, b):
        return np.linalg.norm([a[0]-b[0], a[1]-b[1]])

    def _ear_eye(self, eye):
        h = self._dist(eye[0], eye[1])
        v1 = self._dist(eye[2], eye[3])
        v2 = self._dist(eye[4], eye[5])
        if h <= 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _lar(self, lip, all_pts):
        up = all_pts[self.infer.LIP_INNER_UP]
        lo = all_pts[self.infer.LIP_INNER_LO]
        lc = all_pts[self.infer.MOUTH_LEFT]
        rc = all_pts[self.infer.MOUTH_RIGHT]
        v = self._dist(up, lo)
        w = self._dist(lc, rc)
        if w <= 1e-6:
            return 0.0
        return v / w

    def _lip_state(self, lar, all_pts):
        self.lar_history.append(lar)
        state = "lip_calm"
        if lar > 0.35:
            state = "lip_open"
        elif lar < 0.15:
            state = "lip_compression"
        else:
            state = "lip_calm"
        shiver = False
        if len(self.lar_history) >= 10:
            arr = np.array(self.lar_history)
            diff = np.diff(arr)
            sc = np.sum(np.abs(np.diff(np.sign(diff))) > 0)
            amp = arr.max() - arr.min()
            shiver = sc >= 6 and amp < 0.08
        if state == "lip_compression":
            up = all_pts[self.infer.LIP_INNER_UP]
            lo = all_pts[self.infer.LIP_INNER_LO]
            bite = up[1] >= lo[1] - 0.002
            if bite:
                return "lip_biting"
        if shiver:
            return "lip_shiver"
        return state

    def process_frame(self, frame, ts=None):
        if ts is None:
            ts = time.time()
        sets = self.infer.extract_sets(frame)
        eye_l = sets["left_eye"]
        eye_r = sets["right_eye"]
        ear = (self._ear_eye(eye_l) + self._ear_eye(eye_r)) / 2.0
        lip = sets["lip"]
        all_pts = sets["all"]
        lar = self._lar(lip, all_pts)
        blink_detected = False
        if ts >= self.cooldown_until:
            if self.state == "OPEN":
                if ear < self.ear_thresh:
                    self.state = "CLOSED"
                    self.closed_start = ts
            else:
                if ear >= self.ear_thresh:
                    dur = ts - (self.closed_start or ts)
                    if 0.05 <= dur <= 0.4:
                        self.timestamps.append(ts)
                        cutoff = ts - 60.0
                        while self.timestamps and self.timestamps[0] < cutoff:
                            self.timestamps.popleft()
                        self.cooldown_until = ts + 0.2
                        blink_detected = True
                    self.state = "OPEN"
                    self.closed_start = None
        cutoff = ts - 60.0
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        blink_count = len(self.timestamps)
        blink_per_minute = float(blink_count) if blink_count > 0 else None
        lip_state = self._lip_state(lar, all_pts)
        self.last_result = {
            "blink_detected": blink_detected,
            "blink_count": int(blink_count),
            "blink_per_minute": blink_per_minute,
            "lip_state": lip_state,
            "ear": float(ear),
            "lar": float(lar)
        }
        return self.last_result

    def stats(self):
        ts = time.time()
        cutoff = ts - 60.0
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        c = len(self.timestamps)
        bpm = float(c) if c > 0 else None
        return int(c), bpm
