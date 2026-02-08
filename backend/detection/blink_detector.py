import time
from collections import deque
import numpy as np

class BlinkDetector:
    def __init__(self, debounce_min_ms=200, debounce_max_ms=800, thresh_factor=0.75):
        self.baseline_ear = None
        self.thresh_factor = thresh_factor
        self.state = "OPEN"
        self.closed_start = None
        self.cooldown_until = 0.0
        self.debounce_min = debounce_min_ms / 1000.0
        self.debounce_max = debounce_max_ms / 1000.0
        self.timestamps = deque()
        self.ema_alpha = 0.95

    def _dist(self, p1, p2):
        return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])

    def _ear_eye(self, eye):
        h = self._dist(eye[0], eye[1])
        v1 = self._dist(eye[2], eye[3])
        v2 = self._dist(eye[4], eye[5])
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def compute_ear(self, left_eye, right_eye):
        le = self._ear_eye(left_eye)
        re = self._ear_eye(right_eye)
        return (le + re) / 2.0

    def update(self, left_eye, right_eye, ts=None):
        if ts is None:
            ts = time.time()
        ear = self.compute_ear(left_eye, right_eye)
        if self.baseline_ear is None:
            self.baseline_ear = ear
        else:
            self.baseline_ear = self.ema_alpha * self.baseline_ear + (1.0 - self.ema_alpha) * ear
        thresh = self.baseline_ear * self.thresh_factor
        if ts < self.cooldown_until:
            return self.stats(ts), ear, thresh
        if self.state == "OPEN":
            if ear < thresh:
                self.state = "CLOSED"
                self.closed_start = ts
        else:
            if ear >= thresh:
                dur = ts - (self.closed_start or ts)
                if self.debounce_min <= dur <= self.debounce_max:
                    self.timestamps.append(ts)
                    cutoff = ts - 60.0
                    while self.timestamps and self.timestamps[0] < cutoff:
                        self.timestamps.popleft()
                    self.cooldown_until = ts + 0.2
                self.state = "OPEN"
                self.closed_start = None
        return self.stats(ts), ear, thresh

    def stats(self, ts=None):
        if ts is None:
            ts = time.time()
        cutoff = ts - 60.0
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        c = len(self.timestamps)
        return c, float(c) if c > 0 else None
