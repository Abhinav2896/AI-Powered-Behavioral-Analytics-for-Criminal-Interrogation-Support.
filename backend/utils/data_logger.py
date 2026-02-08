import csv
import os
from datetime import datetime

class DataLogger:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(base_dir, f"session_{ts}.csv")
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "lip_state", "blink_count", "blink_per_min", "confidence"])

    def log(self, lip_state, blink_count, blink_per_min, confidence):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([datetime.now().isoformat(), lip_state, blink_count, blink_per_min, confidence])
