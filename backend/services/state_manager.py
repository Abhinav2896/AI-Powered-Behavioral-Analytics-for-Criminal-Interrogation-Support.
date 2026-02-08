import threading
import time

class AIStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "timestamp": None,
            "fps": 0,
            "face_detected": False,
            "emotion": None,
            "emotion_score": 0.0,
            "blink_detected": False,
            "blink_count": 0,
            "blink_per_minute": None,
            "lip_state": "lip_calm",
            "stress_index": 0.0,
            "micro_expressions": None
        }

    def update(self, data):
        with self._lock:
            self._state.update(data)
            self._state["timestamp"] = time.time()

    def get(self):
        with self._lock:
            return dict(self._state)

manager = AIStateManager()
