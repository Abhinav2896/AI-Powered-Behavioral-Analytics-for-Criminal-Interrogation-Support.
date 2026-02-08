import threading
from .emotion_engine import EmotionEngine
from .video_stream import VideoStream
from .state_manager import manager

_lock = threading.Lock()
_engine = None
_stream = None

def start():
    global _engine, _stream
    with _lock:
        if _engine is None:
            _engine = EmotionEngine()
        if _stream is None:
            _stream = VideoStream(_engine)
            _stream.start()
    return _stream

def stop():
    global _stream
    with _lock:
        if _stream is not None:
            _stream.stop()
            _stream = None

def get_state():
    return manager.get()
