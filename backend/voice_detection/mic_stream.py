
import threading
import time
import numpy as np
import sounddevice as sd
import queue

class MicStream:
    def __init__(self, rate=16000, chunk_duration=3):
        self.rate = rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(rate * chunk_duration)
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        self.stream = None
        self.thread = None

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        # We need to buffer this. sounddevice returns raw bytes or numpy array.
        # We want to accumulate enough for CHUNK_DURATION.
        # However, to keep it simple and real-time, we can just push smaller chunks 
        # and let the engine aggregate, OR we can read in blocking mode in a loop.
        # Let's use a non-blocking stream and push to a queue.
        # Actually, for simplicity and control, let's use a blocking read in a thread.
        pass

    def _record_loop(self):
        print("Microphone recording started...")
        try:
            with sd.InputStream(samplerate=self.rate, channels=1, callback=None, blocksize=self.chunk_size) as stream:
                while self.running:
                    # Read a chunk
                    data, overflow = stream.read(self.chunk_size)
                    if overflow:
                        print("Audio buffer overflow")
                    
                    # data is a numpy array of shape (frames, channels)
                    # We need to flatten it to mono
                    audio_data = data.flatten().astype(np.float32)
                    
                    # Put in queue, overwrite if full (latest data is more important)
                    if self.audio_queue.full():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.audio_queue.put(audio_data)
        except Exception as e:
            print(f"Microphone error: {e}")
            self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_audio(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
