import cv2
import threading
import time
from .emotion_engine import EmotionEngine
from .micro_expression_engine import MicroExpressionEngine
import numpy as np
import time as _time
from collections import deque
from .state_manager import manager

class VideoStream:
    def __init__(self, emotion_engine: EmotionEngine):
        self.emotion_engine = emotion_engine
        self.micro_engine = MicroExpressionEngine()
        self.camera = None
        self.is_running = False
        # Initialize with a "Loading" frame to prevent hanging requests
        self.latest_frame = self._create_loading_frame()
        self.latest_result = None
        self.lock = threading.Lock()
        self.use_mock = False
        self.retry_count = 0
        self.max_retries = 5
        self.black_frame_count = 0  # Counter for consecutive black frames
        
        # FPS Optimization
        self.frame_count = 0
        self.process_every_n_frames = 1 # Process every frame for maximum responsiveness
        self.current_results = [] # Store results for drawing on skipped frames
        
        # BSI smoothing
        self.bsi_history = deque(maxlen=30)
        self.ema_bsi = 0.0
        self.ema_alpha = 0.3
        
        # Flag stability control
        self.active_flags = []  # list of (text, timestamp)
        self.pending_flags = [] # queue of text
        self.last_flag_emit_time = 0.0
        self.flag_min_visible_secs = 3.0
        
        # FPS calculation
        self.fps = 0
        self._fps_counter = 0
        self._fps_last_time = _time.time()
        
    def _create_loading_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Initializing Camera...", (160, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
        
    def start(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        print("Video stream thread started.")

    def stop(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        print("Video stream stopped.")

    def _open_camera(self):
        # Try indices 0, 1, 2 with CAP_DSHOW (Windows) and default
        indices = [0, 1, 2]
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        
        for index in indices:
            for backend in backends:
                backend_name = "CAP_DSHOW" if backend == cv2.CAP_DSHOW else "CAP_ANY"
                print(f"Attempting to open camera index {index} with {backend_name}...", flush=True)
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        # Read a test frame
                        ret, frame = cap.read()
                        if ret and frame is not None and np.mean(frame) > 10:
                            print(f"Camera index {index} ({backend_name}) opened successfully.", flush=True)
                            return cap
                        else:
                            print(f"Camera index {index} opened but returned dark/empty frame (mean={np.mean(frame) if frame is not None else 'None'}). Releasing...", flush=True)
                            cap.release()
                except Exception as e:
                    print(f"Error opening camera {index}: {e}", flush=True)
                    
        print("CRITICAL: Failed to open any camera.", flush=True)
        return None

    def _generate_mock_frame(self):
        # Create a black image with text "Camera Unavailable"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Unavailable", (180, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Using Mock Mode", (200, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return frame

    def _draw_debug_overlay(self, frame, analysis):
        """
        Draws debug information for blink detection and other metrics.
        """
        if frame is None: return
        
        # 1. EAR Value
        ear = analysis.get("ear", 0.0)
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 2. Blink Rate
        blink_rate = analysis.get("blink_rate", 0)
        cv2.putText(frame, f"Blinks/min: {blink_rate}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 3. Blink Detected Indicator
        if analysis.get("blink_detected", False):
            cv2.putText(frame, "BLINK!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (220, 80), 6, (0, 255, 0), -1)
            
        # 4. Threshold Graph (Bottom Left)
        if "ear_history" in analysis:
            history = analysis["ear_history"]
            if len(history) > 1:
                graph_w = 200
                graph_h = 100
                x_start = 10
                y_start = 470
                
                # Draw background
                cv2.rectangle(frame, (x_start, y_start - graph_h), (x_start + graph_w, y_start), (0, 0, 0), -1)
                
                # Plot EAR
                for i in range(1, len(history)):
                    pt1 = (int(x_start + (i-1) * (graph_w / len(history))), 
                           int(y_start - history[i-1] * 300)) # Scale factor 300
                    pt2 = (int(x_start + i * (graph_w / len(history))), 
                           int(y_start - history[i] * 300))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                
                # Plot Threshold
                thresh = analysis.get("blink_threshold", 0)
                if thresh:
                    y_thresh = int(y_start - thresh * 300)
                    cv2.line(frame, (x_start, y_thresh), (x_start + graph_w, y_thresh), (0, 255, 255), 1)
        
        # 5. Lip Compression Debug
        lip_active = analysis.get("lip_compression", False)
        lip_score = float(analysis.get("lip_compression_score", 0.0))
        color = (0, 255, 0) if not lip_active else (0, 0, 255)
        cv2.putText(frame, f"Lip Stress: {lip_score:.0f}", (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if lip_active:
            cv2.putText(frame, "LIP COMPRESSION", (400, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Simple lip distance graph (top-right)
        # We do not have per-frame lip histories in analysis; draw placeholder bar by score
        bar_len = int(min(100, lip_score) * 2)
        cv2.rectangle(frame, (400, 80), (400 + bar_len, 95), (0, 0, 255) if lip_active else (0, 255, 0), -1)

    def _update(self):
        print("DEBUG: VideoStream _update loop started", flush=True)
        while self.is_running:
            try:
                if self.camera is None or not self.camera.isOpened():
                     self.camera = self._open_camera()
                     if self.camera is None or not self.camera.isOpened():
                         print("Failed to open camera.")
                         self.retry_count += 1
                         time.sleep(2)
                         continue

                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame. Retrying...")
                    self.camera.release()
                    self.camera = None
                    self.retry_count += 1
                    time.sleep(1)
                    continue
                
                # Reset retry count on success
                self.retry_count = 0

                # Check for black frames
                if frame is not None:
                    mean_val = np.mean(frame)
                    if self.frame_count % 60 == 0:
                        print(f"DEBUG: Frame mean brightness: {mean_val:.2f}", flush=True)

                    if mean_val < 5.0: # Very dark/black frame
                         self.black_frame_count += 1
                         if self.black_frame_count % 30 == 0:
                             print(f"Warning: Camera frame is very dark (mean={mean_val:.1f}). Ensure lighting is adequate.")
                    else:
                        self.black_frame_count = 0
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # FPS Optimization: Run inference only every N frames
                    self.frame_count += 1
                    
                    # 1. Micro-Expression Analysis (Runs every frame)
                    micro_analysis = self.micro_engine.process_frame(frame)
                    
                    # 2. Emotion Analysis
                    if self.frame_count % self.process_every_n_frames == 0:
                        self.current_results = self.emotion_engine.detect_faces_and_emotions(frame)
                        
                    # Compute FPS
                    self._fps_counter += 1
                    now = _time.time()
                    if now - self._fps_last_time >= 1.0:
                        self.fps = self._fps_counter
                        self._fps_counter = 0
                        self._fps_last_time = now
                    
                    # Unified Behavioral Stress Index (BSI)
                    emotion_label = None
                    if self.current_results:
                        try:
                            emotion_label = self.current_results[0].get("emotion")
                        except Exception:
                            emotion_label = None
                    
                    blink_rate = micro_analysis.get("blink_rate", 0)
                    # BlinkScore: normalize with baseline 15->0, 40->100
                    if blink_rate <= 15:
                        blink_score = 0.0
                    else:
                        blink_score = min(100.0, max(0.0, ((blink_rate - 15) / 25.0) * 100.0))
                    
                    lip_score = float(micro_analysis.get("lip_compression_score", 0.0))
                    
                    # EmotionStressScore mapping
                    emotion_map = {
                        "Fear": 90.0,
                        "Angry": 85.0,
                        "Sad": 75.0,
                        "Surprise": 60.0,
                        "Neutral": 20.0,
                        "Happy": 10.0
                    }
                    emotion_score = float(emotion_map.get(emotion_label, 40.0))
                    
                    bsi_raw = 0.4 * blink_score + 0.3 * lip_score + 0.3 * emotion_score
                    bsi_raw = max(0.0, min(100.0, bsi_raw))
                    
                    # Smoothing: rolling average + EMA
                    self.bsi_history.append(bsi_raw)
                    avg_bsi = sum(self.bsi_history) / len(self.bsi_history)
                    self.ema_bsi = self.ema_alpha * bsi_raw + (1.0 - self.ema_alpha) * self.ema_bsi
                    bsi_smoothed = (avg_bsi + self.ema_bsi) / 2.0
                    micro_analysis["behavioral_stress_index"] = round(bsi_smoothed, 1)
                    micro_analysis["stress_index"] = micro_analysis["behavioral_stress_index"]
                    
                    # Flag grouping & debouncing
                    incoming_flags = micro_analysis.get("flags", []) or []
                    grouped = []
                    for f in incoming_flags:
                        if "Blink" in f:
                            canon = "High Blink Rate (Stress)"
                        elif "Lip Compression" in f:
                            canon = "Lip Compression (Stress Suppression)"
                        elif "Gaze" in f or "Looking Down" in f:
                            canon = "Eye Avoidance (Discomfort)"
                        else:
                            canon = f
                        if canon not in grouped:
                            grouped.append(canon)
                    
                    # enqueue unseen flags
                    for g in grouped:
                        if g not in [af[0] for af in self.active_flags] and g not in self.pending_flags:
                            self.pending_flags.append(g)
                    
                    # expire old active flags (min visible 3s)
                    now_ts = _time.time()
                    self.active_flags = [(t, ts) for (t, ts) in self.active_flags if now_ts - ts < self.flag_min_visible_secs]
                    
                    # rate limit: at most 1 new flag per second
                    if self.pending_flags and (now_ts - self.last_flag_emit_time >= 1.0) and len(self.active_flags) < 5:
                        new_flag = self.pending_flags.pop(0)
                        self.active_flags.append((new_flag, now_ts))
                        self.last_flag_emit_time = now_ts
                    
                    micro_analysis["flags"] = [t for (t, _) in self.active_flags]
                    micro_analysis["fps"] = self.fps
                    
                    face_detected = bool(self.current_results)
                    emotion_label = self.current_results[0]["emotion"] if face_detected else None
                    emotion_confidence = self.current_results[0]["confidence"] if face_detected else 0.0
                    lip_state = "lip_compression" if micro_analysis.get("lip_compression") else "lip_calm"
                    blink_count = micro_analysis.get("blink_count_last_60s", 0)
                    blink_per_minute = micro_analysis.get("blink_per_minute", None)
                    stress_index = micro_analysis.get("stress_index", 0.0)
                    manager.update({
                        "fps": self.fps,
                        "face_detected": face_detected,
                        "emotion": emotion_label,
                        "emotion_score": float(emotion_confidence),
                        "blink_detected": bool(micro_analysis.get("blink_detected", False)),
                        "blink_count": int(blink_count) if isinstance(blink_count, (int, float)) else 0,
                        "blink_per_minute": blink_per_minute if isinstance(blink_per_minute, (int, float)) else None,
                        "lip_state": lip_state,
                        "stress_index": float(stress_index),
                        "micro_expressions": micro_analysis
                    })
                    
                    # Update latest_result for API consumers
                    with self.lock:
                        combined_result = {
                            "emotion_data": self.current_results[0] if self.current_results else None,
                            "micro_data": micro_analysis
                        }
                        self.latest_result = combined_result
                    
                    # Draw results on EVERY frame (using current or cached results)
                    annotated_frame = self.emotion_engine.draw_results(frame, self.current_results)
                    
                    # Draw Micro-Expression Overlay (Blink Graph, EAR, etc.)
                    self._draw_debug_overlay(annotated_frame, micro_analysis)
                    
                    with self.lock:
                        self.latest_frame = annotated_frame
                
                    # Minimal sleep to prevent 100% CPU usage, but keep it fast
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error in video stream: {e}")
                time.sleep(1)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                # Return a basic placeholder if no frame yet
                return None
            
            ret, jpeg = cv2.imencode('.jpg', self.latest_frame)
            if not ret:
                return None
                
            return jpeg.tobytes()

    def get_latest_result(self):
        with self.lock:
            return self.latest_result
