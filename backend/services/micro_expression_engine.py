import cv2
import mediapipe as mp
import numpy as np
import time
import os
from collections import deque

class MicroExpressionEngine:
    def __init__(self):
        # Use the new MediaPipe Tasks API for FaceLandmarker
        self.base_options = mp.tasks.BaseOptions
        self.vision_running_mode = mp.tasks.vision.RunningMode
        
        # Path to model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/face_landmarker.task"))
        
        # Create FaceLandmarker options
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self.base_options(model_asset_path=model_path),
            running_mode=self.vision_running_mode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False
        )
        
        # Initialize the face landmarker
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        
        # Landmark Indices (MediaPipe Face Mesh)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        self.MOUTH_LEFT_CORNER = 61
        self.MOUTH_RIGHT_CORNER = 291
        self.UPPER_LIP_INNER = 13
        self.LOWER_LIP_INNER = 14
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300]
        
        # State Tracking
        self.ear_history = deque(maxlen=5)
        self.blink_timestamps = deque(maxlen=60)
        self.mar_history = deque(maxlen=30)
        self.baseline_mar = 0.35
        self.lip_distance_history = deque(maxlen=30)
        self.corner_distance_history = deque(maxlen=30)
        self.mar_baseline_history = deque(maxlen=30)
        self.baseline_lip_distance = None
        self.baseline_corner_distance = None
        self.baseline_texture_variance = None
        self.baseline_calibration_started = None
        self.baseline_calibration_secs = 4.0
        self.lip_calibration_buffer = []
        
        self.blink_state = "OPEN"
        self.consecutive_closed_frames = 0
        self.blink_start_time = None
        self.last_blink_time = 0
        self.cooldown_timer = 0
        self.baseline_ear = None
        self.blink_threshold = None
        self.calibration_start_time = None
        self.calibration_buffer = []
        
        self.MIN_BLINK_DURATION = 0.05
        self.MAX_BLINK_DURATION = 0.4
        self.MIN_BLINK_INTERVAL = 0.2
        self.MAX_YAW = 30
        self.MAX_PITCH = 20
        
        self.flag_counters = {
            "gaze_down": 0, "lip_compress": 0, "head_shake": 0, "eyebrow_raise": 0, "eyebrow_frown": 0
        }
        self.PERSISTENCE_THRESHOLD = 3
        self.MAR_THRESHOLD = 0.30
        self.lip_compress_frames = 0
        self.lip_release_cooldown = 0.0
        self.LIP_COOLDOWN_SECS = 1.5
        self.LIP_REQUIRED_FRAMES = 8
        self.lip_compress_start_ts = None
        self.lip_score_ema = 0.0
        self.lip_score_alpha = 0.3
        
        self.frame_counter = 0
        self.last_process_time = time.time()
        self.blink_count = 0

    def calculate_ear(self, landmarks, indices):
        try:
            p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
            p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
            p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
            p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
            p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
            p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
            
            A = np.linalg.norm(p2 - p6)
            B = np.linalg.norm(p3 - p5)
            C = np.linalg.norm(p1 - p4)
            
            if C == 0: return 0.0
            return (A + B) / (2.0 * C)
        except Exception:
            return 0.0

    def calculate_mar(self, landmarks, indices):
        try:
            p_up = np.array([landmarks[self.UPPER_LIP_INNER].x, landmarks[self.UPPER_LIP_INNER].y])
            p_lo = np.array([landmarks[self.LOWER_LIP_INNER].x, landmarks[self.LOWER_LIP_INNER].y])
            p_lc = np.array([landmarks[self.MOUTH_LEFT_CORNER].x, landmarks[self.MOUTH_LEFT_CORNER].y])
            p_rc = np.array([landmarks[self.MOUTH_RIGHT_CORNER].x, landmarks[self.MOUTH_RIGHT_CORNER].y])
            A = np.linalg.norm(p_up - p_lo)
            C = np.linalg.norm(p_lc - p_rc)
            if C == 0: return 0.0
            return A / C
        except Exception:
            return 0.0

    def calculate_gaze(self, landmarks):
        return "Center"

    def analyze_eyebrows(self, landmarks, h, w):
        return None

    def analyze_head_pose(self, landmarks):
        return {"yaw": 0, "pitch": 0}

    def process_frame(self, frame):
        self.frame_counter += 1
        current_time = time.time()
        dt = current_time - self.last_process_time
        self.last_process_time = current_time
        
        if self.cooldown_timer > 0: self.cooldown_timer -= dt
        if self.lip_release_cooldown > 0: self.lip_release_cooldown -= dt

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.face_mesh.detect(mp_image)
        
        analysis = {
            "stress_index": 0.0,
            "micro_expressions": [],
            "gaze": "Center",
            "blink_rate": 0.0,
            "flags": [],
            "lip_compression": False,
            "lip_compression_score": 0.0,
            "blink_stress_level": "Normal",
            "behavioral_stress_index": 0.0,
            "ear": 0.0,
            "blink_threshold": 0.0,
            "blink_detected": False,
            "ear_history": list(self.ear_history),
            "mar": 0.0,
            "baseline_mar": float(self.baseline_mar) if self.baseline_mar else 0.0
        }
        
        if detection_result.face_landmarks:
            if self.frame_counter % 30 == 0:
                print("DEBUG: Face detected by MediaPipe", flush=True)
            landmarks = detection_result.face_landmarks[0]
            h, w, _ = frame.shape
            
            # --- EAR & Blinks ---
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
            raw_ear = (left_ear + right_ear) / 2.0
            
            self.ear_history.append(raw_ear)
            median_ear = np.median(self.ear_history)
            mean_ear = sum(self.ear_history) / len(self.ear_history)
            smoothed_ear = 0.5 * median_ear + 0.5 * mean_ear
            analysis["ear"] = smoothed_ear
            
            if self.blink_threshold is None:
                if self.calibration_start_time is None:
                    self.calibration_start_time = current_time
                    analysis["flags"].append("Calibrating Blinks...")
                self.calibration_buffer.append(smoothed_ear)
                if current_time - self.calibration_start_time >= 3.0:
                    if self.calibration_buffer:
                        self.baseline_ear = np.percentile(self.calibration_buffer, 90)
                        self.blink_threshold = self.baseline_ear * 0.75
                        print(f"Calibrated EAR Threshold: {self.blink_threshold:.4f} (Baseline: {self.baseline_ear:.4f})")
                    else:
                         self.blink_threshold = 0.20
            
            analysis["blink_threshold"] = self.blink_threshold if self.blink_threshold else 0.0
            
            if self.blink_threshold:
                if self.frame_counter % 10 == 0:
                    print(f"DEBUG: EAR={smoothed_ear:.3f}, Threshold={self.blink_threshold:.3f}", flush=True)
                
                speaking_detected = False
                if self.baseline_mar:
                    if len(self.mar_history) > 0:
                        if self.mar_history[-1] > self.baseline_mar * 1.2:
                            speaking_detected = True
                
                if smoothed_ear < self.blink_threshold:
                    self.consecutive_closed_frames += 1
                    if self.consecutive_closed_frames >= 3 and self.blink_state != "CLOSED":
                        self.blink_state = "CLOSED"
                        self.blink_start_time = current_time
                else:
                    if self.blink_state == "CLOSED":
                        blink_duration = current_time - (self.blink_start_time or current_time)
                        time_since_last = current_time - self.last_blink_time
                        hp = self.analyze_head_pose(landmarks)
                        pose_invalid = abs(hp.get("yaw", 0)) > self.MAX_YAW or abs(hp.get("pitch", 0)) > self.MAX_PITCH
                        if (self.MIN_BLINK_DURATION <= blink_duration <= self.MAX_BLINK_DURATION and
                            time_since_last >= self.MIN_BLINK_INTERVAL and
                            self.cooldown_timer <= 0 and
                            not speaking_detected and
                            not pose_invalid):
                            self.blink_count += 1
                            self.last_blink_time = current_time
                            self.blink_timestamps.append(current_time)
                            self.cooldown_timer = 0.3
                            analysis["blink_detected"] = True
                            analysis["micro_expressions"].append("Blink")
                            print(f"Blink event detected at {current_time:.3f}")
                    self.blink_state = "OPEN"
                    self.consecutive_closed_frames = 0
                    self.blink_start_time = None

            while self.blink_timestamps and current_time - self.blink_timestamps[0] > 60:
                self.blink_timestamps.popleft()
            
            blink_rate = len(self.blink_timestamps)
            analysis["blink_rate"] = blink_rate
            analysis["blink_count_last_60s"] = blink_rate
            analysis["blink_per_minute"] = float(blink_rate) if blink_rate > 0 else None
            
            if blink_rate > 40:
                analysis["blink_stress_level"] = "High Stress"
                analysis["blink_stress_score"] = 90
                analysis["flags"].append("High Blink Rate (Stress)")
            elif 24 <= blink_rate <= 40:
                analysis["blink_stress_level"] = "Stress"
                analysis["blink_stress_score"] = 60
            else:
                analysis["blink_stress_level"] = "Normal"
                analysis["blink_stress_score"] = 10
            
            # --- Lips ---
            mar = self.calculate_mar(landmarks, self.LIPS)
            analysis["mar"] = float(mar)
            up = landmarks[self.UPPER_LIP_INNER]
            lo = landmarks[self.LOWER_LIP_INNER]
            lip_vertical = abs(up.y - lo.y)
            lc = landmarks[self.MOUTH_LEFT_CORNER]
            rc = landmarks[self.MOUTH_RIGHT_CORNER]
            corner_dist = np.linalg.norm(np.array([lc.x, lc.y]) - np.array([rc.x, rc.y]))
            
            self.mar_history.append(mar)
            self.lip_distance_history.append(lip_vertical)
            self.corner_distance_history.append(corner_dist)
            
            try:
                pts = [landmarks[i] for i in self.LIPS]
                xs = [int(p.x * w) for p in pts]
                ys = [int(p.y * h) for p in pts]
                x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
                y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
                roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if y2 > y1 and x2 > x1 else None
                texture_var = 0.0
                if roi is not None and roi.size > 0:
                    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
                    mag = cv2.magnitude(gx, gy)
                    texture_var = float(np.var(mag) / (np.mean(mag) + 1e-6))
            except Exception:
                texture_var = 0.0
            
            if self.baseline_calibration_started is None:
                self.baseline_calibration_started = current_time
            self.lip_calibration_buffer.append((lip_vertical, mar, corner_dist, texture_var))
            if (self.baseline_lip_distance is None or self.baseline_corner_distance is None) and \
               (current_time - self.baseline_calibration_started >= self.baseline_calibration_secs):
                if self.lip_calibration_buffer:
                    lv = [x[0] for x in self.lip_calibration_buffer]
                    mr = [x[1] for x in self.lip_calibration_buffer]
                    cd = [x[2] for x in self.lip_calibration_buffer]
                    tv = [x[3] for x in self.lip_calibration_buffer]
                    self.baseline_lip_distance = float(np.median(lv))
                    self.baseline_mar = float(np.median(mr)) if mr else self.baseline_mar
                    self.baseline_corner_distance = float(np.median(cd))
                    self.baseline_texture_variance = float(np.median(tv)) if tv else 0.0
                    self.lip_calibration_buffer.clear()
            
            speaking_detected = False
            if len(self.lip_distance_history) >= 10:
                lv_arr = np.array(self.lip_distance_history)
                fluct = lv_arr.max() - lv_arr.min()
                speaking_detected = fluct > 0.03
            
            hp = self.analyze_head_pose(landmarks)
            pose_invalid = abs(hp.get("yaw", 0)) > self.MAX_YAW or abs(hp.get("pitch", 0)) > self.MAX_PITCH
            
            F1 = F2 = F3 = F4 = F5 = 0.0
            if self.baseline_lip_distance and self.baseline_mar and self.baseline_corner_distance:
                F1 = float(lip_vertical / (self.baseline_lip_distance + 1e-6))
                F2 = float(mar / (self.baseline_mar + 1e-6))
                F3 = float(corner_dist / (self.baseline_corner_distance + 1e-6))
            if self.baseline_texture_variance is not None:
                base_tv = max(self.baseline_texture_variance, 1e-6)
                F4 = float(max(0.0, min(1.0, (texture_var - base_tv) / (base_tv * 2.0))))
            if len(self.lip_distance_history) >= 20:
                diffs = np.diff(self.lip_distance_history)
                sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                small_amp = (np.max(self.lip_distance_history) - np.min(self.lip_distance_history)) < 0.02
                F5 = float(min(1.0, (sign_changes / 10.0))) if small_amp else 0.0
            
            compress_candidate = (F1 < 0.80 and F2 < 0.85 and F3 < 0.95)
            open_guard = (F1 > 1.05 or F2 > 1.05)
            normal_closed_guard = (F1 < 0.90 and F3 >= 0.98 and F4 < 0.20)
            
            if compress_candidate and not open_guard and not normal_closed_guard:
                w1, w2, w3, w4, w5 = 0.10, 0.10, 0.20, 0.35, 0.25
                raw_score = (w1 * (1.0 - min(1.0, F1)) + w2 * (1.0 - min(1.0, F2)) + w3 * (1.0 - min(1.0, F3)) + w4 * F4 + w5 * F5)
            else:
                raw_score = 0.0
            raw_score = float(max(0.0, min(1.0, raw_score)))
            self.lip_score_ema = self.lip_score_alpha * raw_score + (1.0 - self.lip_score_alpha) * self.lip_score_ema
            compression_score = float(max(0.0, min(100.0, self.lip_score_ema * 100.0)))
            analysis["lip_compression_score"] = compression_score
            
            ml_conditions = compression_score > 75.0
            if ml_conditions and not speaking_detected and not pose_invalid and compress_candidate and not open_guard and not normal_closed_guard:
                self.lip_compress_frames += 1
                if self.lip_compress_start_ts is None:
                    self.lip_compress_start_ts = current_time
            else:
                if self.lip_compress_frames > 0 and self.lip_release_cooldown <= 0:
                    self.lip_release_cooldown = 2.0
                if self.lip_release_cooldown <= 0:
                    self.lip_compress_frames = 0
                    self.lip_compress_start_ts = None
            
            lip_compression_active = (self.lip_compress_frames >= self.LIP_REQUIRED_FRAMES and 
                                      (self.lip_compress_start_ts is not None and (current_time - self.lip_compress_start_ts) >= 0.3))
            if mar > 0.4: lip_compression_active = False
            
            lip_bite = False
            lip_fold = False
            lip_tremble = False
            if self.baseline_mar and self.baseline_corner_distance:
                if F2 < 0.70 and F3 < 0.92: lip_fold = True
            if self.baseline_lip_distance and lip_vertical < self.baseline_lip_distance * 0.4:
                if up.y >= lo.y - 0.002: lip_bite = True
            lip_tremble = F5 > 0.5
            
            lip_stress_score = 0.0
            if lip_compression_active or lip_bite or lip_fold:
                lip_stress_score = compression_score if lip_compression_active else max(compression_score, 65.0)
                analysis["lip_compression"] = True
                analysis["micro_expressions"].append("Lip Compression (Stress)")
                if lip_tremble: analysis["flags"].append("Lip Tremble (High Stress)")
                elif lip_bite: analysis["flags"].append("Lip Biting (High Stress)")
                elif lip_fold: analysis["flags"].append("Lip Folding (Stress)")
                else: analysis["flags"].append("Lip Compression (Stress)")
            analysis["lip_compression_score"] = float(lip_stress_score)
            
            if detection_result.face_blendshapes:
                blendshapes = detection_result.face_blendshapes[0]
                bs_map = {cat.category_name: cat.score for cat in blendshapes}
                scores = {
                    "Happy": (bs_map.get("mouthSmileLeft", 0) + bs_map.get("mouthSmileRight", 0)) / 2.0,
                    "Sad": (bs_map.get("mouthFrownLeft", 0) + bs_map.get("mouthFrownRight", 0) + bs_map.get("browInnerUp", 0)) / 3.0,
                    "Angry": (bs_map.get("browDownLeft", 0) + bs_map.get("browDownRight", 0)) / 2.0,
                    "Surprise": (bs_map.get("browOuterUpLeft", 0) + bs_map.get("browOuterUpRight", 0) + bs_map.get("jawOpen", 0)) / 3.0,
                    "Fear": (bs_map.get("mouthStretch", 0) + bs_map.get("browInnerUp", 0)) / 2.0
                }
                best_emotion = "Neutral"
                best_score = 0.3
                for emo, score in scores.items():
                    if score > best_score:
                        best_score = score
                        best_emotion = emo
                analysis["emotion"] = best_emotion
                analysis["emotion_score"] = float(best_score)
                if self.frame_counter % 60 == 0:
                     print(f"Blendshape Emotion: {best_emotion} ({best_score:.3f})")

        else:
            if self.frame_counter % 60 == 0:
                print("DEBUG: MediaPipe FaceLandmarker found NO faces", flush=True)
            return analysis
        
        return analysis
