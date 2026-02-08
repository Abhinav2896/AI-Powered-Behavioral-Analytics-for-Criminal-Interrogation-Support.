import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
import os

class EmotionEngine:
    def __init__(self, model_path="backend/model_file.h5", cascade_path="backend/haarcascade_frontalface_default.xml"):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = None
        self.face_cascade = None
        self.tracked_faces = [] # List of (x, y, w, h) for multiple face tracking
        self.prediction_history = [] # Rolling history of prediction vectors for smoothing
        self.history_length = 5 # Number of frames to smooth over (stabilizes UI)
        self.model_path = model_path
        self.cascade_path = cascade_path
        self.load_resources()

    def load_resources(self):
        # 1. Load Model
        try:
            # Adjust paths if running from a different directory
            if not os.path.exists(self.model_path):
                 # Try relative path if absolute failed or verify path
                 if os.path.exists(os.path.join("backend", "model_file.h5")):
                     self.model_path = os.path.join("backend", "model_file.h5")
                 elif os.path.exists("model_file.h5"):
                     self.model_path = "model_file.h5"
            
            print(f"Loading model from {self.model_path}...")
            
            # Define Patches and Custom Objects for Keras Compatibility
            class PatchedInputLayer(InputLayer):
                def __init__(self, batch_shape=None, **kwargs):
                    if batch_shape is not None:
                        kwargs['batch_input_shape'] = batch_shape
                    super().__init__(**kwargs)

            class DTypePolicy:
                def __init__(self, config=None, **kwargs):
                    self.name = "float32"
                    if config and isinstance(config, dict) and 'name' in config:
                        self.name = config['name']
                    self.compute_dtype = self.name
                    self.variable_dtype = self.name
                @classmethod
                def from_config(cls, config):
                    return cls(config)
                def get_config(self):
                    return {"name": self.name}

            custom_objects = {
                'InputLayer': PatchedInputLayer,
                'DTypePolicy': DTypePolicy
            }

            try:
                self.model = load_model(self.model_path, compile=False, custom_objects=custom_objects)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading emotion model with custom objects: {e}")
                # Fallback to standard load if the above fails (unlikely given the patch)
                self.model = load_model(self.model_path, compile=False)
                print("Model loaded successfully (fallback).")

        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model = None


        # 2. Load Cascade
        try:
            if not os.path.exists(self.cascade_path):
                 if os.path.exists(os.path.join("backend", "haarcascade_frontalface_default.xml")):
                     self.cascade_path = os.path.join("backend", "haarcascade_frontalface_default.xml")
                 elif os.path.exists("haarcascade_frontalface_default.xml"):
                     self.cascade_path = "haarcascade_frontalface_default.xml"

            print(f"Loading face cascade from {self.cascade_path}...")
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.face_cascade.empty():
                raise IOError("Failed to load Haarcascade XML file")
            print("Face cascade loaded successfully.")
        except Exception as e:
            print(f"Error loading face cascade: {e}")
            self.face_cascade = None

    def detect_emotion(self, frame):
        """
        Legacy wrapper for backward compatibility.
        Performs detection and drawing in one step.
        """
        results = self.detect_faces_and_emotions(frame)
        annotated_frame = self.draw_results(frame, results)
        return results, annotated_frame

    def detect_faces_and_emotions(self, frame):
        if self.face_cascade is None:
            # Cannot detect faces without cascade
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimized Face Detection Parameters
        # scaleFactor: 1.2 (Faster than 1.1, still accurate enough)
        # minNeighbors: 7 (Strict filtering to remove false positives/accessories)
        # minSize: (60, 60) (Ignore small/distant faces or noise)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        # DEBUG: Log face count
        if len(faces) > 0:
            print(f"DEBUG: EmotionEngine detected {len(faces)} faces", flush=True)
        else:
            pass
            # print("DEBUG: No faces detected by Haar", flush=True)

        results = []
        next_tracked_faces = []
        
        if len(faces) == 0:
            self.tracked_faces = [] # Reset tracking if no faces found
            return results

        # Multi-Face Tracking & Smoothing
        # Match current detected faces to the closest previously tracked faces
        used_track_indices = set()
        
        for (x, y, w, h) in faces:
            # Find closest tracked face
            best_match_idx = -1
            min_dist = float('inf')
            
            for i, (tx, ty, tw, th) in enumerate(self.tracked_faces):
                if i in used_track_indices:
                    continue
                    
                # Calculate center distance
                current_center = (x + w/2, y + h/2)
                tracked_center = (tx + tw/2, ty + th/2)
                dist = np.sqrt((current_center[0] - tracked_center[0])**2 + 
                               (current_center[1] - tracked_center[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            # If match found within reasonable threshold (half width), smooth it
            if best_match_idx != -1 and min_dist < w * 0.5:
                tx, ty, tw, th = self.tracked_faces[best_match_idx]
                used_track_indices.add(best_match_idx)
                
                alpha = 0.8 # Higher alpha = less smoothing, more responsive (snappy)
                x = int(tx * (1 - alpha) + x * alpha)
                y = int(ty * (1 - alpha) + y * alpha)
                w = int(tw * (1 - alpha) + w * alpha)
                h = int(th * (1 - alpha) + h * alpha)
            
            next_tracked_faces.append((x, y, w, h))

            # Process this face
            roi_gray = gray[y:y+h, x:x+w]
            try:
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = np.array(roi)
                    roi = np.expand_dims(roi, axis=-1) # (48, 48, 1)
                    roi = np.expand_dims(roi, axis=0)  # (1, 48, 48, 1)

                    if self.model is not None:
                        raw_prediction = self.model.predict(roi, verbose=0)[0]
                        
                        # Temporal Smoothing: Average predictions over history
                        if len(self.prediction_history) >= self.history_length:
                            self.prediction_history.pop(0)
                        self.prediction_history.append(raw_prediction)
                        
                        # Calculate average prediction vector
                        avg_prediction = np.mean(self.prediction_history, axis=0)
                        
                        label = self.emotion_labels[avg_prediction.argmax()]
                        confidence = float(avg_prediction.max())
                    else:
                        label = "Unknown"
                        confidence = 0.0

                    results.append({
                        "emotion": label,
                        "confidence": confidence,
                        "box": (x, y, w, h)
                    })
            except Exception as e:
                print(f"Error in inference: {e}")
        
        # Sort results by face area (descending) so the largest face is first
        results.sort(key=lambda res: res["box"][2] * res["box"][3], reverse=True)

        # Update tracked faces for next frame
        self.tracked_faces = next_tracked_faces
        return results

    def draw_results(self, frame, results):
        """
        Draws bounding boxes and labels on the frame with a clean, uncluttered style.
        """
        annotated_frame = frame.copy()
        
        for res in results:
            (x, y, w, h) = res["box"]
            # Clean Visual Style
            color = (0, 255, 255) # Yellow
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, thickness)
            
        return annotated_frame
