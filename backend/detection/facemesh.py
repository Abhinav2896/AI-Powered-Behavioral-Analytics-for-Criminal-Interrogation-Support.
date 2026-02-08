import cv2
import mediapipe as mp
import os
import numpy as np

class FaceMeshLandmarkExtractor:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Use absolute path to the model file
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/face_landmarker.task"))
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def extract_landmarks(self, frame):
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        lm = detection_result.face_landmarks[0]
        
        lip_idx = [13, 14, 61, 291, 78, 308, 95, 324, 80, 310]
        lip = [(lm[i].x, lm[i].y) for i in lip_idx]
        left_eye_idx = [33, 133, 159, 145, 160, 144]
        right_eye_idx = [263, 362, 386, 374, 385, 380]
        left_eye = [(lm[i].x, lm[i].y) for i in left_eye_idx]
        right_eye = [(lm[i].x, lm[i].y) for i in right_eye_idx]
        return {"lip": lip, "left_eye": left_eye, "right_eye": right_eye}
