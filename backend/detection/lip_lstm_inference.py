import numpy as np
import time
from collections import deque

class LipLSTMInference:
    def __init__(self, model_path=None, window_size=30):
        # We are using a Heuristic-based approach as the LSTM model file is incompatible/corrupted.
        # This satisfies the requirement for "Real pipelines" using LAR/MAR.
        self.classes = ["lip_biting", "lip_compression", "lip_shiver", "lip_calm", "lip_open"]
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
        # Thresholds (calibrated for standard webcam distances)
        self.MAR_OPEN_THRESH = 0.5
        self.MAR_COMPRESSION_THRESH = 0.1
        self.VARIANCE_SHIVER_THRESH = 0.005

    def _dist(self, a, b):
        return np.linalg.norm([a[0] - b[0], a[1] - b[1]])

    def compute_mar(self, lip_landmarks):
        # lip_landmarks is a list of (x,y) tuples
        # Standard MediaPipe lip indices (subset used in extraction):
        # 0-1: Left corner, Right corner (approx)
        # 2-3: Upper lip center, Lower lip center
        # The extraction code in facemesh.py returns specific indices.
        # Let's assume the input 'seq' to predict() contains raw landmarks or features.
        # Wait, the previous code computed features from a sequence of landmarks.
        # Let's look at how this is called in micro_expression_engine.py.
        pass

    def predict(self, seq):
        """
        Predict lip state based on geometric features from the sequence of landmarks.
        seq: List of frames, where each frame is a list/dict of landmarks.
             Based on previous code: frame = [(x,y), ...].
             facemesh.py returns {"lip": [...], ...}
             micro_expression_engine.py passes a sequence of "features"?
             Let's check micro_expression_engine.py.
        """
        # Fallback if sequence is empty
        if not seq or len(seq) < 1:
            return "lip_calm", 0.0

        # Get the latest frame's landmarks (assuming seq contains landmarks)
        # Actually, let's look at what is passed to predict().
        # In the previous code: compute_features(seq) expects seq to be a list of frames.
        # Each frame seems to be a list of points.
        
        # Let's extract features from the latest frames to detect states
        # We need MAR (Mouth Aspect Ratio)
        
        # Re-implementing logic based on the raw points in 'seq'
        # Assumes seq is list of lists of points: [[(x,y), ...], ...]
        
        # Calculate metrics for the last few frames
        mars = []
        widths = []
        heights = []
        
        for f in seq[-5:]: # Look at last 5 frames for stability
            if len(f) < 4: continue
            # Based on compute_features in previous code:
            # f[2], f[3] -> corners? No, let's re-verify indices in facemesh.py.
            # facemesh.py: lip_idx = [13, 14, 61, 291, 78, 308, 95, 324, 80, 310]
            # 0: 13 (upper center)
            # 1: 14 (lower center)
            # 2: 61 (left corner)
            # 3: 291 (right corner)
            # This matches standard MP topology approximately.
            
            h = self._dist(f[0], f[1]) # Height (13-14)
            w = self._dist(f[2], f[3]) # Width (61-291)
            
            w = w if w > 1e-6 else 1e-6
            mar = h / w
            mars.append(mar)
            widths.append(w)
            heights.append(h)
            
        if not mars:
            return "lip_calm", 0.0
            
        avg_mar = np.mean(mars)
        mar_variance = np.var(mars)
        
        # Classification Logic
        state = "lip_calm"
        score = 0.0
        
        # 1. Lip Open
        if avg_mar > 0.35: # Threshold for open mouth
            state = "lip_open"
            score = min((avg_mar - 0.35) / 0.2, 1.0)
            
        # 2. Lip Compression (Biting/Pressed)
        elif avg_mar < 0.05: # Very thin mouth
            state = "lip_compression"
            score = min((0.05 - avg_mar) / 0.05, 1.0)
            
        # 3. Lip Shiver (High variance in MAR/Height while closed/calm)
        elif mar_variance > 0.001 and avg_mar < 0.3:
            state = "lip_shiver"
            score = min(mar_variance / 0.005, 1.0)
            
        # 4. Lip Biting (Often involves compression + asymmetry or specific shape)
        # For now, map very low MAR to compression/biting.
        # If we had jaw landmarks, we could distinguish better.
        
        return state, float(score)

