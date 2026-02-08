import os
import cv2
import numpy as np
import onnxruntime as ort

class QualcommLandmarkInferencer:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        outs = self.session.get_outputs()
        self.out_names = [o.name for o in outs]
        ins = self.session.get_inputs()
        self.in_name = ins[0].name
        self.input_size = (192, 192)
        self.LEFT_EYE = [33, 133, 159, 145, 160, 144]
        self.RIGHT_EYE = [263, 362, 386, 374, 385, 380]
        self.LIP_INNER_UP = 13
        self.LIP_INNER_LO = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.LIP_SET = [13, 14, 61, 291, 78, 308, 95, 324, 80, 310]

    def _preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img

    def _parse_output(self, outputs, orig_w, orig_h):
        arr = None
        # Try to find 'landmarks' output
        if "landmarks" in self.out_names:
            idx = self.out_names.index("landmarks")
            arr = np.array(outputs[idx])
        else:
            # Heuristic search
            for v in outputs:
                a = np.array(v)
                # Check for (1, 468, 3) or flattened versions
                if a.size == 1404 or a.size == 936:
                    arr = a
                    break
        
        if arr is None:
             # Fallback to last output if it looks reasonable? 
             # Or first output if it's large enough?
             # scores is (1,), landmarks is (1, 468, 3).
             for v in outputs:
                 a = np.array(v)
                 if a.size > 100: # Arbitrary threshold to distinguish from score
                     arr = a
                     break

        if arr is None:
             raise RuntimeError("Could not find landmarks in ONNX output")

        # Reshape to (468, 2) or (468, 3)
        if arr.ndim == 3 and arr.shape[1] == 468: # (1, 468, 3)
             coords = arr[0, :, :2] # Take first batch, x,y
        elif arr.ndim == 2 and arr.shape[1] == 1404: # (1, 1404)
             coords = arr.reshape(468, 3)[:, :2]
        elif arr.ndim == 2 and arr.shape[1] == 936: # (1, 936)
             coords = arr.reshape(468, 2)
        else:
             # Flatten and reshape
             coords = arr.reshape(-1, 3)[:, :2] if arr.size % 3 == 0 else arr.reshape(-1, 2)
        
        # In MediaPipe ONNX, output is usually normalized 0-1 (or relative to input size 192).
        # Since we observed values ~0.5 for random input, it's likely normalized.
        # But some versions divide by input size (192).
        # Let's assume normalized 0-1 for now as per test results (0.5 range).
        # If the values were 0-192, we would see values around 100.
        
        # Test results showed max 0.85, min -0.14. This fits 0-1 range (with some out of bounds).
        
        xs = coords[:, 0]
        ys = coords[:, 1]
        
        # Un-normalize to original image size
        pts = np.stack([xs * orig_w, ys * orig_h], axis=1)
        return pts

    def infer_landmarks(self, frame):
        h, w = frame.shape[:2]
        x = self._preprocess(frame)
        outputs = self.session.run(None, {self.in_name: x})
        pts = self._parse_output(outputs, w, h)
        return pts

    def extract_sets(self, frame):
        pts = self.infer_landmarks(frame)
        def take(idx):
            return [(float(pts[i][0]), float(pts[i][1])) for i in idx]
        lip = take(self.LIP_SET)
        left_eye = take(self.LEFT_EYE)
        right_eye = take(self.RIGHT_EYE)
        return {"lip": lip, "left_eye": left_eye, "right_eye": right_eye, "all": pts}
