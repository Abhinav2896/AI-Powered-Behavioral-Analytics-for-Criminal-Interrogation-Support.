
import onnxruntime as ort
import os
import numpy as np

model_path = r"C:\Users\Abhinav\Desktop\Projects\Emotinal_Recognition\backend\models\facial_landmark.onnx"

try:
    print(f"Loading model from {model_path}...")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    print("Inputs:")
    for i in sess.get_inputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        
    print("\nOutputs:")
    for o in sess.get_outputs():
        print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
        
except Exception as e:
    print(f"Error loading model: {e}")
