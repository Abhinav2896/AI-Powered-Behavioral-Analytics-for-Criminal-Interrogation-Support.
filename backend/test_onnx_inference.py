
import onnxruntime as ort
import os
import numpy as np
import cv2

model_path = r"C:\Users\Abhinav\Desktop\Projects\Emotinal_Recognition\backend\models\facial_landmark.onnx"

try:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    # Create dummy input
    # Shape: [1, 3, 192, 192]
    # Range: 0-1 float32
    input_data = np.random.rand(1, 3, 192, 192).astype(np.float32)
    
    input_name = sess.get_inputs()[0].name
    print(f"Running inference on {input_name}...")
    
    outputs = sess.run(None, {input_name: input_data})
    
    for i, o in enumerate(outputs):
        output_name = sess.get_outputs()[i].name
        print(f"Output: {output_name}, Shape: {o.shape}")
        print(f"Min: {o.min()}, Max: {o.max()}")
        if output_name == "landmarks":
            print("First 5 landmarks:")
            print(o[0, :5, :])

except Exception as e:
    print(f"Error: {e}")
