
import sys
import os
import cv2
import numpy as np
sys.path.append(os.getcwd())

try:
    from services.micro_expression_engine import MicroExpressionEngine
    print("Import successful")
    
    engine = MicroExpressionEngine()
    print("Engine initialized")
    
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
    
    result = engine.process_frame(frame)
    print("Frame processed")
    print(f"Result keys: {list(result.keys())}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
