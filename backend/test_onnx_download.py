
import os
import sys

# Add the current directory to path so we can import from backend
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from utils.model_downloader import ensure_facial_landmark_model

model_dir = os.path.join(os.getcwd(), 'backend', 'models')
print(f"Downloading model to {model_dir}...")

try:
    path = ensure_facial_landmark_model(model_dir)
    print(f"Model downloaded to: {path}")
    if os.path.exists(path):
        print(f"File size: {os.path.getsize(path)} bytes")
except Exception as e:
    print(f"Error: {e}")
