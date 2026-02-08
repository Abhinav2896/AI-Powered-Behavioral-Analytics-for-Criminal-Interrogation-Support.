import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.utils.model_downloader import ensure_facial_landmark_model


def main():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    target = ensure_facial_landmark_model(str(model_dir), target_name="facial_landmark.onnx")
    size = os.path.getsize(target) if os.path.exists(target) else 0
    print(f"Model ready at: {target} ({size} bytes)")


if __name__ == "__main__":
    main()
