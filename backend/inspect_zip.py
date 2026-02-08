
import zipfile
import os
from huggingface_hub import hf_hub_download

repo = "qualcomm/Facial-Landmark-Detection"
filename = "Facial-Landmark-Detection_float.onnx.zip"

try:
    path = hf_hub_download(repo_id=repo, filename=filename)
    print(f"Zip path: {path}")
    print(f"Zip size: {os.path.getsize(path)}")
    
    with zipfile.ZipFile(path, 'r') as zip_ref:
        print("\nFiles in zip:")
        for info in zip_ref.infolist():
            print(f"- {info.filename}: {info.file_size} bytes")
except Exception as e:
    print(f"Error: {e}")
