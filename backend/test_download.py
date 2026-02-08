
from huggingface_hub import list_repo_files, hf_hub_download
import os

repo = "qualcomm/Facial-Landmark-Detection"
try:
    print(f"Listing files for {repo}...")
    files = list_repo_files(repo)
    print("Files found:", files)
except Exception as e:
    print(f"Failed to list files: {e}")

try:
    print(f"Attempting to download Facial-Landmark-Detection.onnx...")
    path = hf_hub_download(repo_id=repo, filename="Facial-Landmark-Detection.onnx")
    print(f"Download successful: {path}")
except Exception as e:
    print(f"Failed to download: {e}")
