
from huggingface_hub import list_repo_files
import os

repo = "qualcomm/MediaPipe-Face-Detection"
try:
    print(f"Listing files for {repo}...")
    files = list_repo_files(repo)
    print("Files found:", files)
except Exception as e:
    print(f"Failed to list files: {e}")
