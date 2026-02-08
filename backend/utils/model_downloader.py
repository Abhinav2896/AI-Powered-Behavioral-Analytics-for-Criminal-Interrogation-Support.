import os
import shutil
import zipfile
from typing import Optional
from huggingface_hub import hf_hub_download, list_repo_files

def ensure_facial_landmark_model(model_dir: str, target_name: str = "facial_landmark.onnx") -> str:
    os.makedirs(model_dir, exist_ok=True)
    target_path = os.path.join(model_dir, target_name)
    
    # If file exists and is not empty, return it.
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        # Check if it's the usable model (not 3DMM parameter model which is ~25KB)
        # The 3DMM model is small, but let's assume if it exists it's fine for now,
        # unless we want to force re-download of the better model.
        # But for now, let's just return it.
        # Actually, if we switched repos, we might want to force update if the file is the old one.
        # But file size check is hard without knowing exact size.
        pass # return target_path # Commented out to force re-download for testing

    # List of repos to try, in order of preference
    # 1. MediaPipe-Face-Detection (likely returns landmarks directly)
    # 2. Facial-Landmark-Detection (returns 3DMM parameters, harder to use)
    repos = [
        "qualcomm/MediaPipe-Face-Detection",
        "qualcomm/Facial-Landmark-Detection"
    ]

    last_error = None

    for repo in repos:
        print(f"Searching for model in {repo}...")
        try:
            files = list_repo_files(repo)
        except Exception as e:
            print(f"Failed to list files from {repo}: {e}")
            continue

        # Priority 1: Zip file containing ONNX (specifically float version)
        chosen_file: Optional[str] = None
        is_zip = False
        
        # Look for FaceLandmarkDetector specifically in the MediaPipe repo
        if "MediaPipe" in repo:
             for f in files:
                if "FaceLandmarkDetector_float.onnx.zip" in f:
                    chosen_file = f
                    is_zip = True
                    break
        
        # Generic fallback
        if chosen_file is None:
             for f in files:
                if "float.onnx.zip" in f.lower():
                    chosen_file = f
                    is_zip = True
                    break
                    
        # Direct ONNX
        if chosen_file is None:
            for f in files:
                if f.lower().endswith(".onnx"):
                    chosen_file = f
                    break
        
        if chosen_file:
            try:
                print(f"Attempting to download {chosen_file} from {repo}...")
                downloaded_path = hf_hub_download(repo_id=repo, filename=chosen_file)
                
                if is_zip:
                    print(f"Extracting {chosen_file}...")
                    with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                        zip_ref.extractall(model_dir)
                        
                        found_onnx = None
                        found_data = None
                        
                        for root, dirs, files_in_dir in os.walk(model_dir):
                            for file in files_in_dir:
                                if file.lower().endswith(".onnx") and file != target_name:
                                    found_onnx = os.path.join(root, file)
                                elif file.lower().endswith(".data"):
                                    found_data = os.path.join(root, file)
                        
                        if found_onnx:
                            print(f"Found extracted ONNX at {found_onnx}")
                            if os.path.exists(target_path):
                                os.remove(target_path)
                            shutil.move(found_onnx, target_path)
                            
                            if found_data:
                                print(f"Found extracted data at {found_data}")
                                target_data_path = os.path.join(model_dir, os.path.basename(found_data))
                                if os.path.exists(target_data_path):
                                    os.remove(target_data_path)
                                shutil.move(found_data, target_data_path)
                            
                            # Cleanup
                            parent_dir = os.path.dirname(found_onnx)
                            if parent_dir != model_dir and os.path.isdir(parent_dir):
                                try:
                                    shutil.rmtree(parent_dir)
                                except:
                                    pass
                                    
                            print(f"Successfully set up model at {target_path}")
                            return target_path
                else:
                    shutil.copyfile(downloaded_path, target_path)
                    print(f"Successfully downloaded to {target_path}")
                    return target_path

            except Exception as e:
                print(f"Failed to download/extract {chosen_file}: {e}")
                last_error = e
                continue

    # If we get here, check if we already have a file and return it (fallback)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    raise RuntimeError(f"Failed to download Qualcomm Facial Landmark ONNX model: {last_error}")
