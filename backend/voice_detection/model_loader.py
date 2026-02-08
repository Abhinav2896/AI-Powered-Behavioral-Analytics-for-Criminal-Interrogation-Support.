
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "wav2vec2_emotion")

def download_voice_model():
    """
    Downloads the model locally if it doesn't exist.
    """
    # Check if key files exist to ensure complete download
    config_path = os.path.join(MODEL_DIR, "config.json")
    model_path = os.path.join(MODEL_DIR, "model.safetensors")
    pytorch_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
    
    if not os.path.exists(MODEL_DIR) or not os.path.exists(config_path) or (not os.path.exists(model_path) and not os.path.exists(pytorch_path)):
        print(f"Downloading voice model {MODEL_NAME} to {MODEL_DIR}...")
        if os.path.exists(MODEL_DIR):
             # Cleanup potential partial download
             import shutil
             try:
                 shutil.rmtree(MODEL_DIR)
             except:
                 pass
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        try:
            processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
            
            processor.save_pretrained(MODEL_DIR)
            model.save_pretrained(MODEL_DIR)
            print("Voice model downloaded successfully.")
        except Exception as e:
            print(f"Failed to download voice model: {e}")
            raise e
    else:
        print(f"Voice model already exists at {MODEL_DIR}")

def load_voice_model():
    """
    Loads the model from the local directory.
    """
    print(f"Loading voice model from {MODEL_DIR}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        return processor, model
    except Exception as e:
        print(f"Error loading voice model: {e}")
        return None, None
