
import threading
import time
import torch
import numpy as np
from .model_loader import load_voice_model, download_voice_model
from .mic_stream import MicStream

class VoiceStressEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VoiceStressEngine, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.processor = None
        self.model = None
        self.mic_stream = None
        self.is_loading = False
        self.model_loaded = False
        self.processing_thread = None
        self.running = False
        
        # Latest result
        self.latest_result = {
            "model_loaded": False,
            # CONSTRAINT: English-only language support. 
            # The current model (wav2vec2-emotion) is fine-tuned on English datasets (IEMOCAP, MSP-Improv).
            # Non-English audio may yield undefined stress scores.
            "language": "en", 
            "arousal": 0.0,
            "dominance": 0.0,
            "valence": 0.0,
            "stress_level": "CALM",
            "stress_score": 0.0
        }
        
        self._initialized = True

    def initialize(self):
        """
        Starts model download/load in a separate thread.
        """
        if self.model_loaded or self.is_loading:
            return

        self.is_loading = True
        threading.Thread(target=self._load_resources, daemon=True).start()

    def _load_resources(self):
        print("Initializing Voice Stress Engine...")
        try:
            download_voice_model()
            self.processor, self.model = load_voice_model()
            
            if self.processor and self.model:
                self.model_loaded = True
                print("Voice Stress Engine loaded successfully.")
                
                # Start mic stream
                self.mic_stream = MicStream(rate=16000, chunk_duration=3)
                self.mic_stream.start()
                
                # Start processing loop
                self.running = True
                self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
                self.processing_thread.start()
            else:
                print("Failed to load Voice Stress Engine models.")
        except Exception as e:
            print(f"Error initializing Voice Stress Engine: {e}")
        finally:
            self.is_loading = False

    def _process_loop(self):
        print("Voice processing loop started.")
        while self.running:
            if not self.mic_stream:
                time.sleep(1)
                continue
                
            audio_chunk = self.mic_stream.get_audio()
            if audio_chunk is not None:
                self._analyze_chunk(audio_chunk)
            else:
                time.sleep(0.1)

    def _analyze_chunk(self, audio_data):
        if not self.model or not self.processor:
            return

        try:
            # Normalize volume if too quiet, skip processing to avoid noise amplification
            max_amp = np.max(np.abs(audio_data))
            print(f"DEBUG: Audio chunk max amplitude: {max_amp:.4f}")
            
            if max_amp < 0.001:
                # Silence
                print("DEBUG: Audio too quiet, skipping.")
                return

            # Normalize audio to -1..1 range for the model
            # This ensures the model receives "normal" volume levels even if mic gain is low
            if max_amp > 0:
                audio_data = audio_data / max_amp * 0.9

            # Process with transformers
            inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Model outputs: arousal, dominance, valence
            # The model 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim' returns regression values
            # Output order: arousal, dominance, valence
            
            scores = logits[0].numpy()
            print(f"DEBUG: Raw model scores: {scores}")
            
            # Apply sensitivity scaling and Sigmoid activation
            # The raw logits seem small or centered around 0. We boost them to drive the sigmoid.
            sensitivity = 10.0 
            
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            # Map raw scores to 0-1 range using sigmoid
            # We assume the model outputs are logits where >0 is high and <0 is low
            arousal_c = sigmoid(scores[0] * sensitivity)
            dominance_c = sigmoid(scores[1] * sensitivity)
            valence_c = sigmoid(scores[2] * sensitivity)
            
            # Logic from requirements:
            # If arousal > 0.7 and dominance < 0.4: stress_level = "HIGH"
            # Else: stress_level = "CALM"
            # stress_score = 0.6 * arousal + 0.4 * (1 - dominance)
            
            stress_score = 0.6 * arousal_c + 0.4 * (1.0 - dominance_c)
            
            # Threshold check
            if arousal_c > 0.6 and dominance_c < 0.5:
                stress_level = "HIGH"
            elif stress_score > 0.6:
                stress_level = "HIGH"
            else:
                stress_level = "CALM"

            # Update state
            self.latest_result = {
                "model_loaded": True,
                "language": "en",
                "arousal": round(arousal_c, 2),
                "dominance": round(dominance_c, 2),
                "valence": round(valence_c, 2),
                "stress_level": stress_level,
                "stress_score": round(stress_score, 2)
            }
            
            # Debug log
            print(f"Voice Analysis: A={arousal_c:.2f}, D={dominance_c:.2f}, V={valence_c:.2f} -> {stress_level} ({stress_score:.2f})")
            
        except Exception as e:
            print(f"Voice inference error: {e}")

    def get_status(self):
        return self.latest_result

    def stop(self):
        self.running = False
        if self.mic_stream:
            self.mic_stream.stop()
