from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from services.video_stream import VideoStream
from services.emotion_engine import EmotionEngine
import asyncio
import json
from datetime import datetime
from collections import deque
import base64
import numpy as np
import cv2
from detection.onnx_landmarks import QualcommLandmarkInferencer
from services.onnx_landmark_engine import OnnxLandmarkEngine
from utils.model_downloader import ensure_facial_landmark_model
from detection.facemesh import FaceMeshLandmarkExtractor
from detection.blink_detector import BlinkDetector
from detection.lip_lstm_inference import LipLSTMInference
from pydantic import BaseModel
from utils.data_logger import DataLogger
RELOAD_MARK = True

router = APIRouter()

from services.state_manager import manager as ai_state_manager

emotion_engine = None
video_stream = None
onnx_engine = None
logger = None
facemesh = None
blink = None
lip_infer = None
lip_window = deque(maxlen=30)
last_lip_state = "lip_calm"
last_conf = 0.0
frame_counter = 0
last_infer_ts = 0.0

def init_services():
    global emotion_engine, video_stream, onnx_engine
    print("DEBUG: init_services called", flush=True)
    try:
        if emotion_engine is None:
            print("DEBUG: Initializing EmotionEngine...", flush=True)
            emotion_engine = EmotionEngine()
            print("DEBUG: EmotionEngine initialized", flush=True)
        if video_stream is None:
            print("DEBUG: Initializing VideoStream...", flush=True)
            video_stream = VideoStream(emotion_engine)
            video_stream.start()
            print("DEBUG: VideoStream started from init_services", flush=True)
        if onnx_engine is None:
            try:
                model_path = ensure_facial_landmark_model(
                    model_dir="c:\\Users\\Abhinav\\Desktop\\Projects\\Emotinal_Recognition\\backend\\models",
                    target_name="facial_landmark.onnx"
                )
                infer = QualcommLandmarkInferencer(model_path)
                onnx_engine = OnnxLandmarkEngine(infer)
            except Exception as e:
                print(f"ONNX init failed: {e}")
                
        # Fallback pipelines to ensure UI stays functional
        global facemesh, blink, lip_infer
        if facemesh is None:
            facemesh = FaceMeshLandmarkExtractor()
        if blink is None:
            blink = BlinkDetector()
        if lip_infer is None:
            lip_infer = LipLSTMInference()
        global logger
        if logger is None:
            try:
                logger = DataLogger()
            except Exception:
                logger = None
    except Exception as e:
        print(f"CRITICAL ERROR in init_services: {e}", flush=True)
        import traceback
        traceback.print_exc()

def shutdown_services():
    global video_stream
    if video_stream:
        video_stream.stop()

@router.get("/status")
async def get_status():
    return {
        "backendConnected": True,
        "modelLoaded": emotion_engine.model is not None,
        "videoStreamActive": video_stream.is_running
    }

@router.get("/status2")
async def get_status2():
    return {"ok": True}
def generate_frames():
    while True:
        frame_bytes = video_stream.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Avoid tight loop if no frame
            import time
            time.sleep(0.1)

@router.get("/video-stream")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

class FrameRequest(BaseModel):
    frame_data: str | None = None

@router.post("/predict_frame")
async def predict_frame(req: FrameRequest):
    # If the request is empty (just checking current state), return from StateManager
    if not req or not req.frame_data:
        state = ai_state_manager.get()
        return {
            "lip_state": state.get("lip_state", "lip_calm"),
            "blink_count": state.get("blink_count", 0),
            "blink_per_min": state.get("blink_per_minute", None),
            "confidence": state.get("emotion_score", 0.0)
        }
        
    # Only process if custom frame data is provided (e.g. upload)
    frame = None
    if req and req.frame_data:
        try:
            b = base64.b64decode(req.frame_data)
            arr = np.frombuffer(b, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            frame = None
            
    if frame is None:
        return {"lip_state": "lip_calm", "blink_count": 0, "blink_per_min": None, "confidence": 0.0}

    # Primary path: ONNX engine
    if onnx_engine is not None:
        try:
            res = onnx_engine.process_frame(frame, ts=datetime.now().timestamp())
            resp = {
                "lip_state": res.get("lip_state", "lip_calm"),
                "blink_count": int(res.get("blink_count", 0)),
                "blink_per_min": res.get("blink_per_minute", None),
                "blink_detected": bool(res.get("blink_detected", False)),
                "confidence": 1.0
            }
            return resp
        except Exception:
            pass
            
    # Fallback path: MediaPipe + LSTM
    lm = facemesh.extract_landmarks(frame)
    if not lm:
        return {"lip_state": "lip_calm", "blink_count": 0, "blink_per_min": None, "confidence": 0.0}
        
    # We can't easily update global blink/lip state from a random uploaded frame
    # without affecting the stream state, so we just do stateless prediction if possible
    # or just return basic info. For now, let's just return calm to avoid side effects.
    return {"lip_state": "lip_calm", "blink_count": 0, "blink_per_min": None, "confidence": 0.0}

@router.get("/get_blink_stats")
async def get_blink_stats():
    # Return from shared state manager which is updated by the video stream
    state = ai_state_manager.get()
    return {
        "blink_count": state.get("blink_count", 0),
        "blink_per_min": state.get("blink_per_minute", None)
    }

@router.get("/health")
async def health():
    return {"status": "healthy"}

@router.get("/emotion")
async def get_emotion():
    result = video_stream.get_latest_result()
    if result:
        emotion_data = result.get("emotion_data")
        micro_data = result.get("micro_data")
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "micro_expressions": micro_data
        }
        
        if emotion_data:
            response["emotion"] = emotion_data["emotion"]
            response["confidence"] = emotion_data["confidence"]
            response["faceDetected"] = True
            response["face_detected"] = True
            response["emotion_score"] = float(emotion_data["confidence"])
        else:
            response["emotion"] = None
            response["confidence"] = 0
            response["faceDetected"] = False
            response["face_detected"] = False
            response["emotion_score"] = 0.0
        
        if isinstance(micro_data, dict):
            response["blink_per_minute"] = micro_data.get("blink_per_minute", None)
            response["stress_index"] = micro_data.get("stress_index", 0.0)
            response["lip_state"] = "lip_compression" if micro_data.get("lip_compression") else "lip_calm"
            
        return response
    else:
        return {
            "emotion": None,
            "confidence": 0,
            "timestamp": datetime.now().isoformat(),
            "faceDetected": False,
            "face_detected": False,
            "emotion_score": 0.0,
            "blink_per_minute": None,
            "stress_index": 0.0,
            "lip_state": "lip_calm",
            "micro_expressions": None
        }

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            result = video_stream.get_latest_result()
            if result:
                emotion_data = result.get("emotion_data")
                micro_data = result.get("micro_data")
                
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "micro_expressions": micro_data,
                    "blink_count": (micro_data.get("blink_count_last_60s", 0) if isinstance(micro_data, dict) else 0)
                }
                
                if emotion_data:
                    data["emotion"] = emotion_data["emotion"]
                    data["confidence"] = emotion_data["confidence"]
                    data["faceDetected"] = True
                else:
                    data["emotion"] = None
                    data["confidence"] = 0
                    data["faceDetected"] = False
            else:
                data = {
                    "emotion": None,
                    "confidence": 0,
                    "timestamp": datetime.now().isoformat(),
                    "faceDetected": False,
                    "micro_expressions": None
                }
            
            await websocket.send_json(data)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@router.get("/debug/state")
async def debug_state():
    state = ai_state_manager.get()
    return JSONResponse(state)
