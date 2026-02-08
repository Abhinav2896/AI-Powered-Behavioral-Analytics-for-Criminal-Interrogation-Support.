# Sentinel: Emotional Recognition & Behavioral Analysis System - Architecture Documentation

## 1. High-Level Overview

Sentinel is a real-time behavioral analysis system designed to detect emotions, micro-expressions, and stress indicators using multi-modal AI inputs (Video & Audio). The system follows a **Client-Server Architecture**:

-   **Backend**: A Python-based FastAPI server that handles hardware access (Camera/Mic), runs heavy AI inference (TensorFlow, MediaPipe, PyTorch), and manages state.
-   **Frontend**: A React (Vite) web application that visualizes the data, renders the video feed, and presents real-time analytics to the user.

---

## 2. Technology Stack

### Backend
-   **Framework**: FastAPI (Async Python web server).
-   **Computer Vision**: OpenCV (Video capture, image processing), MediaPipe (Face Mesh, Landmarks).
-   **AI/ML Models**:
    -   **TensorFlow/Keras**: Facial Emotion Recognition (FER).
    -   **PyTorch/Transformers**: Voice Stress Analysis (Wav2Vec2).
    -   **MediaPipe Tasks**: High-precision face landmarks.
-   **Concurrency**: `threading` for non-blocking I/O (Camera/Mic), `asyncio` for API handling.

### Frontend
-   **Framework**: React 18 (TypeScript) + Vite.
-   **Styling**: Tailwind CSS (Utility-first styling).
-   **Animations**: Framer Motion.
-   **Communication**: WebSocket (Real-time data), HTTP (Control/Status), MJPEG (Video Stream).

---

## 3. Project Structure

```
Emotinal_Recognition/
├── backend/
│   ├── app.py                  # Application Entry Point & Middleware
│   ├── routes/
│   │   └── api_routes.py       # API Endpoints (HTTP & WebSocket)
│   ├── services/
│   │   ├── video_stream.py     # Main Video Loop & Threading
│   │   ├── emotion_engine.py   # Emotion Detection (TensorFlow)
│   │   ├── micro_expression_engine.py # Blink/Lip/Gaze Analysis (MediaPipe)
│   │   ├── state_manager.py    # Thread-safe Shared State
│   │   └── ai_pipeline.py      # Orchestrator (Optional)
│   ├── voice_detection/        # Isolated Voice Module
│   │   ├── stress_engine.py    # Voice AI Logic
│   │   └── mic_stream.py       # Audio Capture
│   └── models/                 # Pre-trained Model Weights (.h5, .onnx, .task)
├── Front-end/
│   ├── src/
│   │   ├── app/
│   │   │   ├── App.tsx         # Main UI Controller
│   │   │   ├── components/     # UI Modules (CameraFeed, Status, etc.)
│   │   └── main.tsx            # React Entry Point
└── requirements.txt            # Python Dependencies
```

---

## 4. Backend Architecture (Deep Dive)

The backend is built as a **Multi-Threaded Service** to ensure the API remains responsive while heavy AI processing occurs in the background.

### A. Entry Point & Lifecycle (`app.py`)
-   **Initialization**: When `python backend/app.py` runs, it triggers the `lifespan` context manager.
-   **Startup**: Calls `api_routes.init_services()`, which launches the `VideoStream` thread and loads AI models.
-   **Server**: Uses `uvicorn` to serve the API on port `8000`.
-   **CORS**: Configured to allow connections from the Frontend (running on a different port, usually `5173` or `5175`).

### B. The Core: Video Stream Service (`services/video_stream.py`)
This is the "heart" of the visual system. It runs in a dedicated **Daemon Thread** (`_update` loop) to prevent blocking the web server.

1.  **Capture**: Opens the webcam using `cv2.VideoCapture` (tries indices 0, 1, 2).
2.  **Preprocessing**: Checks for black frames, flips the image (mirror effect).
3.  **Inference Orchestration**:
    -   **Every Frame**: Runs `micro_engine.process_frame()` for high-frequency events like Blinks.
    -   **Every N Frames**: Runs `emotion_engine.detect_faces_and_emotions()` (Computationally heavy).
4.  **Data Fusion**: Combines raw AI outputs into a `CombinedResult`.
5.  **Overlay Drawing**: Draws bounding boxes, landmarks, and graphs directly onto the frame (`_draw_debug_overlay`).
6.  **State Update**: Pushes the latest metrics to the global `AIStateManager`.

### C. AI Engines
1.  **Emotion Engine (`emotion_engine.py`)**:
    -   Uses **Haar Cascade** for face detection (fast, robust).
    -   Extracts face ROI (Region of Interest), resizes to 48x48 grayscale.
    -   Feeds into a **TensorFlow CNN** model to predict 7 emotions (Angry, Happy, Neutral, etc.).
    -   Uses a history buffer to smooth predictions over time.

2.  **Micro-Expression Engine (`micro_expression_engine.py`)**:
    -   Uses **MediaPipe FaceLandmarker** to get 478 3D face landmarks.
    -   **Blink Detection**: Calculates Eye Aspect Ratio (EAR). If EAR drops below a calibrated threshold for specific duration, it counts a blink.
    -   **Lip Compression**: Analyzes geometric distances (Lip Height, Mouth Width) and Texture Variance (Sobel filters) to detect stress-induced lip biting/compression.

3.  **Voice Stress Engine (`voice_detection/stress_engine.py`)**:
    -   Runs in complete isolation.
    -   Captures audio chunks (3-5s).
    -   Uses **Wav2Vec2** (Hugging Face) to extract acoustic features.
    -   Computes Arousal, Dominance, and Valence to calculate a "Stress Score".

---

## 5. Frontend Architecture & Connectivity

The Frontend is a "Thin Client" that primarily visualizes data provided by the backend.

### A. Connection Mechanisms
1.  **Video Stream (MJPEG)**:
    -   **URL**: `http://localhost:8000/api/video-stream`
    -   **Method**: The backend yields a multipart stream (`multipart/x-mixed-replace`). The browser treats this as a continuous image update.
    -   **Implementation**: `CameraFeed.tsx` uses a standard `<img>` tag. To fix caching issues, a timestamp `?t=...` is appended.

2.  **Real-Time Data (WebSocket)**:
    -   **URL**: `ws://localhost:8000/api/ws`
    -   **Frequency**: Backend pushes updates ~10-30 times per second (throttled).
    -   **Payload**: JSON containing `emotion`, `confidence`, `micro_expressions`, `blink_rate`, `flags`.
    -   **Lifecycle**: Connection is established in `App.tsx` via `useEffect`. Auto-reconnects on failure.

3.  **Command/Status (HTTP REST)**:
    -   **Endpoints**: `/api/status`, `/api/get_blink_stats`.
    -   **Usage**: Used for initial health checks and periodic sync (polling) as a fallback.

### B. UI Lifecycle (`App.tsx`)
1.  **Mount**: Checks Backend Health via HTTP.
2.  **Connect**: Establishes WebSocket connection.
3.  **Render Loop**:
    -   Receives JSON packet.
    -   Updates React State (`currentEmotion`, `microData`).
    -   Triggers re-render of components (Stress Bar, Timeline, Flags).
4.  **Animation**: Uses Framer Motion to smooth transitions between values (e.g., stress bar filling up).

---

## 6. Process Lifecycle Summary

1.  **User starts Backend**:
    -   Models load into memory (RAM).
    -   Camera and Mic threads start capturing.
    -   Web Server (Uvicorn) starts listening.

2.  **User opens Frontend**:
    -   React App loads.
    -   Browser requests MJPEG stream (Video appears).
    -   WebSocket connects.

3.  **Active Session**:
    -   **Input**: Camera captures frame `F(t)`.
    -   **Process**: Backend calculates `Stress(t)`.
    -   **Output**: Backend draws overlay on `F(t)` and sends `Stress(t)` via WebSocket.
    -   **Display**: User sees annotated video and updated charts simultaneously.

4.  **Shutdown**:
    -   Frontend tab closed -> WebSocket disconnects.
    -   Backend stopped (Ctrl+C) -> `shutdown_services()` releases Camera/Mic resources.

---

## 7. Key Algorithms in Detail

### Blink Rate (Stress Indicator)
-   **Concept**: High blink rate (>40 bpm) correlates with stress.
-   **Math**:
    $$ EAR = \frac{||p2-p6|| + ||p3-p5||}{2 \times ||p1-p4||} $$
    *(Vertical eye distance divided by horizontal distance)*.
-   **Logic**:
    -   Compute EAR for every frame.
    -   Maintain a moving average (Smooth EAR).
    -   Dynamic Calibration: Capture first 3 seconds to find user's "resting" EAR. Set Threshold = 75% of resting EAR.
    -   Count transitions from Open -> Closed -> Open.

### Behavioral Stress Index (BSI)
-   **Formula**:
    $$ BSI = 0.4 \times BlinkScore + 0.3 \times LipScore + 0.3 \times EmotionScore $$
-   **Purpose**: Fuses multiple weak signals into one strong confidence metric.
-   **Normalization**: All inputs are normalized to 0-100 scale before fusion.

### Voice Stress (Acoustic)
-   **Pipeline**: Audio Raw -> Wav2Vec2 Feature Extractor -> Classification Head -> Logits -> Sigmoid.
-   **Scaling**: Raw model outputs are often clustered around 0.5. A custom Sigmoid scaling with high sensitivity ($k=10$) is applied to spread values to 0-100 range for better UI readability.
