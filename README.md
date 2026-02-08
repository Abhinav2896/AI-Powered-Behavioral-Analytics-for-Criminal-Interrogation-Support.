# Sentinel: AI-Powered Behavioral Analytics for Criminal Interrogation Support

This repository contains the solution for an advanced **AI-Powered Behavioral Analytics System** designed to support criminal interrogation.
We developed an end-to-end multi-modal deep learning pipeline to enable real-time detection of emotions, micro-expressions, and stress indicators using video and audio inputs.

## 🚀 Project Overview

The core of this project is a **Real-Time Behavioral Analysis System** that fuses data from computer vision and audio processing to provide actionable insights. This system enables investigators to:

✅ **Detect Emotions:** Real-time facial emotion recognition (Angry, Happy, Neutral, etc.).
❌ **Identify Micro-Expressions:** High-precision tracking of blinks, lip compression, and gaze shifts.
🧠 **Analyze Stress Levels:** Compute a "Behavioral Stress Index" (BSI) based on physiological markers.
🗣️ **Monitor Voice Stress:** Analyze acoustic features (Arousal, Dominance, Valence) to detect vocal stress.

### Key Deliverables

*   **Behavioral Analysis Engine:** Multi-threaded Python backend integrating TensorFlow, PyTorch, and MediaPipe.
*   **Real-Time Dashboard:** React-based frontend for visualizing video feeds, stress graphs, and alerts.
*   **Voice Stress Analyzer:** Isolated module using Wav2Vec2 for acoustic feature extraction.
*   **Micro-Expression Tracker:** Algorithms for calculating Eye Aspect Ratio (EAR) and lip geometry changes.

## 🛠️ Technical Implementation

### 1. System Architecture
*   **Backend:** FastAPI (Async Python web server) handling hardware access and AI inference.
*   **Frontend:** React 18 (Vite + TypeScript) for visualization and control.
*   **Communication:** WebSocket (Real-time data), MJPEG (Video Stream), HTTP (Status).

### 2. AI & Processing Pipelines
*   **Emotion Engine:**
    *   **Model:** TensorFlow/Keras CNN.
    *   **Input:** 48x48 Grayscale Face ROI.
    *   **Function:** Predicts 7 core emotions with temporal smoothing.
*   **Micro-Expression Engine:**
    *   **Model:** MediaPipe Face Mesh (478 3D landmarks).
    *   **Logic:** Calculates Blink Rate (EAR) and Lip Compression (Geometry + Texture Variance).
*   **Voice Stress Engine:**
    *   **Model:** Wav2Vec2 (Hugging Face Transformers).
    *   **Function:** Extracts acoustic features from 3-5s audio chunks to compute stress scores.

### 3. Data Fusion & Metrics
*   **Behavioral Stress Index (BSI):** A composite score derived from weighted signals:
    $$ BSI = 0.4 \times BlinkScore + 0.3 \times LipScore + 0.3 \times EmotionScore $$
*   **Blink Rate Analysis:** Dynamic calibration to user's resting blink rate to detect stress-induced rapid blinking.

## 📂 Repository Structure

```
AI_Powered_Behavioral_Analytics/
├── backend/
│   ├── app.py                 # FastAPI Entry Point & Middleware
│   ├── routes/                # API Endpoints (HTTP & WebSocket)
│   ├── services/
│   │   ├── video_stream.py    # Main Video Loop & Threading
│   │   ├── emotion_engine.py  # Emotion Detection Logic
│   │   ├── ai_pipeline.py     # Orchestrator
│   │   └── state_manager.py   # Thread-safe Shared State
│   ├── voice_detection/       # Voice Stress Analysis Module
│   ├── detection/             # Face Mesh & Landmark Logic
│   └── models/                # Pre-trained Model Weights (.h5, .onnx)
├── Front-end/
│   ├── src/
│   │   ├── app/               # Main UI Components
│   │   └── main.tsx           # React Entry Point
│   ├── guidelines/            # UX/UI Guidelines
│   └── vite.config.ts         # Build Configuration
├── old model/                 # Legacy reports and scripts
├── ARCHITECTURE.md            # Detailed System Architecture
└── requirements.txt           # Python Dependencies
```

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   Node.js & npm
*   Webcam & Microphone

### 1. Setup Backend
```bash
# Clone the repository
git clone https://github.com/Abhinav2896/AI-Powered-Behavioral-Analytics-for-Criminal-Interrogation-Support..git
cd AI_Powered_Behavioral_Analytics-for-Criminal-Interrogation-Support

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

### 2. Setup Frontend
```bash
cd Front-end

# Install Node dependencies
npm install
```

### 3. Run the Application
**Start the Backend:**
```bash
# From the root directory
python backend/app.py
```
*The backend will start on `http://localhost:8000`*

**Start the Frontend:**
```bash
# From the Front-end directory
npm run dev
```
*Open the provided local URL (usually `http://localhost:5173`) in your browser.*

## 📊 Performance & Artifacts

### Key Metrics
*   **Inference Speed:** Real-time processing optimized with multi-threading.
*   **Latency:** Low-latency WebSocket communication for instant feedback.
*   **Accuracy:** Calibrated thresholds for blink detection and emotion recognition to minimize false positives.

### Trained Model Weights
The system uses pre-trained models stored in `backend/models/`. Ensure these files are present for the system to function correctly.

---
*This project was developed to provide advanced technological support for criminal interrogation and behavioral analysis.*
