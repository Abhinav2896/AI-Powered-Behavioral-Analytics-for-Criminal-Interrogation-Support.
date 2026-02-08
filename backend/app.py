from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Fix for module imports when running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes import api_routes
from voice_detection import voice_router
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up Emotion Recognition Backend...")
    api_routes.init_services()
    for r in app.router.routes:
        try:
            print("ROUTE", r.path)
        except Exception:
            pass
    yield
    # Shutdown
    print("Shutting down...")
    api_routes.shutdown_services()

app = FastAPI(title="Emotion Recognition API", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development to fix "Disconnected" issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_routes.router, prefix="/api")
app.include_router(voice_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
