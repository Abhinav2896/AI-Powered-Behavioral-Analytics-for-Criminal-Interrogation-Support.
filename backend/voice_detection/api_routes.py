
from fastapi import APIRouter
from .stress_engine import VoiceStressEngine

router = APIRouter()
engine = VoiceStressEngine()

# Initialize engine on import or first request?
# Better to initialize explicitly on startup, but here we can lazy load or 
# assume app startup hooks it.
# Let's trigger initialization when the module is loaded or router created.
engine.initialize()

@router.get("/voice/status")
async def get_voice_status():
    """
    Returns real-time voice stress analysis.
    """
    return engine.get_status()
