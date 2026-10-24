import os
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_service import get_service, EMOTION_META

# Preload models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize service and load models
    service = get_service()
    print("Emotion Recognition API initialized and ready.")
    yield
    print("Emotion Recognition API shutting down.")

app = FastAPI(
    title="Emotion Recognition System API",
    description="Multi-model text emotion recognition inference API powered by Keras BiLSTM-CNN and Scikit-Learn TF-IDF models.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local testing or custom origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Request Schemas
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to analyze")
    model: Optional[str] = Field("keras", description="Model identifier: 'keras', 'logistic', 'svm', or 'compare_all'")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=200, description="List of text samples")
    model: Optional[str] = Field("keras", description="Model identifier: 'keras', 'logistic', 'svm', or 'compare_all'")

# Sample presets for immediate testing
SAMPLE_PRESETS = [
    {
        "emotion": "joy",
        "title": "Dream Job Offer",
        "emoji": "😊",
        "text": "I just received the call offering me my dream role! I am overflowing with joy and cannot stop smiling!"
    },
    {
        "emotion": "sadness",
        "title": "Saying Goodbye",
        "emoji": "😢",
        "text": "It was so hard waving goodbye today. My heart feels so heavy and an empty ache lingers inside me."
    },
    {
        "emotion": "anger",
        "title": "Unfair Treatment",
        "emoji": "😡",
        "text": "I am absolutely disgusted and furious with how unfairly they treated us! This blatant disrespect makes my blood boil."
    },
    {
        "emotion": "fear",
        "title": "Nightmare in the Dark",
        "emoji": "😨",
        "text": "I was walking down the dark alley alone when sudden footsteps behind me made my pulse spike with pure dread."
    },
    {
        "emotion": "love",
        "title": "Heartfelt Devotion",
        "emoji": "❤️",
        "text": "You bring so much gentle warmth and meaning into my life. I cherish and adore you with all of my heart."
    },
    {
        "emotion": "surprise",
        "title": "Unexpected Party",
        "emoji": "😲",
        "text": "I opened the front door to total darkness, then the lights flicked on and everyone shouted Surprise! I was completely stunned!"
    }
]

@app.get("/api/metadata")
def get_metadata():
    """Returns complete model architectures, configurations, labels, and system status."""
    service = get_service()
    return service.get_system_metadata()

@app.get("/api/presets")
def get_presets():
    """Returns curated example prompts covering all 6 emotion classes."""
    return {"presets": SAMPLE_PRESETS}

@app.get("/api/health")
def health_check():
    service = get_service()
    return {"status": "ok", "loaded": service.loaded, "labels": service.labels}

@app.post("/api/predict")
def predict_emotion(req: PredictRequest):
    """Predict emotion for a single text input with probabilities, latency, and sentiment."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    service = get_service()
    try:
        result = service.predict(text, model_type=req.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/api/batch")
def batch_predict(req: BatchPredictRequest):
    """Batch predict emotion across a list of texts with distribution analytics."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="Input texts list cannot be empty.")
    
    service = get_service()
    try:
        result = service.batch_predict(req.texts, model_type=req.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

# Mount static files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Emotion Recognition Backend is running. Frontend static files are loading."}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" Emotion Recognition System Web Server")
    print(" Access the application at: http://localhost:8000")
    print(" Interactive API documentation at: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
