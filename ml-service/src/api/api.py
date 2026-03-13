from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.inference import DebateAnalyzer

app = FastAPI(
    title="AI Debate Analyzer",
    description="Analyze arguments for quality, component type, and stance",
    version="1.0"
)

# Load model once
analyzer = DebateAnalyzer()


# =========================
# Request Schema
# =========================

class ArgumentRequest(BaseModel):
    text: str


# =========================
# Root Endpoint
# =========================

@app.get("/")
def home():
    return {"message": "AI Debate Analyzer API is running"}


# =========================
# Prediction Endpoint
# =========================

@app.post("/analyze")
def analyze_argument(request: ArgumentRequest):

    result = analyzer.predict(request.text)

    return result