from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.inference import DebateAnalyzer
from symspellpy import SymSpell, Verbosity
import pkg_resources
from whisper_normalizer.english import EnglishTextNormalizer

app = FastAPI(
    title="AI Debate Analyzer",
    description="Analyze arguments for quality, component type, and stance",
    version="1.0"
)

# =========================
# Text Cleaning Service
# =========================
class TextCleaner:
    def __init__(self):
        # Initialize SymSpell for fast spelling correction
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
        # Initialize Whisper Normalizer for STT-specific structure fixing
        self.normalizer = EnglishTextNormalizer()

    def clean(self, text: str) -> str:
        # 1. Fix Speech-to-Text artifacts (expand contractions, standardize casing)
        text = self.normalizer(text)
        
        # 2. Compound spelling correction (handles joined words or typos in sentences)
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term if suggestions else text

# Load services once
cleaner = TextCleaner()

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
    # 🔥 Step 1: Fix spelling and structure
    cleaned_text = cleaner.clean(request.text)
    
    # Step 2: Pass cleaned text to the DeBERTa model
    result = analyzer.predict(cleaned_text)
    
    # Return result with the "fixed" text so the user can see the changes
    result["cleaned_input"] = cleaned_text
    return result