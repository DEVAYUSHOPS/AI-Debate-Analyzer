from fastapi import FastAPI, BackgroundTasks # 🔥 ADD BackgroundTasks HERE
from typing import Optional

from pydantic import BaseModel
from src.inference.inference import DebateAnalyzer
from src.rag.rag_pipeline import build_context_with_debug
from src.rag.llm_feedback import generate_feedback
from src.db_service import log_interaction # 🔥 IMPORT YOUR NEW DB SERVICE

from symspellpy import SymSpell
import pkg_resources
from whisper_normalizer.english import EnglishTextNormalizer

app = FastAPI(
    title="AI Debate Analyzer",
    description="Analyze arguments with RAG + AI feedback",
    version="2.0"
)

# =========================
# Text Cleaning Service
# =========================
class TextCleaner:
    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        self.normalizer = EnglishTextNormalizer()

    def clean(self, text: str) -> str:
        text = self.normalizer(text)
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term if suggestions else text


# =========================
# Load Services (Singletons)
# =========================
cleaner = TextCleaner()
analyzer = DebateAnalyzer()


# =========================
# Request Schema
# =========================
class ArgumentRequest(BaseModel):
    text: str
    topic: Optional[str] = None


# =========================
# Root Endpoint
# =========================
@app.get("/")
def home():
    return {"message": "AI Debate Analyzer (RAG + LLM) is running 🚀"}


# =========================
# Main Endpoint
# =========================
@app.post("/analyze")
def analyze_argument(request: ArgumentRequest, background_tasks: BackgroundTasks): 
    
    # Step 1: Clean Text
    cleaned_text = cleaner.clean(request.text)
    cleaned_topic = cleaner.clean(request.topic) if request.topic else None

    # Step 2: RAG Context
    context, retrieval_debug = build_context_with_debug(
        cleaned_text,
        topic=cleaned_topic
    )

    # Step 3: Model Prediction
    result = analyzer.predict(cleaned_text, topic=cleaned_topic)

    scores = {
        "quality": result.get("argument_quality"),
        "component": result.get("component"),
        "stance": result.get("stance")
    }

    # Step 4: LLM Feedback
    try:
        feedback = generate_feedback(
            cleaned_text,
            scores,
            topic=cleaned_topic,
            context=context
        )
    except Exception as e:
        feedback = f"LLM feedback unavailable: {str(e)}"

    # 🔥 Step 5: The RLAIF Flywheel (Runs in the background!)
    background_tasks.add_task(log_interaction, cleaned_text, scores, feedback)

    # Final Response
    return {
        "original_input": request.text,
        "topic": cleaned_topic,
        "cleaned_input": cleaned_text,
        "context": context,
        "retrieval_debug": retrieval_debug,
        "prediction": result,
        "llm_feedback": feedback
    }
