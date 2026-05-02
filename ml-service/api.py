from fastapi import FastAPI, BackgroundTasks # 🔥 ADD BackgroundTasks HERE
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel
from src.inference.inference import DebateAnalyzer
from src.rag.rag_pipeline import build_context_with_debug
from src.rag.llm_feedback import generate_feedback, generate_student_feedback
from src.rag.query_expansion import extract_keywords
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


class StudentFeedbackRequest(BaseModel):
    text: str
    topic: Optional[str] = None
    student_name: Optional[str] = None


def clamp_score(score: float, low: float = 0.0, high: float = 10.0) -> float:
    return round(max(low, min(high, score)), 1)


def has_relevant_context(context: str) -> bool:
    if not context:
        return False

    normalized = context.lower()
    return (
        "no relevant context found" not in normalized
        and "no factual context found" not in normalized
    )


def score_evidence_usage(argument: str, context: str) -> float:
    if not has_relevant_context(context):
        return 2.0

    argument_terms = set(extract_keywords(argument, max_keywords=12))
    context_terms = set(extract_keywords(context, max_keywords=40))

    if not argument_terms:
        return 3.0

    overlap_ratio = len(argument_terms & context_terms) / len(argument_terms)
    return clamp_score(2.5 + overlap_ratio * 7.5)


def build_rubric_scores(argument: str, context: str, prediction: dict) -> dict:
    quality = clamp_score(float(prediction.get("argument_quality", 0)) * 10)
    evidence = score_evidence_usage(argument, context)

    fallacy = prediction.get("fallacy", "None")
    logic_penalty = 2.0 if fallacy and fallacy != "None" else 0.0
    logical_reasoning = clamp_score((quality * 0.75) + 2.0 - logic_penalty)

    word_count = len(argument.split())
    if word_count < 8:
        clarity = 4.0
    elif word_count <= 80:
        clarity = clamp_score(6.5 + quality * 0.25)
    else:
        clarity = clamp_score(6.0 + quality * 0.15)

    rebuttal_markers = [
        "although", "but", "however", "instead", "while", "whereas",
        "counter", "opponents", "because", "therefore"
    ]
    marker_bonus = 1.0 if any(marker in argument.lower() for marker in rebuttal_markers) else 0.0
    rebuttal_readiness = clamp_score((quality * 0.55) + (evidence * 0.25) + marker_bonus)

    overall = clamp_score(
        quality * 0.30
        + evidence * 0.25
        + logical_reasoning * 0.20
        + clarity * 0.15
        + rebuttal_readiness * 0.10
    )

    return {
        "overall": overall,
        "argument_quality": quality,
        "evidence_usage": evidence,
        "logical_reasoning": logical_reasoning,
        "clarity": clarity,
        "rebuttal_readiness": rebuttal_readiness,
    }


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


@app.post("/student-feedback")
def student_feedback(request: StudentFeedbackRequest, background_tasks: BackgroundTasks):
    """
    Student-facing performance feedback endpoint.

    Returns model prediction, rubric scores, RAG context, retrieval debug data,
    and a Markdown feedback report for the student.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Argument text is required")

    cleaned_text = cleaner.clean(request.text)
    cleaned_topic = cleaner.clean(request.topic) if request.topic else None

    context, retrieval_debug = build_context_with_debug(
        cleaned_text,
        topic=cleaned_topic
    )

    prediction = analyzer.predict(cleaned_text, topic=cleaned_topic)
    rubric_scores = build_rubric_scores(cleaned_text, context, prediction)

    feedback, feedback_source, llm_error = generate_student_feedback(
        argument=cleaned_text,
        model_scores=prediction,
        rubric_scores=rubric_scores,
        topic=cleaned_topic,
        context=context,
        student_name=request.student_name,
    )

    scores = {
        "quality": prediction.get("argument_quality"),
        "component": prediction.get("component"),
        "stance": prediction.get("stance"),
    }
    background_tasks.add_task(log_interaction, cleaned_text, scores, feedback)

    return {
        "student_name": request.student_name,
        "topic": cleaned_topic,
        "original_input": request.text,
        "cleaned_input": cleaned_text,
        "prediction": prediction,
        "rubric_scores": rubric_scores,
        "context": context,
        "retrieval_debug": retrieval_debug,
        "feedback_source": feedback_source,
        "llm_error": llm_error,
        "student_feedback": feedback,
    }
