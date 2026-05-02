# src/rag/llm_feedback.py

from .rag_pipeline import build_context
import os
from dotenv import load_dotenv


load_dotenv()


# =========================
# 1. Prompt Builder
# =========================
# =========================
# 1. Prompt Builder (Optimized for Gemini)
# =========================
def build_prompt(argument, context, scores=None, topic=None):
    """
    Builds a structured, constraint-heavy prompt for Gemini using XML delimiters.
    """
    
    score_text = ""
    if scores:
        # Format the quality score nicely if it's a float
        q_score = scores.get('quality', 'N/A')
        if isinstance(q_score, float):
            q_score = f"{q_score:.3f} / 1.0"
            
        score_text = f"""
<deberta_model_analysis>
- Mathematical Quality Score: {q_score}
- Detected Argument Type: {scores.get('component', 'N/A')}
- Detected Stance: {scores.get('stance', 'N/A')}
</deberta_model_analysis>
"""

    topic_text = ""
    if topic:
        topic_text = f"""
<debate_topic>
{topic}
</debate_topic>
"""

    prompt = f"""You are a world-class debate coach and critical analysis AI. 
Your task is to analyze a user's argument using ONLY the provided factual context and the raw machine learning scores.

{score_text}
{topic_text}

<retrieved_factual_context>
{context}
</retrieved_factual_context>

<user_argument>
{argument}
</user_argument>

INSTRUCTIONS:
1. FACT-CHECK: Compare the <user_argument> strictly against the <retrieved_factual_context>. Do not hallucinate outside facts.
2. AUDIT THE MATH: Review the <deberta_model_analysis>. If the DeBERTa quality score is high but the argument is factually incorrect or sarcastic, point out that the structural math missed the semantic truth.
3. TONE: Be concise, professional, brutally honest, and constructive.

Provide your output EXACTLY in the following Markdown format. Do not include any other conversational filler.

### 🔍 Fact Check
(1-2 sentences stating if the argument is supported, contradicted, or unverified by the context.)

### 💪 Structural Strengths
(1 sentence on what the debater did well grammatically or rhetorically.)

### ⚠️ Critical Weaknesses
(Identify logical fallacies, missing evidence, or if the argument relies on sarcasm rather than fact.)

### 💡 Coach's Rewrite
(Provide a specific, 1-sentence rewrite that makes this argument significantly stronger and more factual.)
"""

    return prompt

# =========================
# 2. LLM Call (Gemini Integration)
# =========================

def call_llm(prompt):
    """
    Calls the Google Gemini API using the new google-genai SDK.
    """
    try:
        from google import genai
        from google.genai import types
        
        # Pull the API key from your environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ Gemini API key missing. Please add GEMINI_API_KEY to your .env file."

        # Initialize the modern Client
        client = genai.Client(api_key=api_key)

        # Generate the response using the upgraded 2.5 Flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )

        return response.text

    except ImportError:
        return "⚠️ Google GenAI SDK not installed. Run: pip install google-genai"
    except Exception as e:
        return f"⚠️ Error calling Gemini API: {str(e)}"

# =========================
# 3. Main Feedback Function
# =========================
def generate_feedback(argument, scores=None, topic=None, context=None):
    """
    Full pipeline:
    argument → RAG → LLM → feedback
    """

    # Step 1: Get context from RAG, unless caller already retrieved it.
    if context is None:
        context = build_context(argument, topic=topic)

    # Step 2: Build prompt
    prompt = build_prompt(argument, context, scores, topic=topic)

    # Step 3: Call LLM
    feedback = call_llm(prompt)

    return feedback


# =========================
# 4. Optional: Lightweight Mode (No LLM)
# =========================
def basic_feedback(argument):
    """
    Fallback if no LLM available
    """

    context = build_context(argument)

    if "No relevant context" in context:
        return "❌ No supporting evidence found."

    overlap = sum(word in context.lower() for word in argument.lower().split())

    if overlap > 5:
        return "✅ Argument is reasonably supported by evidence."

    elif overlap > 2:
        return "⚠️ Argument has partial support but needs stronger evidence."

    else:
        return "❓ Argument lacks strong factual grounding."
