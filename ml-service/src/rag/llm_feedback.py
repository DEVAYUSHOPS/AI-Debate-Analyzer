# src/rag/llm_feedback.py

import os

from dotenv import load_dotenv

from .rag_pipeline import build_context


load_dotenv()


# =========================
# 1. Argument Feedback Prompt
# =========================
def build_prompt(argument, context, scores=None, topic=None):
    """
    Builds a structured prompt for argument-level debate coaching.
    """
    score_text = ""
    if scores:
        q_score = scores.get("quality", "N/A")
        if isinstance(q_score, float):
            q_score = f"{q_score:.3f} / 1.0"

        score_text = f"""
<deberta_model_analysis>
- Mathematical Quality Score: {q_score}
- Detected Argument Type: {scores.get("component", "N/A")}
- Detected Stance: {scores.get("stance", "N/A")}
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
1. FACT-CHECK: Compare the user argument strictly against the retrieved factual context. Do not hallucinate outside facts.
2. AUDIT THE MATH: Review the DeBERTa model analysis. If the DeBERTa quality score is high but the argument is factually incorrect or sarcastic, point out that the structural math missed the semantic truth.
3. TONE: Be concise, professional, direct, and constructive.

Provide your output EXACTLY in the following Markdown format. Do not include any other conversational filler.

### Fact Check
(1-2 sentences stating if the argument is supported, contradicted, or unverified by the context.)

### Structural Strengths
(1 sentence on what the debater did well grammatically or rhetorically.)

### Critical Weaknesses
(Identify logical fallacies, missing evidence, or if the argument relies on sarcasm rather than fact.)

### Coach's Rewrite
(Provide a specific, 1-sentence rewrite that makes this argument significantly stronger and more factual.)
"""

    return prompt


# =========================
# 2. LLM Call
# =========================
def call_llm(prompt):
    """
    Calls the Google Gemini API using the google-genai SDK.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Gemini API key missing. Please add GEMINI_API_KEY to your .env file."

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )

        return response.text

    except ImportError:
        return "Google GenAI SDK not installed. Run: pip install google-genai"
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


def llm_unavailable(response_text):
    if not response_text:
        return True

    failure_markers = [
        "api key missing",
        "error calling gemini",
        "google genai sdk not installed",
        "resource_exhausted",
        "quota",
    ]

    normalized = response_text.lower()
    return any(marker in normalized for marker in failure_markers)


# =========================
# 3. Argument Feedback
# =========================
def generate_feedback(argument, scores=None, topic=None, context=None):
    """
    Full argument-level feedback pipeline:
    argument -> RAG -> LLM -> feedback
    """
    if context is None:
        context = build_context(argument, topic=topic)

    prompt = build_prompt(argument, context, scores, topic=topic)
    feedback = call_llm(prompt)

    return feedback


# =========================
# 4. Student Performance Feedback
# =========================
def build_student_feedback_prompt(
    argument,
    context,
    model_scores,
    rubric_scores,
    topic=None,
    student_name=None
):
    student_line = f"Student: {student_name}" if student_name else "Student: Candidate"
    topic_line = f"Topic: {topic}" if topic else "Topic: Not provided"

    prompt = f"""You are an academic debate coach giving feedback to a student.
Be clear, constructive, specific, and professional. Do not be harsh or vague.
Use ONLY the retrieved factual context for factual claims. If evidence is missing, say so.

<student>
{student_line}
</student>

<debate_topic>
{topic_line}
</debate_topic>

<student_argument>
{argument}
</student_argument>

<model_analysis>
- Argument Quality: {model_scores.get("argument_quality", "N/A")}
- Component: {model_scores.get("component", "N/A")}
- Stance: {model_scores.get("stance", "N/A")}
- Fallacy: {model_scores.get("fallacy", "N/A")}
</model_analysis>

<rubric_scores>
- Overall: {rubric_scores.get("overall", "N/A")}/10
- Argument Quality: {rubric_scores.get("argument_quality", "N/A")}/10
- Evidence Usage: {rubric_scores.get("evidence_usage", "N/A")}/10
- Logical Reasoning: {rubric_scores.get("logical_reasoning", "N/A")}/10
- Clarity: {rubric_scores.get("clarity", "N/A")}/10
- Rebuttal Readiness: {rubric_scores.get("rebuttal_readiness", "N/A")}/10
</rubric_scores>

<retrieved_factual_context>
{context}
</retrieved_factual_context>

Return EXACTLY this Markdown structure:

### Overall Performance
(2-3 sentences summarizing the student's performance.)

### Score Breakdown
| Category | Score | Feedback |
| --- | ---: | --- |
| Argument Quality | X/10 | ... |
| Evidence Usage | X/10 | ... |
| Logical Reasoning | X/10 | ... |
| Clarity | X/10 | ... |
| Rebuttal Readiness | X/10 | ... |

### Strengths
- ...
- ...

### Areas To Improve
- ...
- ...

### Factual Grounding
(State whether the argument is supported, partially supported, or unverified by the retrieved context.)

### Action Plan
1. ...
2. ...
3. ...

### Improved Response
(Rewrite the student's argument in 2-4 polished sentences.)
"""

    return prompt


def basic_student_feedback(
    argument,
    model_scores,
    rubric_scores,
    context=None,
    topic=None,
    student_name=None
):
    evidence_note = "The retrieved context was not strong enough to verify the factual claims."
    if context and "No relevant context found" not in context:
        evidence_note = "Relevant context was retrieved, but the student should cite specific evidence directly."

    fallacy = model_scores.get("fallacy", "None")
    fallacy_note = "No simple rule-based fallacy was detected."
    if fallacy and fallacy != "None":
        fallacy_note = f"The argument may contain this issue: {fallacy}."

    return f"""### Overall Performance
{student_name or "The student"} gave a clear argument on the topic "{topic or "not provided"}". The response has a detectable stance and a basic reason, but it needs stronger evidence and deeper explanation to become a high-scoring debate answer.

### Score Breakdown
| Category | Score | Feedback |
| --- | ---: | --- |
| Argument Quality | {rubric_scores.get("argument_quality")}/10 | The argument has a clear structure, but the reasoning can be developed further. |
| Evidence Usage | {rubric_scores.get("evidence_usage")}/10 | {evidence_note} |
| Logical Reasoning | {rubric_scores.get("logical_reasoning")}/10 | {fallacy_note} |
| Clarity | {rubric_scores.get("clarity")}/10 | The main point is understandable and easy to follow. |
| Rebuttal Readiness | {rubric_scores.get("rebuttal_readiness")}/10 | The answer needs more detail to handle counterarguments. |

### Strengths
- The stance is clear: {model_scores.get("stance", "N/A")}.
- The response is concise and directly addresses the topic.

### Areas To Improve
- Add concrete evidence, examples, or data.
- Explain why the reason matters, not only what the reason is.
- Anticipate one likely counterargument.

### Factual Grounding
{evidence_note}

### Action Plan
1. Add one specific factual example.
2. Explain the impact of that evidence on students, schools, or society.
3. Include one sentence responding to the opposing side.

### Improved Response
{argument}
This argument would be stronger if it included a specific piece of evidence and explained how that evidence supports the student's stance.
"""


def generate_student_feedback(
    argument,
    model_scores,
    rubric_scores,
    topic=None,
    context=None,
    student_name=None
):
    if context is None:
        context = build_context(argument, topic=topic)

    prompt = build_student_feedback_prompt(
        argument=argument,
        context=context,
        model_scores=model_scores,
        rubric_scores=rubric_scores,
        topic=topic,
        student_name=student_name,
    )

    feedback = call_llm(prompt)

    if llm_unavailable(feedback):
        return (
            basic_student_feedback(
                argument=argument,
                model_scores=model_scores,
                rubric_scores=rubric_scores,
                context=context,
                topic=topic,
                student_name=student_name,
            ),
            "fallback",
            feedback,
        )

    return feedback, "gemini", None


# =========================
# 5. Optional: Lightweight Mode
# =========================
def basic_feedback(argument):
    """
    Fallback if no LLM is available.
    """
    context = build_context(argument)

    if "No relevant context" in context:
        return "No supporting evidence found."

    overlap = sum(word in context.lower() for word in argument.lower().split())

    if overlap > 5:
        return "Argument is reasonably supported by evidence."

    if overlap > 2:
        return "Argument has partial support but needs stronger evidence."

    return "Argument lacks strong factual grounding."
