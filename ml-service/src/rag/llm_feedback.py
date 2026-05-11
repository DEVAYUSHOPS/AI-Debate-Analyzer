import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .rag_pipeline import build_context


load_dotenv(Path(__file__).resolve().parents[2] / ".env")
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
            return "LLM API key missing. Please add GEMINI_API_KEY to your .env file."

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
        return f"Error calling LLM API: {str(e)}"


def llm_unavailable(response_text):
    if not response_text:
        return True

    failure_markers = [
        "api key missing",
        "error calling llm",
        "google genai sdk not installed",
        "resource_exhausted",
        "quota",
    ]

    normalized = response_text.lower()
    return any(marker in normalized for marker in failure_markers)


def clamp_10(score, fallback=0.0):
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = fallback

    return round(max(0.0, min(10.0, numeric_score)), 1)


def parse_json_object(response_text):
    if not response_text:
        return None

    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


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

Be clear, fair, and constructive. Do NOT be overly harsh.
Do NOT assume missing retrieved context means the student is wrong.

IMPORTANT RULES:
1. The retrieved context may be incomplete. If no evidence is found, say:
   "No supporting evidence was retrieved from the knowledge base."
   (Do NOT say the argument is false.)
2. If the student uses phrases like "research indicates", "studies show":
   treat this as weak or implicit evidence — not zero evidence.
3. Only use the retrieved context when it is clearly relevant.
4. Focus on helping the student improve, not penalizing system limitations.

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
(2-3 sentences summarizing performance fairly.)

### Score Breakdown
| Category | Score | Feedback |
| --- | ---: | --- |
| Argument Quality | X/10 | ... |
| Evidence Usage | X/10 | If phrases like "research indicates" are used, mention missing citation instead of saying no evidence. |
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
Use ONE of the following:
- "Supported by retrieved context"
- "Partially supported by retrieved context"
- "No supporting evidence was retrieved from the knowledge base"

### Action Plan
1. ...
2. ...
3. ...

### Improved Response
(Rewrite in 2-4 strong sentences with better evidence framing.)
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
    evidence_note = "No supporting evidence was retrieved from the knowledge base. Consider adding a specific study or statistic."
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

    return feedback, "llm", None


# =========================
# 5. Debate Flow Feedback
# =========================
def build_argument_turn_feedback_prompt(
    argument,
    context,
    model_scores,
    rubric_scores,
    topic=None,
    student_name=None,
    round_name=None,
):
    student_line = student_name or "Candidate"
    topic_line = topic or "Not provided"
    round_line = round_name or "Argument"

    return f"""You are a debate coach giving feedback for one speaking turn.

Write only what helps this speaker improve this exact argument. Do not compare
against the opponent in this per-turn feedback.

Topic: {topic_line}
Round: {round_line}
Speaker: {student_line}

<argument>
{argument}
</argument>

<nlp_model_analysis>
- Argument Quality: {model_scores.get("argument_quality", "N/A")}
- Component: {model_scores.get("component", "N/A")}
- Stance: {model_scores.get("stance", "N/A")}
- Fallacy: {model_scores.get("fallacy", "N/A")}
</nlp_model_analysis>

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

Return ONLY valid JSON with this shape:
{{
  "recommendation": "One concise paragraph, 45-90 words, with the most important improvement recommendation.",
  "improved_statement": "A stronger version of the student's argument in 2-4 sentences."
}}
"""


def basic_argument_turn_feedback(
    argument,
    model_scores,
    rubric_scores,
    context=None,
    topic=None,
    student_name=None,
    round_name=None,
):
    evidence_note = "Add one concrete source, example, statistic, or study to make the claim easier to verify."
    if context and "No supporting evidence retrieved" not in context:
        evidence_note = "Use the retrieved evidence more directly by naming the fact and connecting it to the claim."

    fallacy = model_scores.get("fallacy", "None")
    fallacy_note = ""
    if fallacy and fallacy != "None":
        fallacy_note = f" Also revise the reasoning to avoid {fallacy.lower()}."

    recommendation = (
        f"{student_name or 'The speaker'} has a usable point for {round_name or 'this turn'}, "
        f"but it needs sharper reasoning and evidence. {evidence_note}{fallacy_note} "
        f"Aim for one clear claim, one specific reason, and one sentence explaining why it matters for the topic."
    )

    improved_statement = (
        f"On the topic {topic or 'being debated'}, {argument.strip()} "
        "This point is stronger when it is supported with a specific example or statistic and linked clearly to the debate impact."
    )

    return {
        "recommendation": recommendation,
        "improved_statement": improved_statement,
    }


def generate_argument_turn_feedback(
    argument,
    model_scores,
    rubric_scores,
    topic=None,
    context=None,
    student_name=None,
    round_name=None,
):
    if context is None:
        context = build_context(argument, topic=topic)

    prompt = build_argument_turn_feedback_prompt(
        argument=argument,
        context=context,
        model_scores=model_scores,
        rubric_scores=rubric_scores,
        topic=topic,
        student_name=student_name,
        round_name=round_name,
    )

    feedback = call_llm(prompt)

    if llm_unavailable(feedback):
        return (
            basic_argument_turn_feedback(
                argument=argument,
                model_scores=model_scores,
                rubric_scores=rubric_scores,
                context=context,
                topic=topic,
                student_name=student_name,
                round_name=round_name,
            ),
            "fallback",
            feedback,
        )

    parsed = parse_json_object(feedback)
    if not parsed:
        fallback = basic_argument_turn_feedback(
            argument=argument,
            model_scores=model_scores,
            rubric_scores=rubric_scores,
            context=context,
            topic=topic,
            student_name=student_name,
            round_name=round_name,
        )
        fallback["recommendation"] = feedback.strip()
        return fallback, "llm_unparsed", "Could not parse LLM JSON response."

    return (
        {
            "recommendation": str(parsed.get("recommendation", "")).strip(),
            "improved_statement": str(parsed.get("improved_statement", "")).strip(),
        },
        "llm",
        None,
    )


def build_overall_debate_feedback_prompt(
    topic,
    speaker_a,
    speaker_b,
    turn_summaries,
    speaker_scores,
):
    summaries_json = json.dumps(turn_summaries, ensure_ascii=False, indent=2)

    return f"""You are a fair debate judge comparing two candidates across a full debate.

Use the NLP turn scores as evidence, but judge the debate holistically across
opening statement, rebuttal, and closing statement. Consider clarity, reasoning,
evidence, responsiveness, and final persuasive force.

Topic: {topic or "Not provided"}
Speaker A: {speaker_a}
Speaker B: {speaker_b}

<average_nlp_scores_out_of_10>
Speaker A: {speaker_scores.get("speakerA", 0)}
Speaker B: {speaker_scores.get("speakerB", 0)}
</average_nlp_scores_out_of_10>

<turn_summaries>
{summaries_json}
</turn_summaries>

Return ONLY valid JSON with this shape:
{{
  "speakerA_score": 0.0,
  "speakerB_score": 0.0,
  "winner_key": "speakerA",
  "overall_comparison": "One paragraph comparing both candidates across the debate.",
  "speakerA_feedback": "One paragraph of candidate-specific feedback for Speaker A.",
  "speakerB_feedback": "One paragraph of candidate-specific feedback for Speaker B.",
  "final_verdict": "One paragraph explaining why the winner won."
}}

Rules:
- Scores must be numbers from 0 to 10 with at most one decimal place.
- winner_key must be either "speakerA" or "speakerB".
- The winner must match the higher score. If scores are tied, choose the candidate with the stronger rebuttal and closing.
"""


def basic_overall_debate_feedback(topic, speaker_a, speaker_b, speaker_scores):
    score_a = clamp_10(speaker_scores.get("speakerA", 0))
    score_b = clamp_10(speaker_scores.get("speakerB", 0))
    winner_key = "speakerA" if score_a >= score_b else "speakerB"
    winner_name = speaker_a if winner_key == "speakerA" else speaker_b

    return {
        "speakerA_score": score_a,
        "speakerB_score": score_b,
        "winner_key": winner_key,
        "winner_name": winner_name,
        "overall_comparison": (
            f"{speaker_a} averaged {score_a}/10 and {speaker_b} averaged {score_b}/10 across the debate. "
            "The comparison is based on the recorded NLP scores and the strongest arguments from each round."
        ),
        "speakerA_feedback": (
            f"{speaker_a} should keep the strongest claim from each round, add more specific evidence, "
            "and make the closing statement clearly summarize why their side outweighs the opponent."
        ),
        "speakerB_feedback": (
            f"{speaker_b} should strengthen evidence, answer the opponent more directly in rebuttal, "
            "and use the closing statement to connect reasoning to the debate topic."
        ),
        "final_verdict": f"{winner_name} wins on the available average score and overall consistency.",
    }


def normalize_overall_feedback(parsed, speaker_a, speaker_b, speaker_scores):
    fallback = basic_overall_debate_feedback(None, speaker_a, speaker_b, speaker_scores)

    score_a = clamp_10(parsed.get("speakerA_score"), fallback["speakerA_score"])
    score_b = clamp_10(parsed.get("speakerB_score"), fallback["speakerB_score"])

    winner_key = parsed.get("winner_key")
    if winner_key not in {"speakerA", "speakerB"}:
        winner_key = "speakerA" if score_a >= score_b else "speakerB"

    if score_a > score_b:
        winner_key = "speakerA"
    elif score_b > score_a:
        winner_key = "speakerB"

    winner_name = speaker_a if winner_key == "speakerA" else speaker_b

    return {
        "speakerA_score": score_a,
        "speakerB_score": score_b,
        "winner_key": winner_key,
        "winner_name": winner_name,
        "overall_comparison": str(
            parsed.get("overall_comparison") or fallback["overall_comparison"]
        ).strip(),
        "speakerA_feedback": str(
            parsed.get("speakerA_feedback") or fallback["speakerA_feedback"]
        ).strip(),
        "speakerB_feedback": str(
            parsed.get("speakerB_feedback") or fallback["speakerB_feedback"]
        ).strip(),
        "final_verdict": str(
            parsed.get("final_verdict") or fallback["final_verdict"]
        ).strip(),
    }


def generate_overall_debate_feedback(
    topic,
    speaker_a,
    speaker_b,
    turn_summaries,
    speaker_scores,
):
    prompt = build_overall_debate_feedback_prompt(
        topic=topic,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        turn_summaries=turn_summaries,
        speaker_scores=speaker_scores,
    )

    feedback = call_llm(prompt)

    if llm_unavailable(feedback):
        return (
            basic_overall_debate_feedback(topic, speaker_a, speaker_b, speaker_scores),
            "fallback",
            feedback,
        )

    parsed = parse_json_object(feedback)
    if not parsed:
        fallback = basic_overall_debate_feedback(topic, speaker_a, speaker_b, speaker_scores)
        fallback["overall_comparison"] = feedback.strip()
        return fallback, "llm_unparsed", "Could not parse LLM JSON response."

    return (
        normalize_overall_feedback(parsed, speaker_a, speaker_b, speaker_scores),
        "llm",
        None,
    )


def build_full_debate_feedback_prompt(
    topic,
    speaker_a,
    speaker_b,
    turn_summaries,
    speaker_scores,
):
    summaries_json = json.dumps(turn_summaries, ensure_ascii=False, indent=2)

    return f"""You are a fair debate coach and judge.

Evaluate the full debate in one pass. Use the NLP scores as model evidence, but
write the feedback yourself based on the actual arguments.

Topic: {topic or "Not provided"}
Speaker A: {speaker_a}
Speaker B: {speaker_b}

<average_nlp_scores_out_of_10>
Speaker A: {speaker_scores.get("speakerA", 0)}
Speaker B: {speaker_scores.get("speakerB", 0)}
</average_nlp_scores_out_of_10>

<turns>
{summaries_json}
</turns>

Return ONLY valid JSON with this shape:
{{
  "turn_feedback": [
    {{
      "turn_id": "same turn_id from the input",
      "recommendation": "One concise paragraph, 45-90 words, with the most important recommendation for this exact argument.",
      "improved_statement": "A stronger version of this argument in 2-4 sentences."
    }}
  ],
  "overall": {{
    "speakerA_score": 0.0,
    "speakerB_score": 0.0,
    "winner_key": "speakerA",
    "overall_comparison": "One paragraph comparing both candidates across opening, rebuttal, and closing.",
    "speakerA_feedback": "One paragraph of candidate-specific feedback for Speaker A.",
    "speakerB_feedback": "One paragraph of candidate-specific feedback for Speaker B.",
    "final_verdict": "One paragraph explaining why the winner won."
  }}
}}

Rules:
- Include exactly one turn_feedback item for every input turn_id.
- Per-turn recommendation must be only one paragraph.
- Improved statement must preserve the speaker's side of the debate.
- Scores must be numbers from 0 to 10 with at most one decimal place.
- winner_key must be either "speakerA" or "speakerB".
- The winner must match the higher score. If scores are tied, choose the candidate with the stronger rebuttal and closing.
"""


def basic_turn_feedback_from_summary(turn):
    speaker_name = turn.get("speaker_name") or "The speaker"
    round_name = turn.get("round") or "this round"
    argument = turn.get("argument") or "The original argument needs more detail."
    context = turn.get("context") or ""

    evidence_note = "Add a concrete example, fact, or statistic and explain why it matters."
    if context:
        evidence_note = "Use the retrieved evidence more directly and connect it to the main claim."

    return {
        "turn_id": turn.get("turn_id"),
        "recommendation": (
            f"{speaker_name}'s {round_name} has a clear direction, but it should be made more persuasive. "
            f"{evidence_note} The argument will improve if it states one precise claim, supports it with one specific reason, "
            "and then links that reason back to the debate topic."
        ),
        "improved_statement": (
            f"{argument} This position would be stronger with a specific example and a clearer explanation of how it affects the debate outcome."
        ),
    }


def normalize_turn_feedback_items(parsed_items, turn_summaries):
    parsed_by_id = {}

    if isinstance(parsed_items, list):
        for item in parsed_items:
            if not isinstance(item, dict):
                continue

            turn_id = item.get("turn_id")
            if not turn_id:
                continue

            parsed_by_id[str(turn_id)] = {
                "turn_id": str(turn_id),
                "recommendation": str(item.get("recommendation", "")).strip(),
                "improved_statement": str(item.get("improved_statement", "")).strip(),
            }

    normalized = []
    for turn in turn_summaries:
        turn_id = str(turn.get("turn_id"))
        item = parsed_by_id.get(turn_id)

        if not item or not item.get("recommendation") or not item.get("improved_statement"):
            item = basic_turn_feedback_from_summary(turn)

        normalized.append(item)

    return normalized


def basic_full_debate_feedback(topic, speaker_a, speaker_b, turn_summaries, speaker_scores):
    return {
        "turn_feedback": [
            basic_turn_feedback_from_summary(turn)
            for turn in turn_summaries
        ],
        "overall": basic_overall_debate_feedback(
            topic,
            speaker_a,
            speaker_b,
            speaker_scores,
        ),
    }


def normalize_full_debate_feedback(
    parsed,
    topic,
    speaker_a,
    speaker_b,
    turn_summaries,
    speaker_scores,
):
    if not isinstance(parsed, dict):
        return basic_full_debate_feedback(
            topic,
            speaker_a,
            speaker_b,
            turn_summaries,
            speaker_scores,
        )

    turn_feedback = normalize_turn_feedback_items(
        parsed.get("turn_feedback"),
        turn_summaries,
    )
    overall = normalize_overall_feedback(
        parsed.get("overall") if isinstance(parsed.get("overall"), dict) else {},
        speaker_a,
        speaker_b,
        speaker_scores,
    )

    return {
        "turn_feedback": turn_feedback,
        "overall": overall,
    }


def generate_full_debate_feedback(
    topic,
    speaker_a,
    speaker_b,
    turn_summaries,
    speaker_scores,
):
    prompt = build_full_debate_feedback_prompt(
        topic=topic,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        turn_summaries=turn_summaries,
        speaker_scores=speaker_scores,
    )

    feedback = call_llm(prompt)

    if llm_unavailable(feedback):
        return (
            basic_full_debate_feedback(
                topic,
                speaker_a,
                speaker_b,
                turn_summaries,
                speaker_scores,
            ),
            "fallback",
            feedback,
        )

    parsed = parse_json_object(feedback)
    if not parsed:
        fallback = basic_full_debate_feedback(
            topic,
            speaker_a,
            speaker_b,
            turn_summaries,
            speaker_scores,
        )
        fallback["overall"]["overall_comparison"] = feedback.strip()
        return fallback, "llm_unparsed", "Could not parse LLM JSON response."

    return (
        normalize_full_debate_feedback(
            parsed,
            topic,
            speaker_a,
            speaker_b,
            turn_summaries,
            speaker_scores,
        ),
        "llm",
        None,
    )


# =========================
# 6. Optional: Lightweight Mode
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
