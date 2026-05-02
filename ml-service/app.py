import streamlit as st
import requests

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI Debate Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Debate Analyzer")
st.write("Analyze arguments with DeBERTa, RAG, and AI coaching.")

# =========================
# Mode Selection (🔥 NEW)
# =========================
mode = st.radio(
    "Select Mode:",
    ["Standard Analysis", "Student Feedback"]
)

# =========================
# Input
# =========================
student_name = None
if mode == "Student Feedback":
    student_name = st.text_input("Enter your name (optional):")

topic = st.text_input("Enter debate topic/motion (optional):")

argument = st.text_area(
    "Enter your argument:",
    height=200
)

# =========================
# Analyze Button
# =========================
if st.button("Analyze Argument"):

    if argument.strip() == "":
        st.warning("Please enter an argument.")
    else:
        with st.spinner("Running AI analysis..."):
            try:
                # =========================
                # 🔥 SWITCH API BASED ON MODE
                # =========================
                if mode == "Standard Analysis":
                    response = requests.post(
                        "http://localhost:8000/analyze",
                        json={"topic": topic, "text": argument}
                    )
                else:
                    response = requests.post(
                        "http://localhost:8000/student-feedback",
                        json={
                            "topic": topic,
                            "text": argument,
                            "student_name": student_name
                        }
                    )

                response.raise_for_status()
                result_data = response.json()

                # =========================
                # COMMON: Prediction
                # =========================
                prediction = result_data["prediction"]

                st.subheader("DeBERTa Diagnostics")
                col1, col2, col3 = st.columns(3)

                quality_score = prediction.get("argument_quality", 0)

                col1.metric("Quality Score", f"{quality_score:.3f}")
                col2.metric("Component", prediction.get("component", "N/A"))
                col3.metric("Stance", prediction.get("stance", "N/A"))

                st.progress(min(max(quality_score, 0), 1))

                st.divider()

                # =========================
                # MODE: Standard Analysis
                # =========================
                if mode == "Standard Analysis":
                    st.subheader("🤖 AI Coach Feedback")
                    st.write(result_data.get("llm_feedback", "No feedback available"))

                # =========================
                # MODE: Student Feedback (🔥 NEW)
                # =========================
                else:
                    rubric = result_data.get("rubric_scores", {})

                    st.subheader("📚 Rubric Evaluation")

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Overall", rubric.get("overall", 0))
                    r2.metric("Evidence", rubric.get("evidence_usage", 0))
                    r3.metric("Logic", rubric.get("logical_reasoning", 0))

                    r4, r5 = st.columns(2)
                    r4.metric("Clarity", rubric.get("clarity", 0))
                    r5.metric("Rebuttal", rubric.get("rebuttal_readiness", 0))

                    st.divider()

                    st.subheader("🤖 AI Coach Feedback")
                    st.markdown(result_data.get("student_feedback", "No feedback available"))

                    # Optional debug
                    with st.expander("🔍 Debug Info"):
                        st.write("Feedback Source:", result_data.get("feedback_source"))
                        if result_data.get("llm_error"):
                            st.error(result_data.get("llm_error"))

            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to backend. Is FastAPI running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {e}")