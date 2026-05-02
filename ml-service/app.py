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
# Input
# =========================
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
        # Show a loading spinner while the API does the heavy lifting
        with st.spinner("Running DeBERTa, fetching RAG context, and generating AI feedback..."):
            try:
                # 🔥 Make the HTTP POST request to your FastAPI backend
                response = requests.post(
                    "http://localhost:8000/analyze",
                    json={"topic": topic, "text": argument}
                )
                
                # Check if the API returned an error (like a 500 or 404)
                response.raise_for_status()
                
                # Parse the JSON response from api.py
                result_data = response.json()
                
                # Extract the pieces
                prediction = result_data["prediction"]
                llm_feedback = result_data["llm_feedback"]
                
                # --- 1. Render PyTorch Metrics ---
                st.subheader("DeBERTa Diagnostics")
                col1, col2, col3 = st.columns(3)

                # Ensure these keys match exactly what your predict() function outputs
                quality_score = prediction.get("argument_quality", 0)
                col1.metric("Quality Score", f"{quality_score:.3f}")
                col2.metric("Component", prediction.get("component", "N/A"))
                col3.metric("Stance", prediction.get("stance", "N/A"))
                
                st.progress(min(max(quality_score, 0), 1))

                st.divider()

                # --- 2. Render LLM Feedback ---
                st.subheader("🤖 AI Coach Feedback")
                st.write(llm_feedback)
                
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend. Is FastAPI running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
