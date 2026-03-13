import streamlit as st
from src.inference.inference import DebateAnalyzer

# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    return DebateAnalyzer()

analyzer = load_model()

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="AI Debate Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Debate Analyzer")
st.write("Analyze arguments for **quality, component type, and stance**.")

# =========================
# Input
# =========================

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

        result = analyzer.predict(argument)

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Argument Quality", result["argument_quality"])
        col2.metric("Component", result["component"])
        col3.metric("Stance", result["stance"])
        
        st.metric("Logical Fallacy", result["fallacy"])

        st.progress(min(max(result["argument_quality"],0),1))