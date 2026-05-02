# 🧠 AI Debate Analyzer – ML Service

An AI-powered argument evaluation system that combines **NLP, Retrieval-Augmented Generation (RAG), and Generative AI** to analyze debate arguments and provide structured coaching feedback.

This service performs argument analysis using a fine-tuned transformer model, retrieves supporting evidence, and generates human-like feedback grounded in factual context.

---

## 🔥 Key Features

- Multitask NLP model (DeBERTa + LoRA)
- Argument quality scoring (regression)
- Component classification (Claim / Premise / MajorClaim)
- Topic-aware stance detection (PRO / CON)
- Rule-based fallacy detection
- Multi-source RAG pipeline (Wikipedia + academic retrieval)
- Query rewriting for improved retrieval
- Rubric-based evaluation system
- LLM-powered coaching feedback
- Retrieval debugging and transparency
- RLAIF-style feedback loop for continuous improvement

---

## 🧠 System Architecture

```text
User Input
   ↓
DeBERTa Model (quality, stance, component)
   ↓
Query Rewriting (intent-aware)
   ↓
Multi-query Expansion
   ↓
Multi-source Retrieval
   → Wikipedia (general knowledge)
   → Academic sources (Semantic Scholar)
   ↓
Deduplication + MMR Filtering
   ↓
Structured Context
   ↓
LLM (Feedback Generation)
   ↓
Rubric Scoring + Final Output
```

This modular design separates prediction, retrieval, and reasoning layers for better interpretability and extensibility.

---

## 🧠 ML Model Architecture

The core model is a **multitask DeBERTa-v3-base transformer** fine-tuned using **LoRA (PEFT)**.

### Tasks:

- Argument Quality (regression)
- Component Detection (classification)
- Stance Detection (classification)

### Training Strategy:

- Task-specific prefixes:

  ```
  [QUALITY] argument
  [COMPONENT] argument
  [STANCE] topic + argument
  ```

- Shared encoder + task-specific heads
- Weighted loss for imbalance handling

### Output Example:

```json
{
  "argument_quality": 0.695,
  "component": "Claim",
  "stance": "CON",
  "fallacy": "None"
}
```

---

## 🔍 RAG Pipeline

The system uses an **enhanced RAG pipeline** to improve factual grounding:

### Steps:

1. Query rewriting based on argument intent (research / policy / general)
2. Multi-query expansion for broader coverage
3. Multi-source retrieval:
   - Wikipedia (general knowledge)
   - Semantic Scholar (research evidence)

4. Deduplication of retrieved chunks
5. Embedding-based ranking
6. Diversity-aware filtering (MMR-style)
7. Structured context generation

This ensures:

- Better evidence retrieval
- Reduced hallucination
- Improved reasoning quality

---

## 🧠 Query Intelligence Layer

Raw debate arguments are not optimal for retrieval.

The system includes a **query rewriting module** that:

- Extracts key terms
- Detects argument intent (research vs policy vs general)
- Generates retrieval-friendly queries

This significantly improves performance for research-heavy arguments.

---

## 📊 Rubric-Based Evaluation

Beyond model predictions, the system evaluates arguments using a **custom rubric**:

- Argument Quality
- Evidence Usage
- Logical Reasoning
- Clarity
- Rebuttal Readiness

This enables multi-dimensional assessment instead of relying on a single score.

---

## 🧠 Handling Ambiguity

The system does not assume all arguments are fully supported.

It classifies factual grounding as:

- Supported
- Partially supported
- Unsupported

This reflects real-world uncertainty and improves reliability.

---

## 🤖 LLM Feedback Generation

A generative AI model (Gemini) is used to:

- Provide structured coaching feedback
- Explain strengths and weaknesses
- Suggest improvements
- Rewrite arguments

Important:

> The LLM is used for interpretation, not prediction.
> Core scoring is handled by the trained model and retrieval pipeline.

---

## 🧪 Example Input

```json
{
  "topic": "Schools should ban smartphones",
  "text": "Research indicates that smartphone use during instructional time reduces academic performance."
}
```

---

## 📦 API Endpoints

### `/analyze`

Returns:

- Model predictions
- Retrieved context
- LLM feedback

### `/student-feedback`

Returns:

- Model predictions
- Rubric scores
- Structured coaching feedback

---

## 🧠 RLAIF Feedback Loop

The system logs difficult cases where:

- Model confidence is high
- But LLM identifies weaknesses

These are stored and later used for retraining.

### Flow:

1. User input → model prediction
2. LLM critique
3. Hard cases logged
4. Converted into training data
5. Improves future performance

---

## 📊 Evaluation Metrics

| Task                | Metric              | Score |
| ------------------- | ------------------- | ----- |
| Argument quality    | Pearson correlation | ~0.65 |
| Component detection | Macro F1            | ~0.78 |
| Stance detection    | Macro F1            | ~0.82 |

---

## ⚙️ Tech Stack

### Core:

- PyTorch
- Hugging Face Transformers
- SentenceTransformers

### Backend:

- FastAPI

### Frontend:

- Streamlit

### Retrieval:

- ChromaDB
- Wikipedia API
- Semantic Scholar API

### LLM:

- Google Gemini

---

## ⚠️ Limitations

- RAG coverage depends on available sources
- Academic retrieval is query-sensitive
- LLM feedback depends on API quota
- Stance model requires correction layer for negation

---

## 🚀 Key Contributions

- Designed a multitask NLP system for argument analysis
- Built a query-aware multi-source RAG pipeline
- Integrated academic retrieval for evidence grounding
- Developed a rubric-based evaluation framework
- Implemented LLM-based coaching feedback
- Added RLAIF-style learning loop

---

## 🧠 Project Type

This project is a **research-oriented prototype**, demonstrating:

- NLP modeling
- Retrieval systems
- LLM integration
- End-to-end AI pipeline design

---

## 🎯 Summary

This system combines:

- **NLP (transformers)**
- **RAG (retrieval + grounding)**
- **Generative AI (LLM feedback)**

into a complete **argument evaluation and coaching system**.

---
