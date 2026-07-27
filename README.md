# AI Debate Analyzer

> **An end-to-end NLP + Generative AI system that evaluates debates using Transformer models, Retrieval-Augmented Generation (RAG), and Large Language Models.**

---

# Overview

AI Debate Analyzer is an intelligent debate evaluation platform that analyzes debates between two speakers and provides both **quantitative scores** and **qualitative feedback**.

Unlike traditional text classification projects, this system combines **classical NLP, Deep Learning, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs)** to deliver comprehensive debate analysis.

The application accepts structured debate rounds consisting of:

* Opening Statements
* Rebuttals
* Closing Statements

For every submission, the system:

* Evaluates argument quality using a fine-tuned DeBERTa model
* Retrieves relevant factual information from Wikipedia
* Generates contextual feedback using an LLM
* Produces detailed diagnostics and overall scores
* Displays everything through an interactive web interface

---

# Features

### NLP Evaluation

* Argument quality classification
* Confidence scoring
* Transformer-based inference
* Fine-tuned DeBERTa model

### Generative AI

* AI-generated personalized feedback
* Strengths identification
* Weakness detection
* Suggestions for improvement

### Retrieval-Augmented Generation (RAG)

* Retrieves contextual information from Wikipedia
* Grounds LLM responses with factual evidence
* Reduces hallucinations

### Interactive Dashboard

* Speaker-wise evaluation
* Quality score visualization
* Diagnostic reports
* AI feedback panel

---

# Project Architecture

```
                    User Debate

                         │

                         ▼

               Frontend (Next.js)

                         │

                         ▼

              FastAPI Backend API

                         │

          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼

   DeBERTa Evaluation          Wikipedia Retrieval
        Model                     (RAG Layer)

          │                             │
          └──────────────┬──────────────┘
                         ▼

                LLM Feedback Generator

                         ▼

                Final Debate Report

                         ▼

                  Frontend Display
```

---

# Tech Stack

## Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

---

## Backend

* FastAPI
* Python
* Uvicorn

---

## Machine Learning

* PyTorch
* Hugging Face Transformers
* DeBERTa-v3
* PEFT
* LoRA

---

## NLP

* Sentence Transformers
* Tokenization
* Semantic embeddings
* Text preprocessing

---

## Generative AI

* OpenAI GPT
* Prompt Engineering

---

## Retrieval

* Wikipedia API
* Retrieval-Augmented Generation (RAG)

---

## Deployment

* Docker (optional)
* Render / Railway / Azure
* Vercel (Frontend)

---

# Folder Structure

```
AI-Debate-Analyzer/

│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   └── package.json
│
├── ml-service/
│   ├── src/
│   │
│   ├── inference/
│   ├── models/
│   ├── rag/
│   ├── prompts/
│   ├── utils/
│   ├── app.py
│   └── requirements.txt
│
├── datasets/
│
├── notebooks/
│
├── README.md
│
└── LICENSE
```

---

# Machine Learning Pipeline

```
Raw Debate

      │

      ▼

Text Preprocessing

      │

      ▼

Tokenizer (DeBERTa)

      │

      ▼

Fine-tuned DeBERTa

      │

      ▼

Quality Classification

      │

      ▼

Confidence Score
```

---

# RAG Pipeline

```
Debate Topic

      │

      ▼

Wikipedia Search

      │

      ▼

Relevant Context Retrieval

      │

      ▼

Prompt Construction

      │

      ▼

LLM

      │

      ▼

Grounded Feedback
```

---

# LLM Feedback Pipeline

The retrieved Wikipedia context is combined with:

* Debate transcript
* Model predictions
* Confidence scores
* Speaker information

The LLM then generates:

* Overall feedback
* Argument analysis
* Logical consistency
* Evidence quality
* Suggestions
* Final summary

---

# Model Details

### Base Model

```
microsoft/deberta-v3-base
```

---

### Fine-Tuning Method

* LoRA
* PEFT

---

### Framework

* Hugging Face Transformers
* PyTorch

---

### Output

The model predicts the debate quality category and returns a confidence score.

---

# Input Format

Example:

```
Topic:
Should AI Replace Teachers?

Speaker A Opening:
AI can personalize education and improve accessibility...

Speaker B Opening:
Teachers provide emotional intelligence and mentorship...

Speaker A Rebuttal:
While empathy matters, AI can assist...

Speaker B Rebuttal:
Technology cannot fully understand students...

Speaker A Closing:
AI should support education.

Speaker B Closing:
Teachers remain irreplaceable.
```

---

# Output Example

```
Speaker A

Quality Score:
87/100

Confidence:
94%

Strengths
✔ Logical reasoning
✔ Clear structure

Weaknesses
✘ Limited supporting evidence

Suggestions
• Include statistical evidence
• Address opposing arguments

--------------------------------

Speaker B

Quality Score:
81/100

Confidence:
91%

Strengths
✔ Strong emotional appeal

Weaknesses
✘ Needs stronger factual support

Suggestions
• Cite educational research
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/DEVAYUSHOPS/AI-Debate-Analyzer.git

cd AI-Debate-Analyzer
```

---

# Backend Setup

```bash
cd ml-service

python -m venv .venv

source .venv/bin/activate

# Windows

.venv\Scripts\activate

pip install -r requirements.txt
```

Run:

```bash
uvicorn app:app --reload
```

---

# Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

---

# Environment Variables

Create:

```
ml-service/.env
```

Example:

```env
OPENAI_API_KEY=your_api_key
```

---

# API Endpoint

```
POST /analyze
```

Request:

```json
{
  "topic": "...",
  "speakerA": {
    "opening": "...",
    "rebuttal": "...",
    "closing": "..."
  },
  "speakerB": {
    "opening": "...",
    "rebuttal": "...",
    "closing": "..."
  }
}
```

---

# Future Improvements

* Multi-turn debate support
* Real-time debate analysis
* Speech-to-text integration
* Audio debate evaluation
* Bias detection
* Fallacy detection
* Citation verification
* Custom knowledge base
* Multi-language support
* Debate ranking leaderboard

---

# Learning Outcomes

This project demonstrates practical experience in:

* Natural Language Processing (NLP)
* Transformer-based text classification
* Fine-tuning large language models with LoRA/PEFT
* Retrieval-Augmented Generation (RAG)
* Prompt Engineering
* REST API development using FastAPI
* Frontend integration with Next.js
* End-to-end ML deployment
* Model inference optimization
* Building production-ready AI applications
---

# License

This project is licensed under the MIT License.

---

# Authors

**Ayush (Frontend)**
**Yuvraj (ML and AI)**

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## Acknowledgements

This project builds upon the open-source ecosystem, including:

* Hugging Face Transformers
* PyTorch
* Microsoft DeBERTa
* OpenAI API
* Wikipedia API
* FastAPI
* Next.js
* React
* Tailwind CSS
