# AI Debate Analyzer - ML Service

This folder contains the machine learning backend for the AI Debate Analyzer project. The ML service analyzes debate arguments using a multitask NLP model, retrieves factual context through a lightweight RAG pipeline, and generates coaching-style feedback with an LLM.

The current system is a working research prototype. It is suitable for demonstrating the ML pipeline, inference flow, stance handling, retrieval debugging, and end-to-end API behavior.

## What This Service Does

Given a debate topic and an argument, the service predicts:

- Argument quality score between 0 and 1
- Argument component type: `MajorClaim`, `Claim`, or `Premise`
- Stance relative to the debate topic: `PRO` or `CON`
- Simple logical fallacy label
- Factual context retrieved from ChromaDB or Wikipedia
- LLM-generated debate feedback

Example input:

```json
{
  "topic": "Schools should ban smartphones",
  "text": "Schools should not completely ban smartphones because students may need them for emergency communication, accessibility tools, and quick academic research."
}
```

Example prediction:

```json
{
  "argument_quality": 0.695,
  "component": "Claim",
  "stance": "CON",
  "fallacy": "None",
  "raw_stance": "PRO",
  "stance_reason": "topic_negation_conflict:ban,schools,smartphones"
}
```

## ML Architecture

The core model is a multitask DeBERTa-v3-base model with LoRA fine-tuning.

It has one shared encoder and three task-specific heads:

- Regression head for argument quality
- Classification head for argument component detection
- Classification head for stance detection

The model uses task prefixes during training and inference:

```text
[QUALITY] argument text
[COMPONENT] argument text
[STANCE] topic: debate topic argument: argument text
```

The stance output includes both:

- `raw_stance`: direct neural model output
- `stance`: final topic-aware stance after correction

The correction layer handles explicit topic opposition patterns such as:

```text
Topic: Schools should ban smartphones
Argument: Schools should not ban smartphones...
```

In such cases, the raw model may predict `PRO`, but the corrected stance becomes `CON`.

## RAG Pipeline

The RAG layer improves factual grounding before feedback generation.

Flow:

1. Build a compact retrieval query from the topic and argument keywords.
2. Search the local ChromaDB cache.
3. Validate cache relevance using topic-specific lexical overlap.
4. Reject misleading cache hits, such as school-uniform context for a smartphone debate.
5. Fall back to Wikipedia search if local context is missing or irrelevant.
6. Cache useful Wikipedia summaries for future requests.
7. Pass retrieved context to Gemini for feedback generation.

The API also returns `retrieval_debug` so the retrieval behavior can be inspected during testing.

Example debug fields:

```json
{
  "source": "wikipedia",
  "cache_hit": false,
  "cache_rejected": true,
  "cache_relevance_reason": "missing_required_topic_terms:cellphone,mobile,phone,smartphone",
  "wikipedia_title": "Mobile phone use in schools"
}
```

## Project Structure

```text
ml-service/
  api.py                         FastAPI backend
  app.py                         Streamlit demo UI
  requirements.txt               Python dependencies
  extract_rlaif_data.py          Converts logged feedback into retraining data
  rlaif_waiting_room.db          SQLite database for hard examples

  src/
    inference/
      inference.py               Model loading and prediction
      fallacy_detector.py        Rule-based fallacy detector

    models/
      train.py                   DeBERTa + LoRA multitask training script
      utils.py                   Task-balanced sampling helper

    evaluation/
      evaluate.py                Evaluation script
      debate_model.pt            Model weights, not tracked by git

    rag/
      query_expansion.py         Keyword extraction and retrieval query building
      retriever.py               ChromaDB + Wikipedia retrieval
      rag_pipeline.py            Context building and filtering
      llm_feedback.py            Gemini feedback generation
      filtering.py               Semantic filtering utilities

    db_service.py                RLAIF hard-negative logging

  notebooks/
    preprocessing.ipynb          Dataset preprocessing
    *_eda.ipynb                  Dataset exploration notebooks
    data/train                   Hugging Face training dataset
    data/val                     Hugging Face validation dataset
    data/test                    Hugging Face test dataset
```

## Setup

From PowerShell:

```powershell
cd E:\AI-Debate-Analyzer\ml-service

# Activate the existing virtual environment
..\debate_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Make src imports work
$env:PYTHONPATH = (Get-Location).Path
```

Optional environment variables:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key"
$env:HF_TOKEN = "your_huggingface_token"
```

`GEMINI_API_KEY` is required only for LLM feedback. The DeBERTa prediction path can still run without Gemini.

## Run the FastAPI Backend

```powershell
cd E:\AI-Debate-Analyzer\ml-service
..\debate_env\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Test the API

```powershell
$body = @{
  topic = "Schools should ban smartphones"
  text = "Schools should ban smartphones because they distract students during lessons and reduce classroom attention."
} | ConvertTo-Json

$res = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/analyze" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

$res.prediction
$res.context
$res.retrieval_debug
$res.llm_feedback
```

## Student Feedback Endpoint

Use `/student-feedback` when you want candidate/student performance feedback instead of only single-argument diagnostics.

```powershell
$body = @{
  student_name = "Aarav"
  topic = "Schools should ban smartphones"
  text = "Schools should ban smartphones because they distract students during lessons and reduce classroom attention."
} | ConvertTo-Json

$res = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/student-feedback" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

$res.prediction
$res.rubric_scores
$res.student_feedback
```

This endpoint returns:

- `prediction`: model quality, component, stance, and fallacy output
- `rubric_scores`: overall, argument quality, evidence usage, reasoning, clarity, and rebuttal readiness
- `context`: retrieved factual context
- `retrieval_debug`: RAG trace for debugging
- `student_feedback`: Markdown feedback report for the student
- `feedback_source`: `gemini` or `fallback`

If Gemini quota is exhausted, this endpoint still returns fallback feedback based on the model and rubric scores.

## Run the Streamlit Demo

Start FastAPI first, then in another terminal:

```powershell
cd E:\AI-Debate-Analyzer\ml-service
..\debate_env\Scripts\Activate.ps1

streamlit run app.py
```

The Streamlit app accepts:

- Debate topic or motion
- User argument

It displays:

- Quality score
- Component type
- Stance
- LLM feedback

## Run Direct Terminal Inference

```powershell
cd E:\AI-Debate-Analyzer\ml-service
..\debate_env\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

python -m src.inference.inference
```

## Training

```powershell
cd E:\AI-Debate-Analyzer\ml-service
..\debate_env\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

python src\models\train.py
```

The training script:

- Loads preprocessed Hugging Face datasets from `notebooks/data`
- Adds task prefixes
- Fine-tunes DeBERTa-v3-base using LoRA
- Uses weighted losses for imbalanced tasks
- Saves the best checkpoint as `debate_model.pt`

If training saves `debate_model.pt` in the service root, copy it to the evaluation folder used by inference:

```powershell
Copy-Item .\debate_model.pt .\src\evaluation\debate_model.pt
```

## Evaluation

```powershell
cd E:\AI-Debate-Analyzer\ml-service
..\debate_env\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

python src\evaluation\evaluate.py
```

Evaluation reports:

- Pearson correlation for argument quality
- Macro F1 for component classification
- Macro F1 for stance classification

Fill in the table below after running evaluation on the final checkpoint:

| Task | Metric | Score |
| --- | --- | --- |
| Argument quality | Pearson correlation | TBD |
| Component detection | Macro F1 | TBD |
| Stance detection | Macro F1 | TBD |

## RLAIF Feedback Loop

The service logs hard examples into SQLite when Gemini identifies weaknesses in cases where the DeBERTa quality score was relatively high.

Flow:

1. User submits an argument.
2. DeBERTa predicts quality/component/stance.
3. Gemini provides factual and structural critique.
4. If the model appears overconfident, the interaction is logged.
5. `extract_rlaif_data.py` converts logged examples into extra training data.
6. Future training can merge this dataset into the main training set.

Run extraction:

```powershell
cd E:\AI-Debate-Analyzer\ml-service
..\debate_env\Scripts\Activate.ps1

python extract_rlaif_data.py
```

## Known Limitations

- The raw stance head can overpredict `PRO`; the current system uses a transparent topic-aware correction layer.
- RAG currently relies on ChromaDB and Wikipedia, so factual coverage is limited by retrieved pages.
- Gemini feedback depends on API quota and may return a quota error on the free tier.
- The Next.js frontend currently needs full integration with this FastAPI service.
- Final model metrics should be updated after running `evaluate.py` on the latest checkpoint.

## Interview Summary

This ML service demonstrates a complete NLP prototype:

- Multitask transformer modeling
- LoRA/PEFT fine-tuning
- Argument quality scoring
- Component and stance classification
- Topic-aware stance correction
- RAG-based factual grounding
- LLM-based debate coaching
- Retrieval debugging
- RLAIF-style hard-example logging

The project is best described as a working research prototype rather than a production-ready system.
