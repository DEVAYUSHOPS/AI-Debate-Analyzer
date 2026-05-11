# AI Debate Analyzer ML Service

FastAPI backend for the AI Debate Analyzer. This service runs the NLP pipeline, retrieves evidence, generates debate coaching feedback, and returns structured analysis to the Next.js frontend.

## Features

- Multitask DeBERTa-based argument analysis
- Argument quality scoring
- Component classification: `MajorClaim`, `Claim`, `Premise`
- Topic-aware stance detection: `PRO` or `CON`
- Rule-based fallacy detection
- Retrieval-augmented generation using Wikipedia, academic retrieval, embeddings, ChromaDB, and FAISS
- Gemini-powered debate coaching feedback
- Full debate feedback for multi-round speaker comparisons
- RLAIF-style logging of interactions for later review

## Tech Stack

- FastAPI
- Uvicorn
- PyTorch
- Hugging Face Transformers
- PEFT/LoRA
- Sentence Transformers
- ChromaDB and FAISS
- Google Gemini via `google-genai`
- Streamlit demo app in `app.py`

## Project Structure

```txt
ml-service/
  api.py                         FastAPI application
  app.py                         Optional Streamlit demo UI
  Dockerfile                     Docker deployment config
  requirements.txt               Python dependencies
  src/inference/                 Model loading and prediction
  src/rag/                       Retrieval and LLM feedback pipeline
  src/evaluation/debate_model.pt Local model path, ignored by Git
  src/db_service.py              RLAIF interaction logging
  chroma_db/                     Local vector database files
```

## Environment Variables

Create `ml-service/.env` locally:

```env
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_hugging_face_token
HF_MODEL_REPO=DEVAYUSHOPS/ai-debate-analyzer-model
HF_MODEL_FILENAME=debate_model.pt
```

`HF_MODEL_REPO` is used when `src/evaluation/debate_model.pt` is not available locally. The service downloads the model from Hugging Face at startup.

Do not commit `.env`.

## Model File

The API expects a trained model at:

```txt
src/evaluation/debate_model.pt
```

Because model files are large, `*.pt` is ignored by Git. For deployment, upload the model to Hugging Face and set:

```env
HF_MODEL_REPO=DEVAYUSHOPS/ai-debate-analyzer-model
HF_MODEL_FILENAME=debate_model.pt
```

The backend will use the local model if it exists. Otherwise, it downloads the model with `huggingface_hub`.

## Local Development

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI service:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Open:

```txt
http://localhost:8000
```

FastAPI docs are available at:

```txt
http://localhost:8000/docs
```

## Docker

Build the image:

```bash
docker build -t ai-debate-ml .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env ai-debate-ml
```

The Dockerfile starts the app with:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

## API Endpoints

### `GET /`

Health check.

### `POST /analyze`

Analyze a single argument.

Example request:

```json
{
  "topic": "Schools should ban smartphones",
  "text": "Research indicates that smartphone use during instructional time reduces academic performance."
}
```

### `POST /student-feedback`

Return student-facing rubric feedback for one argument.

Example request:

```json
{
  "topic": "Schools should ban smartphones",
  "text": "Smartphones distract students during class.",
  "student_name": "Candidate"
}
```

### `POST /debate-feedback`

Analyze a full debate with named speakers and rounds. This is the endpoint used by the Next.js frontend.

Example request:

```json
{
  "topic": "Schools should ban smartphones",
  "speakerA": "Alex",
  "speakerB": "Riya",
  "mode": "text",
  "rounds": [
    {
      "round": "Opening",
      "speakerA": "Phones distract students and reduce focus.",
      "speakerB": "Phones can support learning when used responsibly."
    }
  ]
}
```

## Deployment

Deploy this folder to Render as a Docker web service.

Use these settings:

```txt
Root Directory: ml-service
Runtime: Docker
```

Add these Render environment variables:

```env
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_hugging_face_token
HF_MODEL_REPO=DEVAYUSHOPS/ai-debate-analyzer-model
HF_MODEL_FILENAME=debate_model.pt
```

After deployment, set the frontend's `ML_SERVICE_URL` to the Render service URL.

## Troubleshooting

If you see an error from `/usr/local/bin/uvicorn`, make sure the deployed Dockerfile uses:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

If the service cannot find `debate_model.pt`, confirm that the model exists in the Hugging Face repo and that `HF_TOKEN`, `HF_MODEL_REPO`, and `HF_MODEL_FILENAME` are set correctly.

If Docker builds are slow, that is expected. PyTorch, Transformers, ChromaDB, and FAISS make this image large.
