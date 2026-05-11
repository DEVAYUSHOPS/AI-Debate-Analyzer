# Deployment

This project deploys as two services:

- `frontend/`: Next.js app, deploy to Vercel.
- `ml-service/`: FastAPI ML API, deploy to Render as a Docker web service.

## 1. ML Service on Render

Create a new Render Web Service from this repo.

Use these settings:

```txt
Root Directory: ml-service
Runtime: Docker
```

Add these environment variables in Render:

```env
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_hugging_face_token
```

The Dockerfile starts the API with:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

After deploy, Render will give you a URL like:

```txt
https://your-ml-service.onrender.com
```

## 2. Frontend on Vercel

Create a new Vercel project from this repo.

Use these settings:

```txt
Root Directory: frontend
Build Command: npm run build
Install Command: npm install
```

Add these environment variables in Vercel:

```env
MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/ai_debate_analyzer
ML_SERVICE_URL=https://your-ml-service.onrender.com
```

Do not use `http://localhost:8000` in production. That only works on your computer.

## 3. Important Model File Note

The backend needs this file at startup:

```txt
ml-service/src/evaluation/debate_model.pt
```

The repository currently ignores `*.pt` files in `.gitignore`, so this model may not be uploaded to GitHub.

If Render cannot find `debate_model.pt`, use one of these:

1. Git LFS for `debate_model.pt`.
2. Upload the model to Hugging Face and download it during deployment.
3. Keep the model inside the Docker build context if your Git host includes it.

For a first deployment, Git LFS is usually the simplest path.

## 4. Local Docker Test

Start Docker Desktop first, then run:

```powershell
cd "e:\Projects Web Dev\ai_debate_analyzer\ml-service"
docker build -t ai-debate-ml .
docker run -p 8000:8000 --env-file .env ai-debate-ml
```

Open:

```txt
http://localhost:8000
```
