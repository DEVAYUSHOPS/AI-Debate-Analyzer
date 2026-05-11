# AI Debate Analyzer Frontend

Next.js frontend for the AI Debate Analyzer. It lets users enter debate rounds, submit transcripts for analysis, view speaker scores, inspect generated feedback, browse saved debates, and download reports.

This app is designed to run separately from the ML backend in `../ml-service`.

## Tech Stack

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS
- MongoDB with Mongoose
- Recharts and React Flow for visualizations
- jsPDF for report export

## Project Structure

```txt
frontend/
  app/                  Next.js app router pages and API routes
  app/api/analyze/      Calls the FastAPI ML service and saves results
  app/api/debates/      Lists saved debates
  app/api/debate/[id]/  Loads a saved debate by ID
  components/           UI components
  hooks/                Debate timer and debate engine hooks
  lib/                  Shared config and MongoDB connection
  models/               Mongoose models
  types/                Shared TypeScript types
```

## Environment Variables

Create `frontend/.env.local` for local development:

```env
MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/ai_debate_analyzer
ML_SERVICE_URL=http://localhost:8000
```

For production, `ML_SERVICE_URL` must point to the deployed FastAPI service:

```env
ML_SERVICE_URL=https://your-ml-service.onrender.com
```

Do not commit `.env.local`.

## Local Development

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open:

```txt
http://localhost:3000
```

The ML service should also be running at `http://localhost:8000`, or the `/api/analyze` route will fail.

## Available Scripts

```bash
npm run dev      # Start local Next.js dev server
npm run build    # Build production app
npm run start    # Start production server after build
npm run lint     # Run ESLint
```

## Deployment

Deploy this folder to Vercel.

Use these settings:

```txt
Root Directory: frontend
Install Command: npm install
Build Command: npm run build
```

Add these Vercel environment variables:

```env
MONGODB_URI=your_mongodb_atlas_connection_string
ML_SERVICE_URL=https://your-render-ml-service-url.onrender.com
```

## Notes

- The frontend does not run the ML model directly.
- The Next.js API route `app/api/analyze/route.ts` forwards debate data to the FastAPI backend.
- Saved debates are stored in MongoDB through `lib/db.ts` and `models/Debate.ts`.
