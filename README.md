# Smart Microblog Privacy Guard

A simple privacy-focused microblogging app that scans user posts for PII, assigns a risk level, warns before publishing, and stores posts in PostgreSQL.

## Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- PostgreSQL database server
- Git

## Backend setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file:

   ```bash
   copy .env.example .env
   ```

5. Edit `.env` and set your local PostgreSQL connection:

   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/smart_microblog
   ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
   ```

6. Create the database in PostgreSQL if needed:

   ```sql
   CREATE DATABASE smart_microblog;
   ```

7. Install the required spaCy model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

8. Start the backend:

   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

   If you are running from inside the backend folder directly, use:

   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Frontend setup

1. In a second terminal, go to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install frontend dependencies:

   ```bash
   npm install
   ```

3. Create a frontend environment file if needed:

   ```bash
   copy .env.example .env
   ```

4. Start the app:

   ```bash
   npm run dev
   ```

## Expected development URLs

- Backend: http://127.0.0.1:8000
- Frontend: http://127.0.0.1:5173

## Common troubleshooting

### `DATABASE_URL is not configured`

- Make sure `.env` exists in the project root.
- Ensure `DATABASE_URL` is set correctly.
- Check that the PostgreSQL server is running.

### `The spaCy model 'en_core_web_sm' is missing`

Run:

```bash
python -m spacy download en_core_web_sm
```

### `CORS policy blocked` errors

- Ensure `ALLOWED_ORIGINS` includes the frontend origin.
- Verify both frontend and backend are running on the URLs above.

### `Could not reach the backend` in the browser

- Confirm the backend is running with `uvicorn`.
- Check the frontend API base URL in `frontend/.env` or the default in `frontend/src/api/client.js`.

## Project structure overview

- `backend/` — FastAPI app, PII detection, risk scoring, DB layer
- `frontend/` — Vite + React app
- `research_paper_evidence/` — dataset/model evaluation scripts

## Notes

- The application intentionally keeps setup minimal and does not add auth, deployment, or extra product features.
- Do not commit real credentials or secrets into the repository.
- Use `.env` locally only; `.env.example` is the safe template.

This project is currently in a reliability-first milestone: local setup, configuration validation, database safety, and test coverage are the immediate goals before additional product work.