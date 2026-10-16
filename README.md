# Cineverse — AI Movie Discovery

A movie discovery app that pairs the repository's existing CountVectorizer/cosine-similarity model with a FastAPI API, TMDB movie data, an OpenAI tool-calling assistant, and a responsive React interface.

## Architecture

- `frontend/`: React + Vite + Tailwind UI. Its API client calls the backend, and the Node static server reads `VITE_API_URL` at runtime.
- `backend/main.py`: FastAPI routes, Pydantic validation, CORS, health check, and a basic per-IP chat rate limit.
- `backend/recommendations.py`: Loads the existing `movie.pkl` DataFrame, normalizes missing fields, builds one reusable sparse CountVectorizer representation, and calculates only the requested movie's cosine similarities.
- `backend/tmdb.py`: Async TMDB access with request timeouts, error mapping, and a bounded in-memory metadata cache.
- `backend/chat.py`: OpenAI tool calling for TMDB search/discovery/details and the local similarity model. Conversation turns are retained in bounded in-process session memory.

## Run locally

1. Use Python 3.11+ and Node.js 20+.
2. Copy `.env.example` to `.env`, then set `TMDB_API_KEY` and `OPENAI_API_KEY`. The local ML recommendations work without either key.
3. Install backend dependencies and start the API from the repository root:

   ```powershell
   python -m pip install -r requirements-dev.txt
   uvicorn backend.main:app --reload --port 8000
   ```

4. In another terminal:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

5. Visit `http://localhost:5173`. The API health endpoint is `http://localhost:8000/health`.

Set `FRONTEND_ORIGINS` to a comma-separated list of allowed frontend origins. Do not use `*` for a deployed app.

## API

- `GET /health`
- `GET /api/recommendations/{movie_title}` — up to 10 local ML recommendations and similarity scores.
- `GET /api/movies/catalog?q=...` — title suggestions from the bundled local dataset.
- `GET /api/movies/trending`
- `GET /api/movies/search?q=...`
- `GET /api/movies/discover?genre_id=...&year_after=...&year_before=...&min_rating=...`
- `GET /api/genres`
- `GET /api/movies/{movie_id}` — TMDB details and a trailer URL when available.
- `POST /api/chat` — `{ "message": "...", "session_id": "optional-session-id" }`.

Movie-card ratings are explicitly labeled as TMDB ratings. IMDb ratings are not inferred.

## Railway deployment (not yet deployed)

Deploy the frontend and backend as **two separate Railway services** from the same repository. No deployment has been created or authorized by this implementation.

### Backend service

1. Create a Railway project and add a service from this repository and the `feature/ai-movie-chatbot` branch.
2. Leave the service root directory at the repository root. In service build settings, select the Dockerfile builder and set the Dockerfile path to `/backend/Dockerfile`.
3. Set `OPENAI_API_KEY`, `TMDB_API_KEY`, and `FRONTEND_ORIGINS` to the deployed frontend's HTTPS origin. Optionally set `OPENAI_MODEL`. The Docker image bundles `movie.pkl`; no machine-specific file path is required.
4. Set the health check path to `/health`, deploy, and generate a public domain. Railway supplies the port through `$PORT`.

### Frontend service

1. Add a second service for the same repository and branch.
2. Set its root directory to `/frontend`, select the Dockerfile builder, and set the Dockerfile path to `/Dockerfile`.
3. Set `VITE_API_URL` to the backend service's public HTTPS origin (no trailing slash), then deploy and generate a public domain. The frontend server injects this runtime value; no API key is included in the browser bundle.
4. Update the backend `FRONTEND_ORIGINS` to the final frontend origin and redeploy the backend.

The legacy TMDB API key was embedded in the old Streamlit source on the feature branch. It has been removed from the replacement implementation, but removing it from the current source does not erase Git history: **rotate/revoke that key in TMDB before deploying** and review repository history if the branch was shared.

## Tests and builds

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
cd frontend
npm install
npm run build
```

The AI chat and live TMDB discovery require valid provider keys. Their missing-key and validation responses are covered by local API tests; actual provider calls require credentials and network access.
