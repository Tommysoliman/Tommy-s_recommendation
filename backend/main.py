import logging
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Path as PathParameter, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from backend import chat
from backend.config import FRONTEND_ORIGINS, MOVIE_DATA_PATH, OPENAI_API_KEY, TMDB_API_KEY
from backend.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Movie,
    RecommendationResponse,
    WhereToWatch,
    movie_from_tmdb,
)
from backend.recommendations import MovieRecommender
from backend.tmdb import TMDBClient


logging.basicConfig(level=logging.INFO)
# httpx logs full request URLs, which include the TMDB api_key query parameter.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cineverse Movie Discovery API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

tmdb = TMDBClient()
_chat_requests: dict[str, deque[float]] = defaultdict(deque)
_CHAT_REQUEST_LIMIT = 10
_CHAT_WINDOW_SECONDS = 60


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    try:
        return MovieRecommender(MOVIE_DATA_PATH)
    except (OSError, ValueError) as error:
        logger.exception("Unable to load the local movie recommendation dataset.")
        raise HTTPException(status_code=503, detail="The movie recommendation dataset is unavailable.") from error


async def check_chat_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    recent_requests = _chat_requests[client_host]
    while recent_requests and now - recent_requests[0] >= _CHAT_WINDOW_SECONDS:
        recent_requests.popleft()
    if len(recent_requests) >= _CHAT_REQUEST_LIMIT:
        raise HTTPException(status_code=429, detail="Chat request limit reached. Please wait before trying again.")
    recent_requests.append(now)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        dataset_loaded = get_recommender().loaded
    except HTTPException:
        dataset_loaded = False
    return HealthResponse(
        status="ok" if dataset_loaded else "degraded",
        movie_dataset_loaded=dataset_loaded,
        tmdb_configured=bool(TMDB_API_KEY),
        openai_configured=bool(OPENAI_API_KEY),
    )


@app.get("/api/recommendations/{movie_title}", response_model=RecommendationResponse)
async def recommendations(movie_title: str) -> RecommendationResponse:
    try:
        matches = get_recommender().recommend(movie_title, limit=10)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return RecommendationResponse(
        query=movie_title,
        recommendations=[Movie(**match) for match in matches],
    )


@app.get("/api/movies/catalog", response_model=list[Movie])
async def local_movie_catalog(
    q: str = Query(min_length=1, max_length=100),
) -> list[Movie]:
    return [Movie(**movie) for movie in get_recommender().search(q)]


@app.get("/api/movies/trending", response_model=list[Movie])
async def trending_movies() -> list[Movie]:
    return [movie_from_tmdb(result) for result in await tmdb.trending()]


@app.get("/api/movies/search", response_model=list[Movie])
async def search_movies(
    q: str = Query(min_length=1, max_length=100),
) -> list[Movie]:
    return [movie_from_tmdb(result) for result in await tmdb.search(q)]


@app.get("/api/movies/discover", response_model=list[Movie])
async def discover_movies(
    genre_id: int | None = Query(default=None, ge=1),
    year_after: int | None = Query(default=None, ge=1888, le=2100),
    year_before: int | None = Query(default=None, ge=1888, le=2100),
    min_rating: float | None = Query(default=None, ge=0, le=10),
) -> list[Movie]:
    if year_after is not None and year_before is not None and year_after >= year_before:
        raise HTTPException(status_code=422, detail="year_after must not be later than year_before.")
    results = await tmdb.discover(
        genre_id=genre_id,
        year_after=year_after,
        year_before=year_before,
        min_rating=min_rating,
    )
    return [movie_from_tmdb(result) for result in results]


@app.get("/api/genres")
async def movie_genres() -> list[dict[str, Any]]:
    return await tmdb.genres()


@app.get("/api/movies/{movie_id}", response_model=Movie)
async def movie_details(
    movie_id: int = PathParameter(..., ge=1),
    region: str = Query("US", pattern=r"^[A-Za-z]{2}$"),
) -> Movie:
    data = await tmdb.details(movie_id)
    movie = movie_from_tmdb(data)
    country = region.upper()
    providers = (data.get("watch/providers") or {}).get("results", {}).get(country)
    if providers:
        names = lambda key: [p["provider_name"] for p in providers.get(key, []) if p.get("provider_name")]
        movie.where_to_watch = WhereToWatch(
            region=country,
            link=providers.get("link"),
            stream=names("flatrate"),
            rent=names("rent"),
            buy=names("buy"),
        )
    videos = data.get("videos", {}).get("results", [])
    trailer = next(
        (
            video
            for video in videos
            if video.get("site") == "YouTube"
            and video.get("type") == "Trailer"
            and video.get("key")
        ),
        None,
    )
    if trailer:
        movie.trailer_url = f"https://www.youtube.com/watch?v={trailer['key']}"
    return movie


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    await check_chat_limit(request)
    recommender = get_recommender()
    session_id, answer, recommendations = await chat.respond(
        payload.message,
        payload.session_id,
        tmdb,
        recommender,
    )
    return ChatResponse(
        session_id=session_id,
        message=answer,
        recommendations=recommendations,
    )
