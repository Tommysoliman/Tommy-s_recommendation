from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Movie(BaseModel):
    movie_id: int | None = None
    title: str
    overview: str = ""
    poster_url: str | None = None
    release_date: str | None = None
    trailer_url: str | None = None
    genres: list[str] = Field(default_factory=list)
    rating: float | None = None
    rating_source: str | None = None
    similarity_score: float | None = None
    explanation: str | None = None


class RecommendationResponse(BaseModel):
    query: str
    recommendations: list[Movie]


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    recommendations: list[Movie] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    movie_dataset_loaded: bool
    tmdb_configured: bool
    openai_configured: bool


class ErrorResponse(BaseModel):
    detail: str


def movie_from_tmdb(result: dict[str, Any]) -> Movie:
    poster_path = result.get("poster_path")
    return Movie(
        movie_id=result.get("id"),
        title=result.get("title") or result.get("name") or "Untitled",
        overview=result.get("overview") or "",
        poster_url=(
            f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        ),
        release_date=result.get("release_date") or None,
        genres=[
            str(genre.get("name"))
            for genre in result.get("genres", [])
            if isinstance(genre, dict) and genre.get("name")
        ],
        rating=float(result["vote_average"]) if result.get("vote_average") is not None else None,
        rating_source="TMDB" if result.get("vote_average") is not None else None,
    )
