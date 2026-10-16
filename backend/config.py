import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def frontend_origins() -> list[str]:
    configured = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173")
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
_movie_data_path = Path(os.getenv("MOVIE_DATA_PATH", "movie.pkl"))
MOVIE_DATA_PATH = (
    _movie_data_path if _movie_data_path.is_absolute() else PROJECT_ROOT / _movie_data_path
)
FRONTEND_ORIGINS = frontend_origins()
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
