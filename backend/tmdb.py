import asyncio
import time
from typing import Any

import httpx
from fastapi import HTTPException

from backend.config import TMDB_API_KEY, TMDB_BASE_URL


class TMDBClient:
    def __init__(self, api_key: str = TMDB_API_KEY) -> None:
        self.api_key = api_key
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.configured:
            raise HTTPException(status_code=503, detail="Movie data is unavailable: TMDB_API_KEY is not configured.")
        request_params = {"api_key": self.api_key, "language": "en-US", **(params or {})}
        cache_key = f"{path}:{str(sorted(request_params.items()))}"
        async with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and cached[0] > time.monotonic():
                return cached[1]

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
                response = await client.get(f"{TMDB_BASE_URL}{path}", params=request_params)
                if response.status_code == 429:
                    raise HTTPException(status_code=503, detail="TMDB rate limit reached. Please try again shortly.")
                response.raise_for_status()
                data = response.json()
        except HTTPException:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise HTTPException(status_code=502, detail="Movie data service timed out or is unavailable.") from error
        except httpx.HTTPStatusError as error:
            raise HTTPException(status_code=502, detail="Movie data service returned an error.") from error
        except ValueError as error:
            raise HTTPException(status_code=502, detail="Movie data service returned invalid data.") from error

        async with self._cache_lock:
            self._cache[cache_key] = (time.monotonic() + 300, data)
            if len(self._cache) > 512:
                now = time.monotonic()
                self._cache = {key: value for key, value in self._cache.items() if value[0] > now}
        return data

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        data = await self._get("/search/movie", {"query": query, "include_adult": "false", "page": 1})
        return (data.get("results") or [])[:limit]

    async def discover(
        self,
        *,
        genre_id: int | None = None,
        year_after: int | None = None,
        year_before: int | None = None,
        min_rating: float | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if query:
            results = await self.search(query, limit=20)
            filtered = []
            for result in results:
                release_date = str(result.get("release_date") or "")
                year = int(release_date[:4]) if release_date[:4].isdigit() else None
                if year_after is not None and (year is None or year <= year_after):
                    continue
                if year_before is not None and (year is None or year >= year_before):
                    continue
                if min_rating is not None and float(result.get("vote_average") or 0) < min_rating:
                    continue
                if genre_id is not None and genre_id not in (result.get("genre_ids") or []):
                    continue
                filtered.append(result)
            return filtered[:limit]

        params: dict[str, Any] = {"sort_by": "popularity.desc", "include_adult": "false", "page": 1}
        if genre_id is not None:
            params["with_genres"] = genre_id
        if year_after is not None:
            params["primary_release_date.gte"] = f"{year_after + 1}-01-01"
        if year_before is not None:
            params["primary_release_date.lte"] = f"{year_before - 1}-12-31"
        if min_rating is not None:
            params["vote_average.gte"] = min_rating
            params["vote_count.gte"] = 50
        data = await self._get("/discover/movie", params)
        return (data.get("results") or [])[:limit]

    async def details(self, movie_id: int) -> dict[str, Any]:
        return await self._get(f"/movie/{movie_id}", {"append_to_response": "videos"})

    async def similar(self, movie_id: int, limit: int = 10) -> list[dict[str, Any]]:
        data = await self._get(f"/movie/{movie_id}/similar", {"page": 1})
        return (data.get("results") or [])[:limit]

    async def trending(self, limit: int = 12) -> list[dict[str, Any]]:
        data = await self._get("/trending/movie/week")
        return (data.get("results") or [])[:limit]

    async def genres(self) -> list[dict[str, Any]]:
        data = await self._get("/genre/movie/list")
        return data.get("genres") or []

    async def search_by_title(self, title: str) -> dict[str, Any] | None:
        matches = await self.search(title, limit=5)
        normalized = title.strip().casefold()
        return next(
            (movie for movie in matches if (movie.get("title") or "").strip().casefold() == normalized),
            matches[0] if matches else None,
        )
