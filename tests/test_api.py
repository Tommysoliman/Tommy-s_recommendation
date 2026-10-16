import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from backend import chat, main
from backend.tmdb import TMDBClient


client = TestClient(main.app)


def test_health_reports_dataset_and_optional_credentials():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["movie_dataset_loaded"] is True


def test_recommendations_are_case_insensitive_and_capped_at_ten():
    response = client.get("/api/recommendations/aVaTaR")

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "aVaTaR"
    assert len(body["recommendations"]) == 10
    assert all(item["title"] != "Avatar" for item in body["recommendations"])
    assert all(0 <= item["similarity_score"] <= 1 for item in body["recommendations"])


def test_unknown_local_movie_returns_not_found():
    response = client.get("/api/recommendations/not-a-real-movie")

    assert response.status_code == 404


def test_local_catalog_search_is_case_insensitive():
    response = client.get("/api/movies/catalog", params={"q": "avatar"})

    assert response.status_code == 200
    assert any(movie["title"] == "Avatar" for movie in response.json())


def test_invalid_discovery_year_range_is_rejected():
    response = client.get(
        "/api/movies/discover",
        params={"year_after": 2024, "year_before": 2018},
    )

    assert response.status_code == 422


def test_discovery_year_bounds_are_exclusive():
    tmdb = TMDBClient(api_key="test-key")

    async def search(query, limit):
        return [
            {"title": "Before", "release_date": "2020-12-31", "vote_average": 8.0},
            {"title": "Inside", "release_date": "2021-06-01", "vote_average": 8.0},
            {"title": "After", "release_date": "2022-01-01", "vote_average": 8.0},
        ]

    tmdb.search = search
    results = asyncio.run(
        tmdb.discover(query="thriller", year_after=2020, year_before=2022)
    )

    assert [movie["title"] for movie in results] == ["Inside"]


def test_tmdb_network_failures_are_reported(monkeypatch):
    class OfflineClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr("backend.tmdb.httpx.AsyncClient", OfflineClient)

    with pytest.raises(HTTPException) as error:
        asyncio.run(TMDBClient(api_key="test-key")._get("/trending/movie/week"))

    assert error.value.status_code == 502


def test_missing_tmdb_key_is_reported(monkeypatch):
    monkeypatch.setattr(main, "tmdb", TMDBClient(api_key=""))

    response = client.get("/api/movies/trending")

    assert response.status_code == 503
    assert "TMDB_API_KEY" in response.json()["detail"]


def test_chat_rejects_oversized_messages():
    response = client.post("/api/chat", json={"message": "x" * 2001})

    assert response.status_code == 422


def test_chat_rejects_blank_messages():
    response = client.post("/api/chat", json={"message": "  "})

    assert response.status_code == 422


def test_chat_reports_missing_openai_key(monkeypatch):
    monkeypatch.setattr(main.chat, "OPENAI_API_KEY", "")
    monkeypatch.setattr(main, "tmdb", TMDBClient(api_key="test-key"))

    response = client.post("/api/chat", json={"message": "Find a thriller"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_chat_uses_tools_and_returns_retrieved_movies(monkeypatch):
    chat._conversation_store.clear()
    completion_calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            completion_calls.append(kwargs["messages"])
            if len(completion_calls) == 1:
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="search_movies",
                                arguments=json.dumps({"query": "Arrival"}),
                            ),
                        )
                    ],
                )
            else:
                message = SimpleNamespace(
                    content="Arrival is a thoughtful science-fiction drama.",
                    tool_calls=None,
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            pass

    class FakeTMDB:
        configured = True

        async def search(self, query, limit):
            return [
                {
                    "id": 329865,
                    "title": "Arrival",
                    "overview": "A linguist works to communicate with visitors.",
                    "vote_average": 7.6,
                    "release_date": "2016-11-10",
                }
            ]

    monkeypatch.setattr(chat, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(chat, "AsyncOpenAI", FakeOpenAI)

    result = asyncio.run(chat.respond("Find thoughtful science fiction", None, FakeTMDB(), object()))

    assert result[1] == "Arrival is a thoughtful science-fiction drama."
    assert result[2][0].title == "Arrival"
    assert result[2][0].rating_source == "TMDB"
    assert completion_calls[1][-1]["role"] == "tool"


def test_chat_retains_prior_turns_for_same_session(monkeypatch):
    chat._conversation_store.clear()
    completion_calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            completion_calls.append(kwargs["messages"])
            answer = "Here are action movies." if len(completion_calls) == 1 else "Here are action movies released after 2018."
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=answer, tool_calls=None))
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            pass

    class FakeTMDB:
        configured = True

    monkeypatch.setattr(chat, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(chat, "AsyncOpenAI", FakeOpenAI)
    recommender = object()

    session_id, _, _ = asyncio.run(chat.respond("Recommend action movies", None, FakeTMDB(), recommender))
    _, answer, _ = asyncio.run(
        chat.respond("Only after 2018", session_id, FakeTMDB(), recommender)
    )

    prior_turns = completion_calls[1][1:-1]
    assert [turn["role"] for turn in prior_turns] == ["user", "assistant"]
    assert prior_turns[0]["content"] == "Recommend action movies"
    assert answer == "Here are action movies released after 2018."
