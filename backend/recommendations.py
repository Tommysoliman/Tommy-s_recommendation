import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_ALLOWED_PICKLE_GLOBALS = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
}


class _MovieDataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"Unsupported movie dataset object: {module}.{name}")
        return super().find_class(module, name)


class MovieRecommender:
    def __init__(self, data_path: Path) -> None:
        with data_path.open("rb") as dataset_file:
            data = _MovieDataUnpickler(dataset_file).load()

        if isinstance(data, dict):
            data = data.get("movie", data.get("movies"))
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The movie dataset must contain a pandas DataFrame.")
        if not {"title", "tags"}.issubset(data.columns):
            raise ValueError("The movie dataset must include 'title' and 'tags' columns.")

        self.movies = data.copy()
        self.movies["title"] = self.movies["title"].fillna("").astype(str).str.strip()
        self.movies["tags"] = self.movies["tags"].fillna("").astype(str)
        self._title_indexes: dict[str, int] = {}
        for index, title in enumerate(self.movies["title"]):
            if title:
                self._title_indexes.setdefault(title.casefold(), index)

        self._vectors = self._vectorize(self.movies["tags"])

    @staticmethod
    def _vectorize(tags: pd.Series):
        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        try:
            return vectorizer.fit_transform(tags)
        except ValueError as error:
            if "empty vocabulary" not in str(error).lower():
                raise
            return csr_matrix((len(tags), 0), dtype="float64")

    @property
    def loaded(self) -> bool:
        return True

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        normalized_query = query.strip().casefold()
        if not normalized_query:
            return []
        matches = self.movies[
            self.movies["title"].str.casefold().str.contains(normalized_query, regex=False)
        ].head(max(0, min(limit, 20)))
        return [
            {
                "movie_id": int(row["movie_id"]) if "movie_id" in row and pd.notna(row["movie_id"]) else None,
                "title": str(row["title"]),
            }
            for _, row in matches.iterrows()
        ]

    def recommend(self, title: str, limit: int = 10) -> list[dict[str, Any]]:
        normalized_title = title.strip().casefold()
        if not normalized_title:
            raise ValueError("Enter a movie title.")
        if normalized_title not in self._title_indexes:
            raise KeyError(f"Movie '{title.strip()}' was not found in the recommendation dataset.")

        source_index = self._title_indexes[normalized_title]
        if self._vectors.shape[1] == 0:
            scores = [0.0] * len(self.movies)
        else:
            scores = cosine_similarity(
                self._vectors[source_index], self._vectors, dense_output=True
            ).ravel()

        ranked_indexes = sorted(
            (index for index in range(len(self.movies)) if index != source_index),
            key=lambda index: (-float(scores[index]), index),
        )[: max(0, min(limit, 10))]
        results: list[dict[str, Any]] = []
        for index in ranked_indexes:
            row = self.movies.iloc[index]
            movie_id = row.get("movie_id")
            overview = row.get("overview", "")
            results.append(
                {
                    "movie_id": int(movie_id) if pd.notna(movie_id) else None,
                    "title": str(row["title"]),
                    "overview": str(overview) if pd.notna(overview) else "",
                    "similarity_score": round(float(scores[index]), 4),
                }
            )
        return results
