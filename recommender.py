"""Content-based movie recommendation engine.

This module uses TF-IDF vectorization and cosine similarity to recommend
movies with similar genres and descriptions. It is intentionally small and
readable for a course project submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecommendationResult:
    title: str
    genres: str
    year: int
    rating: float
    similarity_score: float
    explanation: str


class MovieRecommender:
    """A content-based movie recommender using TF-IDF and cosine similarity."""

    def __init__(self, csv_path: str | Path = "movies.csv") -> None:
        self.csv_path = Path(csv_path)
        self.movies = self._load_movies()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.feature_matrix = self.vectorizer.fit_transform(self.movies["combined_features"])
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def _load_movies(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Could not find dataset at {self.csv_path}")

        movies = pd.read_csv(self.csv_path)
        required_columns = {"title", "genres", "description", "year", "rating"}
        missing = required_columns - set(movies.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        movies = movies.dropna(subset=["title", "genres", "description"]).copy()
        movies["combined_features"] = (
            movies["title"].astype(str) + " " +
            movies["genres"].astype(str) + " " +
            movies["description"].astype(str)
        )
        return movies.reset_index(drop=True)

    def get_titles(self) -> List[str]:
        return sorted(self.movies["title"].tolist())

    def recommend(
        self,
        title: str,
        top_n: int = 5,
        genre_filter: Optional[str] = None,
        min_rating: float = 0.0,
    ) -> List[RecommendationResult]:
        """Return similar movies for a selected title.

        Args:
            title: Movie title selected by the user.
            top_n: Number of recommendations to return.
            genre_filter: Optional genre filter chosen by the user.
            min_rating: Minimum rating selected by the user.
        """
        if title not in self.movies["title"].values:
            raise ValueError(f"Movie title '{title}' was not found in the dataset.")

        movie_index = self.movies.index[self.movies["title"] == title][0]
        similarity_scores = list(enumerate(self.similarity_matrix[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)

        results: List[RecommendationResult] = []
        for index, score in similarity_scores:
            candidate = self.movies.iloc[index]
            if candidate["title"] == title:
                continue
            if candidate["rating"] < min_rating:
                continue
            if genre_filter and genre_filter != "All":
                candidate_genres = str(candidate["genres"]).lower()
                if genre_filter.lower() not in candidate_genres:
                    continue

            explanation = self._build_explanation(title, candidate["title"], candidate["genres"], score)
            results.append(
                RecommendationResult(
                    title=str(candidate["title"]),
                    genres=str(candidate["genres"]),
                    year=int(candidate["year"]),
                    rating=float(candidate["rating"]),
                    similarity_score=round(float(score), 3),
                    explanation=explanation,
                )
            )
            if len(results) >= top_n:
                break
        return results

    def available_genres(self) -> List[str]:
        genre_set = set()
        for genres in self.movies["genres"]:
            for genre in str(genres).split():
                genre_set.add(genre.strip())
        return ["All"] + sorted(genre_set)

    @staticmethod
    def _build_explanation(source_title: str, candidate_title: str, genres: str, score: float) -> str:
        return (
            f"{candidate_title} is recommended because its genre/description profile "
            f"is similar to {source_title}. Shared context includes: {genres}. "
            f"Cosine similarity score: {score:.3f}."
        )
