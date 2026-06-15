"""Content-based hobby recommendation engine.

This module uses TF-IDF vectorization and cosine similarity to recommend
hobbies with similar categories and descriptions. It is intentionally small and
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
    categories: str
    difficulty: int
    popularity: float
    similarity_score: float
    explanation: str


class HobbyRecommender:
    """A content-based hobby recommender using TF-IDF and cosine similarity."""

    def __init__(self, csv_path: str | Path = "hobbies.csv") -> None:
        self.csv_path = Path(csv_path)
        self.hobbies = self._load_hobbies()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.feature_matrix = self.vectorizer.fit_transform(self.hobbies["combined_features"])
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def _load_hobbies(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Could not find dataset at {self.csv_path}")

        hobbies = pd.read_csv(self.csv_path)
        required_columns = {"title", "categories", "description", "difficulty", "popularity"}
        missing = required_columns - set(hobbies.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        hobbies = hobbies.dropna(subset=["title", "categories", "description"]).copy()
        hobbies["combined_features"] = (
            hobbies["title"].astype(str) + " " +
            hobbies["categories"].astype(str) + " " +
            hobbies["description"].astype(str)
        )
        return hobbies.reset_index(drop=True)

    def get_titles(self) -> List[str]:
        return sorted(self.hobbies["title"].tolist())

    def recommend(
        self,
        title: str,
        top_n: int = 5,
        category_filter: Optional[str] = None,
        min_popularity: float = 0.0,
    ) -> List[RecommendationResult]:
        """Return similar hobbies for a selected title.

        Args:
            title: Hobby title selected by the user.
            top_n: Number of recommendations to return.
            category_filter: Optional category filter chosen by the user.
            min_popularity: Minimum popularity selected by the user.
        """
        if title not in self.hobbies["title"].values:
            raise ValueError(f"Hobby title '{title}' was not found in the dataset.")

        hobby_index = self.hobbies.index[self.hobbies["title"] == title][0]
        similarity_scores = list(enumerate(self.similarity_matrix[hobby_index]))
        similarity_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)

        results: List[RecommendationResult] = []
        for index, score in similarity_scores:
            candidate = self.hobbies.iloc[index]
            if candidate["title"] == title:
                continue
            if candidate["popularity"] < min_popularity:
                continue
            if category_filter and category_filter != "All":
                candidate_categories = str(candidate["categories"]).lower()
                if category_filter.lower() not in candidate_categories:
                    continue

            explanation = self._build_explanation(title, candidate["title"], candidate["categories"], score)
            results.append(
                RecommendationResult(
                    title=str(candidate["title"]),
                    categories=str(candidate["categories"]),
                    difficulty=int(candidate["difficulty"]),
                    popularity=float(candidate["popularity"]),
                    similarity_score=round(float(score), 3),
                    explanation=explanation,
                )
            )
            if len(results) >= top_n:
                break
        return results

    def available_categories(self) -> List[str]:
        category_set = set()
        for categories in self.hobbies["categories"]:
            for category in str(categories).split():
                category_set.add(category.strip())
        return ["All"] + sorted(category_set)

    @staticmethod
    def _build_explanation(source_title: str, candidate_title: str, categories: str, score: float) -> str:
        return (
            f"{candidate_title} is recommended because its category/description profile "
            f"is similar to {source_title}. Shared context includes: {categories}. "
            f"Cosine similarity score: {score:.3f}."
        )
