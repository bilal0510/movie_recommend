"""Blend genre predictions with adjusted-cosine movie similarity."""

from functools import lru_cache
from math import sqrt
from statistics import mean

from recommend import NEUTRAL, Movie, Prediction


class HybridModel:
    def __init__(self, training: dict[int, dict[int, float]]):
        self.movie_vectors: dict[int, dict[int, float]] = {}
        for user, ratings in training.items():
            center = mean(ratings.values())
            for movie, rating in ratings.items():
                self.movie_vectors.setdefault(movie, {})[user] = rating - center
        self.similarity = lru_cache(maxsize=150_000)(self._similarity)

    def _similarity(self, first: int, second: int) -> float:
        left = self.movie_vectors.get(first, {})
        right = self.movie_vectors.get(second, {})
        shared = left.keys() & right.keys()
        if len(shared) < 3:
            return 0.0

        norm = sqrt(sum(left[user] ** 2 for user in shared) * sum(right[user] ** 2 for user in shared))
        if norm == 0:
            return 0.0
        cosine = sum(left[user] * right[user] for user in shared) / norm
        return cosine * len(shared) / (len(shared) + 10)

    def predict(self, movie_id: int, profile: dict[int, float], genre_score: float) -> float:
        score, _, _ = self._estimate(movie_id, profile, genre_score)
        return score

    def explain(
        self, movie_id: int, profile: dict[int, float], genre: Prediction, movies: dict[int, Movie]
    ) -> Prediction:
        score, rated_id, contribution = self._estimate(movie_id, profile, genre.score)
        if rated_id is None:
            return Prediction(score, f"{genre.evidence}\nNo MovieLens relationship contributed a rating adjustment.")
        direction = "raised" if contribution > 0 else "lowered"
        movie = movies[rated_id].title
        return Prediction(
            score,
            f"{genre.evidence}\nMovieLens relationship with {movie} "
            f"({profile[rated_id]:g}/5) {direction} the rating-based estimate most.",
        )

    def _estimate(
        self, movie_id: int, profile: dict[int, float], genre_score: float
    ) -> tuple[float, int | None, float]:
        deviation = 0.0
        weight = 0.0
        strongest_id = None
        strongest_contribution = 0.0
        for rated_id, rating in profile.items():
            first, second = sorted((movie_id, rated_id))
            similarity = self.similarity(first, second)
            contribution = similarity * (rating - NEUTRAL)
            deviation += contribution
            weight += abs(similarity)
            if abs(contribution) > abs(strongest_contribution):
                strongest_id = rated_id
                strongest_contribution = contribution

        if not weight:
            return genre_score, None, 0.0

        # Treat the genre estimate as one vote and add collaborative evidence.
        score = NEUTRAL + (genre_score - NEUTRAL + deviation) / (1 + weight)
        return min(5.0, max(0.5, score)), strongest_id, strongest_contribution
