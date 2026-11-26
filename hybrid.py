"""Blend genre predictions with adjusted-cosine movie similarity."""

from functools import lru_cache
from math import sqrt
from statistics import mean

from recommend import NEUTRAL


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
        deviation = 0.0
        weight = 0.0
        for rated_id, rating in profile.items():
            first, second = sorted((movie_id, rated_id))
            similarity = self.similarity(first, second)
            deviation += similarity * (rating - NEUTRAL)
            weight += abs(similarity)

        if not weight:
            return genre_score

        # Treat the genre estimate as one vote and add collaborative evidence.
        score = NEUTRAL + (genre_score - NEUTRAL + deviation) / (1 + weight)
        return min(5.0, max(0.5, score))
