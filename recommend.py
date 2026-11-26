"""Recommend an unseen movie for two viewers using MovieLens genres."""

import argparse
import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

NEUTRAL = 3.0


@dataclass(frozen=True)
class Movie:
    title: str
    genres: frozenset[str]


@dataclass(frozen=True)
class Prediction:
    score: float
    evidence: str


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    movie: Movie
    first: Prediction
    second: Prediction
    score: float


def load_movies(path: Path) -> dict[int, Movie]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"movieId", "title", "genres"}.issubset(reader.fieldnames or ()):
            raise ValueError("The movie catalog must have movieId, title, and genres columns.")
        movies = {}
        for row in reader:
            genres = frozenset(row["genres"].split("|")) - {"", "(no genres listed)"}
            movies[int(row["movieId"])] = Movie(row["title"], genres)
    return movies


def parse_ratings(tokens: list[str], movies: dict[int, Movie]) -> dict[int, float]:
    ratings = {}
    for token in tokens:
        movie_id_text, separator, rating_text = token.partition("=")
        try:
            if not separator:
                raise ValueError
            movie_id = int(movie_id_text)
            rating = float(rating_text)
        except ValueError:
            raise ValueError(f"Expected MOVIE_ID=RATING, got {token!r}.") from None
        if movie_id not in movies:
            raise ValueError(f"Unknown movie ID {movie_id}. Use --search to find it.")
        if not 0.5 <= rating <= 5:
            raise ValueError(f"Rating for movie {movie_id} must be between 0.5 and 5.")
        if movie_id in ratings:
            raise ValueError(f"Movie {movie_id} was rated twice by the same viewer.")
        ratings[movie_id] = rating
    return ratings


def predict(movie: Movie, movies: dict[int, Movie], ratings: dict[int, float]) -> Prediction:
    weighted_deviation = 0.0
    total_similarity = 0.0
    strongest_contribution = -1.0
    evidence = "No rated movie shares a genre"
    for movie_id, rating in ratings.items():
        rated = movies[movie_id]
        shared = movie.genres & rated.genres
        if not shared:
            continue
        similarity = len(shared) / sqrt(len(movie.genres) * len(rated.genres))
        contribution = similarity * (rating - NEUTRAL)
        weighted_deviation += contribution
        total_similarity += similarity
        if abs(contribution) > strongest_contribution:
            strongest_contribution = abs(contribution)
            genres = ", ".join(sorted(shared))
            evidence = f"{rated.title} ({rating:g}/5), shared genres: {genres}"

    # The extra weight on the neutral prior limits swings from sparse evidence.
    return Prediction(NEUTRAL + weighted_deviation / (1 + total_similarity), evidence)


def group_score(first: float, second: float, balance: float) -> float:
    average = (first + second) / 2
    return (1 - balance) * average + balance * min(first, second)


def recommend(
    movies: dict[int, Movie],
    first: dict[int, float],
    second: dict[int, float],
    balance: float,
) -> list[Recommendation]:
    candidates = []
    for movie_id in movies.keys() - first.keys() - second.keys():
        movie = movies[movie_id]
        first_prediction = predict(movie, movies, first)
        second_prediction = predict(movie, movies, second)
        score = group_score(first_prediction.score, second_prediction.score, balance)
        candidates.append(Recommendation(movie_id, movie, first_prediction, second_prediction, score))
    candidates.sort(key=lambda item: (-item.score, item.movie.title.casefold(), item.movie_id))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=Path("data/ml-latest-small/movies.csv"))
    parser.add_argument("--search", metavar="TITLE", help="Find movie IDs by title.")
    parser.add_argument("--viewer-a", nargs="+", metavar="ID=RATING")
    parser.add_argument("--viewer-b", nargs="+", metavar="ID=RATING")
    parser.add_argument("--balance", type=float, default=0.5, help="0 for average, 1 for lower score.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive.")
    if not 0 <= args.balance <= 1:
        parser.error("--balance must be between 0 and 1.")
    if args.search is None and (args.viewer_a is None or args.viewer_b is None):
        parser.error("Provide ratings for both viewers, or use --search to find movie IDs.")

    try:
        movies = load_movies(args.movies)
        if args.search is None:
            first = parse_ratings(args.viewer_a, movies)
            second = parse_ratings(args.viewer_b, movies)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.search is not None:
        matches = [(movie_id, movie.title) for movie_id, movie in movies.items()
                   if args.search.casefold() in movie.title.casefold()]
        for movie_id, title in matches[:args.limit]:
            print(f"{movie_id:>6}  {title}")
        if not matches:
            print("No matching titles.")
        return

    results = recommend(movies, first, second, args.balance)
    if not results:
        print("No movies remain that both viewers have yet to rate.")
    for number, result in enumerate(results[:args.limit], 1):
        print(f"{number}. {result.movie.title}  |  group score {result.score:.2f}")
        print(f"   Viewer A {result.first.score:.2f}/5 — {result.first.evidence}")
        print(f"   Viewer B {result.second.score:.2f}/5 — {result.second.evidence}")


if __name__ == "__main__":
    main()
