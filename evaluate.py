"""Compare movie-night rankings on held-out MovieLens ratings."""

import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path
from random import Random

from recommend import Movie, group_score, load_movies, predict

DEFAULT_DATA = Path("data/ml-latest-small")
METHODS = ("Popularity", "Average", "Balanced")


def load_ratings(path: Path, movies: dict[int, Movie]) -> dict[int, dict[int, float]]:
    ratings = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            user = int(row["userId"])
            movie = int(row["movieId"])
            if movie in movies:
                ratings.setdefault(user, {})[movie] = float(row["rating"])
    return ratings


def evaluate(movies, ratings, pair_limit, top_k, holdout_size, seed):
    rng = Random(seed)
    user_pairs = list(combinations(sorted(ratings), 2))
    rng.shuffle(user_pairs)
    popularity = Counter(movie for profile in ratings.values() for movie in profile)
    both_liked = Counter()
    rating_total = Counter()
    pairs_used = 0

    for first_id, second_id in user_pairs:
        first, second = ratings[first_id], ratings[second_id]
        shared = first.keys() & second.keys()
        if len(shared) < holdout_size or min(len(first), len(second)) <= holdout_size:
            continue

        held_out = set(rng.sample(sorted(shared), holdout_size))
        first_train = {movie: rating for movie, rating in first.items() if movie not in held_out}
        second_train = {movie: rating for movie, rating in second.items() if movie not in held_out}
        scores = {}
        for movie_id in held_out:
            movie = movies[movie_id]
            first_score = predict(movie, movies, first_train).score
            second_score = predict(movie, movies, second_train).score
            # Each held-out movie has a rating from both viewers. Exclude both.
            scores[movie_id] = (
                popularity[movie_id] - 2,
                group_score(first_score, second_score, 0),
                group_score(first_score, second_score, 0.5),
            )

        for index, method in enumerate(METHODS):
            ranked = sorted(
                held_out,
                key=lambda movie_id: (-scores[movie_id][index], movies[movie_id].title.casefold(), movie_id),
            )
            for movie_id in ranked[:top_k]:
                both_liked[method] += min(first[movie_id], second[movie_id]) >= 4
                rating_total[method] += (first[movie_id] + second[movie_id]) / 2

        pairs_used += 1
        if pairs_used == pair_limit:
            break

    return pairs_used, both_liked, rating_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--holdout", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.pairs < 1 or not 1 <= args.top_k <= args.holdout:
        parser.error("Require --pairs >= 1 and 1 <= --top-k <= --holdout.")

    try:
        movies = load_movies(args.data_dir / "movies.csv")
        ratings = load_ratings(args.data_dir / "ratings.csv", movies)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    pairs_used, both_liked, rating_total = evaluate(
        movies, ratings, args.pairs, args.top_k, args.holdout, args.seed
    )
    if not pairs_used:
        parser.error("No eligible user pairs. Try a smaller --holdout.")

    total = pairs_used * args.top_k
    print(f"{pairs_used} pairs, {args.holdout} held-out movies per pair, top {args.top_k}, seed {args.seed}")
    print(f"{'Method':<12} {'Both liked (≥4)':>17} {'Mean rating':>13}")
    for method in METHODS:
        print(f"{method:<12} {both_liked[method] / total:>17.1%} {rating_total[method] / total:>13.3f}")


if __name__ == "__main__":
    main()
