"""Compare genre and hybrid rankings across nested MovieLens training sets."""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from random import Random

from evaluate import DEFAULT_DATA, load_ratings
from hybrid import HybridModel
from recommend import group_score, load_movies, predict

TRAINING_FRACTIONS = (25, 50, 100)
PROFILE_SIZE = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=300)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive.")

    try:
        movies = load_movies(args.data_dir / "movies.csv")
        ratings = load_ratings(args.data_dir / "ratings.csv", movies)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    rng = Random(args.seed)
    users = sorted(ratings)
    rng.shuffle(users)
    test_count = len(users) // 4
    if test_count < 2:
        parser.error("Need enough users to form separate training and test sets.")

    training_users = users[test_count:]
    training = {user: ratings[user] for user in training_users}
    sizes = {fraction: max(1, len(training) * fraction // 100) for fraction in TRAINING_FRACTIONS}
    models = {
        fraction: HybridModel({user: training[user] for user in training_users[:size]})
        for fraction, size in sizes.items()
    }
    popularity = Counter(movie for profile in training.values() for movie in profile)
    pairs = list(combinations(users[:test_count], 2))
    rng.shuffle(pairs)
    both_liked = Counter()
    rating_total = Counter()
    evaluated = 0

    for first_user, second_user in pairs:
        first, second = ratings[first_user], ratings[second_user]
        shared = first.keys() & second.keys()
        if len(shared) < 20 or min(len(first), len(second)) < 20 + PROFILE_SIZE:
            continue

        held_out = rng.sample(sorted(shared), 20)
        held_out_set = set(held_out)
        first_profile = {movie: first[movie] for movie in
                         rng.sample(sorted(first.keys() - held_out_set), PROFILE_SIZE)}
        second_profile = {movie: second[movie] for movie in
                          rng.sample(sorted(second.keys() - held_out_set), PROFILE_SIZE)}
        scores = {"Popularity": {movie: popularity[movie] for movie in held_out}}
        for movie in held_out:
            genre_first = predict(movies[movie], movies, first_profile).score
            genre_second = predict(movies[movie], movies, second_profile).score
            predictions = {"Genre": (genre_first, genre_second)}
            for fraction, model in models.items():
                predictions[f"Hybrid {fraction}%"] = (
                    model.predict(movie, first_profile, genre_first),
                    model.predict(movie, second_profile, genre_second),
                )

            for model_name, (first_score, second_score) in predictions.items():
                for rule, balance in (("Average", 0.0), ("Balanced", 0.5)):
                    scores.setdefault(f"{model_name} {rule}", {})[movie] = group_score(
                        first_score, second_score, balance
                    )

        for method, ranking in scores.items():
            ranked = sorted(
                held_out,
                key=lambda movie: (-ranking[movie], movies[movie].title.casefold(), movie),
            )
            for movie in ranked[:5]:
                both_liked[method] += min(first[movie], second[movie]) >= 4
                rating_total[method] += (first[movie] + second[movie]) / 2

        evaluated += 1
        if evaluated == args.pairs:
            break

    if not evaluated:
        parser.error("No eligible pairs in the test split.")

    print(f"{evaluated} pairs, {PROFILE_SIZE} input ratings per viewer, 20 held-out movies per pair")
    print(f"Seed {args.seed}, training users: {sizes}, test users: {test_count}")
    print(f"{'Method':<22} {'Both liked (≥4)':>17} {'Mean rating':>13}")
    total = evaluated * 5
    for method, hits in both_liked.items():
        print(f"{method:<22} {hits / total:>17.1%} {rating_total[method] / total:>13.3f}")


if __name__ == "__main__":
    main()
