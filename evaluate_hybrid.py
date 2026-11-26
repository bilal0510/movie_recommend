"""Compare genre and hybrid rankings across nested MovieLens training sets."""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from random import Random
from zipfile import BadZipFile

from audit_metadata import open_csv

from evaluate import DEFAULT_DATA, load_ratings
from evaluate_metadata import sample_profiles
from hybrid import HybridModel
from recommend import group_score, load_movies, predict, read_movies

TRAINING_FRACTIONS = (25, 50, 100)
PROFILE_SIZE = 10


def sample_split(path: Path, training_users: int, test_users: int, seed: int, rng: Random):
    profiles, popularity = sample_profiles(path, training_users + test_users, seed)
    if len(profiles) < training_users + test_users:
        raise ValueError("Dataset has fewer users than requested.")
    rng.shuffle(profiles)
    return profiles, popularity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--training-users", type=int, help="Cap the largest training sample.")
    parser.add_argument("--test-users", type=int, default=1200, help="Test sample in capped mode.")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive.")
    if args.training_users is not None and (args.training_users < 4 or args.test_users < 2):
        parser.error("Capped mode requires at least four training users and two test users.")

    rng = Random(args.seed)
    try:
        if args.training_users is None:
            movies = load_movies(args.data_dir / "movies.csv")
            ratings = load_ratings(args.data_dir / "ratings.csv", movies)
            users = sorted(ratings)
            rng.shuffle(users)
            test_count = len(users) // 4
            test_users, training_users = users[:test_count], users[test_count:]
            popularity = Counter(movie for user in training_users for movie in ratings[user])
        else:
            with open_csv(args.data_dir, "movies.csv") as stream:
                movies = read_movies(stream)
            profiles, popularity = sample_split(
                args.data_dir, args.training_users, args.test_users, args.seed, rng
            )
            ratings = {
                index: {movie: rating for movie, rating in profile.items() if movie in movies}
                for index, profile in enumerate(profiles)
            }
            del profiles
            test_users = list(range(args.test_users))
            training_users = list(range(args.test_users, len(ratings)))
            # The sampler removed every sampled user. Add back training users only.
            for user in training_users:
                popularity.update(ratings[user].keys())
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        parser.error(str(error))

    if len(test_users) < 2:
        parser.error("Need enough users to form separate training and test sets.")

    sizes = {fraction: max(1, len(training_users) * fraction // 100) for fraction in TRAINING_FRACTIONS}
    models = {
        fraction: HybridModel({user: ratings[user] for user in training_users[:size]})
        for fraction, size in sizes.items()
    }
    if args.training_users is None:
        sampled_popularity = None
    else:
        sampled_popularity = Counter(movie for user in training_users for movie in ratings[user])

    pairs = list(combinations(test_users, 2))
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
        scores = {"Popularity": popularity}
        if sampled_popularity is not None:
            scores["Popularity (sample)"] = sampled_popularity
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
    print(f"Seed {args.seed}, training users: {sizes}, test users: {len(test_users)}")
    print(f"{'Method':<22} {'Both liked (≥4)':>17} {'Mean rating':>13}")
    total = evaluated * 5
    for method, hits in both_liked.items():
        print(f"{method:<22} {hits / total:>17.1%} {rating_total[method] / total:>13.3f}")


if __name__ == "__main__":
    main()
