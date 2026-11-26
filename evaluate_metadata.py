"""Measure the contribution of director, cast, and plot to movie recommendations."""

import argparse
import csv
import re
from collections import Counter
from itertools import combinations, groupby
from math import log, sqrt
from pathlib import Path
from random import Random
from zipfile import BadZipFile

from audit_metadata import has_text, load_catalog, load_metadata, match, open_csv
from recommend import NEUTRAL, Movie, group_score, predict

FEATURES = ("Director", "Cast", "Plot")
WORDS = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
PROFILE_SIZE = 10
HOLDOUT_SIZE = 20
TOP_K = 5


def names(value: str, limit: int | None = None) -> frozenset[str]:
    if not has_text(value):
        return frozenset()
    entries = value.split(",")[:limit]
    return frozenset(name.strip().casefold() for name in entries if name.strip())


def extract_fields(row):
    return names(row["director"]), names(row["cast"], 5), row["overview"]


def load_features(movielens: Path, metadata: Path):
    years, links = load_catalog(movielens)
    by_tmdb, by_imdb = load_metadata(metadata, links, extract_fields)
    features = {}
    for movie in years:
        _, fields = match(links.get(movie, ("", "")), by_tmdb, by_imdb)
        if fields:
            features[movie] = fields
    return features


class PlotSimilarity:
    def __init__(self, features):
        self.documents = {
            movie: plot for movie, (_, _, plot) in features.items() if has_text(plot)
        }
        document_frequency = Counter()
        for plot in self.documents.values():
            document_frequency.update(set(WORDS.findall(plot.casefold())))
        count = len(self.documents)
        self.idf = {
            word: log((1 + count) / (1 + frequency)) + 1
            for word, frequency in document_frequency.items()
            if frequency >= 2 and frequency <= count / 2
        }
        self.vectors = {}

    def vector(self, movie):
        if movie not in self.vectors:
            tokens = Counter(WORDS.findall(self.documents.get(movie, "").casefold()))
            weights = {
                word: frequency * self.idf[word]
                for word, frequency in tokens.items()
                if word in self.idf
            }
            norm = sqrt(sum(weight * weight for weight in weights.values()))
            self.vectors[movie] = {word: weight / norm for word, weight in weights.items()}
        return self.vectors[movie]


def sample_profiles(path: Path, size: int, seed: int):
    """Read user-sorted ratings once and retain a uniform reservoir of profiles."""
    rng = Random(seed)
    profiles = []
    popularity = Counter()
    with open_csv(path, "ratings.csv") as stream:
        rows = csv.reader(stream)
        header = next(rows, [])
        user_col, movie_col, rating_col = (header.index(field) for field in ("userId", "movieId", "rating"))
        for index, (_, rows_for_user) in enumerate(groupby(rows, key=lambda row: row[user_col])):
            slot = index if index < size else rng.randrange(index + 1)
            if slot < size:
                profile = {int(row[movie_col]): float(row[rating_col]) for row in rows_for_user}
                popularity.update(profile.keys())
                if slot == len(profiles):
                    profiles.append(profile)
                else:
                    profiles[slot] = profile
            else:
                popularity.update(int(row[movie_col]) for row in rows_for_user)

    for profile in profiles:
        # Hold all sampled evaluation users out of the popularity counts.
        popularity.subtract(profile.keys())
    return profiles, popularity


def similarities(first: frozenset, second: frozenset) -> float | None:
    if not first or not second:
        return None
    return len(first & second) / sqrt(len(first) * len(second))


def score_features(movie_id, profile, movies, credits, plots):
    deviations = [0.0] * len(FEATURES)
    weights = [0.0] * len(FEATURES)
    movie = movies[movie_id]
    directors, cast, _ = credits.get(movie_id, (frozenset(), frozenset(), ""))
    plot = plots.vector(movie_id)

    for rated_id, rating in profile.items():
        rated = movies[rated_id]
        genre = similarities(movie.genres, rated.genres) or 0.0
        other_directors, other_cast, _ = credits.get(rated_id, (frozenset(), frozenset(), ""))
        other_plot = plots.vector(rated_id)
        plot_score = (
            sum(weight * other_plot.get(word, 0.0) for word, weight in plot.items())
            if plot and other_plot else None
        )
        extra_scores = (
            similarities(directors, other_directors),
            similarities(cast, other_cast),
            plot_score,
        )
        for index, extra in enumerate(extra_scores):
            affinity = genre if extra is None else (genre + extra) / 2
            deviations[index] += affinity * (rating - NEUTRAL)
            weights[index] += affinity

    return [NEUTRAL + deviation / (1 + weight) for deviation, weight in zip(deviations, weights)]


def update_metrics(scores, candidates, first, second, movies, counts):
    top_k = min(TOP_K, len(candidates))
    for method, ranking in scores.items():
        ranked = sorted(candidates, key=lambda movie: (-ranking[movie], movies[movie].title.casefold(), movie))
        for movie in ranked[:top_k]:
            counts[method, "both"] += min(first[movie], second[movie]) >= 4
            counts[method, "stars"] += (first[movie] + second[movie]) / 2
    return top_k


def evaluate(movies, credits, profiles, popularity, pair_limit, seed, sparse_limit):
    plots = PlotSimilarity(credits)
    rng = Random(seed)
    # Randomize the pairs from the bounded sample of evaluation users.
    pairs = list(combinations(range(len(profiles)), 2))
    rng.shuffle(pairs)
    overall, sparse = Counter(), Counter()
    sparse_error = Counter()
    evaluated = sparse_pairs = sparse_picks = sparse_count = 0
    for left, right in pairs:
        first, second = profiles[left], profiles[right]
        shared = first.keys() & second.keys() & movies.keys()
        if len(shared) < HOLDOUT_SIZE:
            continue
        held_out = rng.sample(sorted(shared), HOLDOUT_SIZE)
        held_out_ids = set(held_out)
        available = [sorted((profile.keys() & movies.keys()) - held_out_ids) for profile in (first, second)]
        if min(map(len, available)) < PROFILE_SIZE:
            continue
        first_input, second_input = (
            {movie: profile[movie] for movie in rng.sample(pool, PROFILE_SIZE)}
            for profile, pool in zip((first, second), available)
        )
        scores = {"Popularity": popularity, "Genres": {}}
        scores.update((feature, {}) for feature in FEATURES)
        for movie_id in held_out:
            movie = movies[movie_id]
            first_scores = [predict(movie, movies, first_input).score]
            second_scores = [predict(movie, movies, second_input).score]
            first_scores.extend(score_features(movie_id, first_input, movies, credits, plots))
            second_scores.extend(score_features(movie_id, second_input, movies, credits, plots))
            for method, first_score, second_score in zip(("Genres", *FEATURES), first_scores, second_scores):
                scores[method][movie_id] = group_score(first_score, second_score, 0)
                if popularity[movie_id] <= sparse_limit:
                    sparse_error[method] += (
                        abs(first_score - first[movie_id]) + abs(second_score - second[movie_id])
                    )

            sparse_count += popularity[movie_id] <= sparse_limit

        update_metrics(scores, held_out, first, second, movies, overall)
        sparse_movies = [movie for movie in held_out if popularity[movie] <= sparse_limit]
        if len(sparse_movies) > TOP_K:
            sparse_pairs += 1
            sparse_picks += update_metrics(scores, sparse_movies, first, second, movies, sparse)
        evaluated += 1
        if evaluated == pair_limit:
            break
    return (evaluated, evaluated * TOP_K, overall), (sparse_pairs, sparse_picks, sparse), (sparse_count, sparse_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movielens", type=Path, default=Path("data/ml-32m.zip"))
    parser.add_argument("--metadata", type=Path, default=Path("data/tmdb-movies-daily-updates.zip"))
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparse-limit", type=int, default=100)
    args = parser.parse_args()
    if args.pairs < 1 or args.sparse_limit < 0:
        parser.error("Require positive --pairs and nonnegative --sparse-limit.")

    try:
        with open_csv(args.movielens, "movies.csv") as stream:
            movies = {
                int(row["movieId"]): Movie(
                    row["title"], frozenset(row["genres"].split("|")) - {"", "(no genres listed)"}
                )
                for row in csv.DictReader(stream)
            }
        credits = load_features(args.movielens, args.metadata)
        profiles, popularity = sample_profiles(args.movielens, 1200, args.seed)
    except (OSError, ValueError, BadZipFile, KeyError) as error:
        parser.error(str(error))

    overall, sparse, (sparse_count, sparse_error) = evaluate(movies, credits, profiles, popularity, args.pairs, args.seed, args.sparse_limit)
    if not overall[0]:
        parser.error("No eligible pairs in the evaluation sample.")
    print(f"Seed {args.seed}; sampled {len(profiles):,} users; {PROFILE_SIZE} input ratings per viewer.")
    for label, (pairs, picks, counts) in (
        ("All held-out movies", overall),
        (f"At most {args.sparse_limit} training ratings", sparse),
    ):
        print(f"\n{label} | {pairs} pairs, {picks} top picks per method")
        if not picks:
            continue
        print(f"{'Method':<20} {'Both liked (≥4)':>17} {'Mean rating':>13}")
        for method in ("Popularity", "Genres", *FEATURES):
            label = f"Genres + {method.lower()}" if method in FEATURES else method
            print(f"{label:<20} {counts[method, 'both'] / picks:>17.1%} {counts[method, 'stars'] / picks:>13.3f}")

    print(f"\nSparse prediction error | {sparse_count} held-out pair-film observations")
    if sparse_count:
        for method in ("Genres", *FEATURES):
            label = f"Genres + {method.lower()}" if method in FEATURES else method
            print(f"{label:<20} MAE {sparse_error[method] / (2 * sparse_count):.3f}")


if __name__ == "__main__":
    main()
