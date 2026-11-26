"""Audit director, cast, and plot coverage for a MovieLens/TMDB join."""

import argparse
import csv
import io
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from zipfile import BadZipFile, ZipFile

FEATURES = ("director", "cast", "overview")
RELEASE_YEAR = re.compile(r"\((\d{4})\)\s*$")


@contextmanager
def open_csv(source: Path, filename: str):
    if source.is_dir():
        with (source / filename).open(encoding="utf-8-sig", newline="") as stream:
            yield stream
    elif source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            matches = [entry for entry in archive.namelist() if Path(entry).name == filename]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {filename} in {source}.")
            with archive.open(matches[0]) as raw, io.TextIOWrapper(
                raw, encoding="utf-8-sig", newline=""
            ) as stream:
                yield stream
    elif source.name == filename:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            yield stream
    else:
        raise ValueError(f"Expected a directory, ZIP archive, or a file named {filename}.")


def read_rows(stream, required):
    rows = csv.DictReader(stream)
    missing = set(required) - set(rows.fieldnames or ())
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
    return rows


def imdb_id(value: str) -> str:
    digits = value.strip().removeprefix("tt")
    return digits.lstrip("0") if digits.isdigit() else ""


def has_text(value: str) -> bool:
    return value.strip().casefold() not in {"", "nan", "null", "none", "n/a", "\\n"}


def load_catalog(path: Path):
    with open_csv(path, "movies.csv") as stream:
        movies = {
            int(row["movieId"]): RELEASE_YEAR.search(row["title"])
            for row in read_rows(stream, ("movieId", "title"))
        }
    with open_csv(path, "links.csv") as stream:
        links = {
            int(row["movieId"]): (row["tmdbId"].strip(), imdb_id(row["imdbId"]))
            for row in read_rows(stream, ("movieId", "tmdbId", "imdbId"))
        }
    return {movie: int(year[1]) if year else None for movie, year in movies.items()}, links


def load_metadata(path: Path, links):
    wanted_tmdb = {tmdb for tmdb, _ in links.values()} - {""}
    wanted_imdb = {imdb for _, imdb in links.values()} - {""}
    by_tmdb, by_imdb = {}, {}
    with open_csv(path, "TMDB_all_movies.csv") as stream:
        for row in read_rows(stream, ("id", "imdb_id", *FEATURES)):
            tmdb, imdb = row["id"].strip(), imdb_id(row["imdb_id"])
            if tmdb not in wanted_tmdb and imdb not in wanted_imdb:
                continue
            record = (tmdb, imdb, tuple(has_text(row[field]) for field in FEATURES))
            if tmdb in wanted_tmdb:
                by_tmdb[tmdb] = None if tmdb in by_tmdb else record
            if imdb in wanted_imdb:
                by_imdb[imdb] = None if imdb in by_imdb else record
    return by_tmdb, by_imdb


def match(links, by_tmdb, by_imdb):
    tmdb, imdb = links
    if tmdb in by_tmdb:
        record = by_tmdb[tmdb]
        if record is None:
            return "Ambiguous", ()
        if imdb and record[1] and record[1] != imdb:
            return "Conflict", ()
        return "TMDB", record[2]
    if imdb in by_imdb:
        record = by_imdb[imdb]
        return ("IMDb fallback", record[2]) if record is not None else ("Ambiguous", ())
    return "Unmatched", ()


def count_ratings(path: Path) -> Counter:
    with open_csv(path, "ratings.csv") as stream:
        rows = csv.reader(stream)
        header = next(rows, None) or []
        if "movieId" not in header:
            raise ValueError("ratings.csv is missing movieId.")
        index = header.index("movieId")
        return Counter(int(row[index]) for row in rows)


def audit(movielens: Path, metadata: Path, recent_year: int, few_ratings: int) -> None:
    years, links = load_catalog(movielens)
    by_tmdb, by_imdb = load_metadata(metadata, links)
    ratings = count_ratings(movielens)
    groups = {
        "All films": Counter(),
        f"Released {recent_year}+": Counter(),
        f"{few_ratings} or fewer ratings": Counter(),
        f"More than {few_ratings} ratings": Counter(),
    }
    join_counts = Counter()
    for movie, year in years.items():
        status, fields = match(links.get(movie, ("", "")), by_tmdb, by_imdb)
        join_counts[status] += 1
        cohorts = ["All films"]
        if year is not None and year >= recent_year:
            cohorts.append(f"Released {recent_year}+")
        cohorts.append(
            f"{few_ratings} or fewer ratings" if ratings[movie] <= few_ratings
            else f"More than {few_ratings} ratings"
        )
        for cohort in cohorts:
            counts = groups[cohort]
            counts["total"] += 1
            if fields:
                counts["matched"] += 1
                counts.update(feature for feature, present in zip(FEATURES, fields) if present)
                counts["all three"] += all(fields)

    print("Join outcomes")
    for status, count in sorted(join_counts.items()):
        print(f"  {status:<14} {count:>8,}")
    print("\nCoverage as a share of all MovieLens titles in each cohort")
    columns = ("matched", *FEATURES, "all three")
    print(f"{'Cohort':<24} {'Films':>8}" + "".join(f"{column:>12}" for column in columns))
    for cohort, counts in groups.items():
        total = counts["total"]
        coverage = [f"{counts[column] / total:.1%}" if total else "—" for column in columns]
        print(f"{cohort:<24} {total:>8,}" + "".join(f"{value:>12}" for value in coverage))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movielens", type=Path, required=True, help="MovieLens directory or ZIP")
    parser.add_argument("--metadata", type=Path, required=True, help="Kaggle CSV or ZIP")
    parser.add_argument("--recent-year", type=int, default=2020)
    parser.add_argument("--few-ratings", type=int, default=5)
    args = parser.parse_args()
    if args.few_ratings < 0:
        parser.error("--few-ratings must be nonnegative.")
    try:
        audit(args.movielens, args.metadata, args.recent_year, args.few_ratings)
    except (OSError, ValueError, BadZipFile) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
