"""Build the local demo archive from the seed-42 benchmark's 10,000-user sample."""

import argparse
import csv
import io
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from audit_metadata import open_csv
from evaluate_hybrid import sample_split

SEED = 42
TEST_USERS = 1_200
TRAINING_POOL = 20_000
DEMO_USERS = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/ml-32m.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/ml-32m-demo.zip"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.suffix.lower() != ".zip":
        parser.error("Choose a separate output path ending in .zip.")

    try:
        profiles, _ = sample_split(args.source, TRAINING_POOL, TEST_USERS, SEED, Random(SEED))
        training = profiles[TEST_USERS:TEST_USERS + DEMO_USERS]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as directory:
            temporary = Path(directory) / "demo.zip"
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
                with open_csv(args.source, "movies.csv") as movies:
                    archive.writestr("movies.csv", movies.read())
                with archive.open("ratings.csv", "w") as raw, io.TextIOWrapper(
                    raw, encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("userId", "movieId", "rating"))
                    for user, profile in enumerate(training, 1):
                        writer.writerows((user, movie, rating) for movie, rating in profile.items())
            temporary.replace(args.output)
    except (OSError, ValueError, BadZipFile) as error:
        parser.error(str(error))
    print(f"Prepared {args.output} with {len(training):,} users and {sum(map(len, training)):,} ratings.")


if __name__ == "__main__":
    main()
