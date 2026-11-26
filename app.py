"""Interactive comparison of movie rankings for two viewers."""

from collections import Counter
from os import environ
from pathlib import Path
from zipfile import BadZipFile

import streamlit as st

from audit_metadata import open_csv
from evaluate import read_ratings
from hybrid import HybridModel
from recommend import Movie, Prediction, Recommendation, group_score, read_movies, recommend

DATA_DIR = Path(__file__).resolve().parent / "data" / "ml-latest-small"
EXAMPLE = ({1: 5.0, 2: 2.0}, {1: 3.0, 2: 5.0})
TOP_K = 5


@st.cache_data
def load_catalog(path: Path) -> tuple[dict[int, Movie], Counter[int]]:
    with open_csv(path, "movies.csv") as source:
        movies = read_movies(source)
    with open_csv(path, "ratings.csv") as source:
        profiles = read_ratings(source, movies)
    popularity = Counter(movie_id for profile in profiles.values() for movie_id in profile)
    return movies, popularity


@st.cache_resource
def load_hybrid(path: Path) -> HybridModel:
    movies, _ = load_catalog(path)
    with open_csv(path, "ratings.csv") as source:
        return HybridModel(read_ratings(source, movies))


@st.cache_data(max_entries=4)
def score_catalog(
    path: Path, first: dict[int, float], second: dict[int, float], model_name: str
) -> list[Recommendation]:
    movies, _ = load_catalog(path)
    if model_name == "Genre":
        return recommend(movies, first, second, balance=0)

    model = load_hybrid(path)

    def adjust(movie_id: int, profile: dict[int, float], genre: Prediction) -> Prediction:
        return model.explain(movie_id, profile, genre, movies)

    return recommend(movies, first, second, balance=0, adjust=adjust)


def edit_profile(
    label: str, movies: dict[int, Movie], movie_labels: dict[int, str]
) -> dict[int, float]:
    st.subheader(f"Viewer {label}")
    profile_key = f"profile_{label}"
    profile = st.session_state[profile_key]
    options = list(movie_labels)
    if len(options) > 20_000:
        query = st.text_input(f"Search for Viewer {label}", key=f"search_{label}").strip().casefold()
        matches = [option for option in options if query in movie_labels[option].casefold()] if len(query) >= 2 else []
        if len(query) >= 2 and not matches:
            st.caption("No matching titles.")
        elif len(matches) > 100:
            st.caption("Showing the first 100 matches. Add more of the title to narrow the search.")
        options = list(dict.fromkeys([*matches[:100], *profile]))
    movie_id = st.selectbox(
        f"Find a movie for Viewer {label}",
        options,
        index=None,
        format_func=movie_labels.__getitem__,
        placeholder="Choose a movie" if len(movie_labels) > 20_000 else "Type a movie title",
        key=f"movie_{label}",
    )
    if movie_id is not None:
        rating = st.number_input(
            f"Viewer {label} rating",
            min_value=0.5,
            max_value=5.0,
            value=profile.get(movie_id, 3.0),
            step=0.5,
            key=f"rating_{label}_{movie_id}",
        )
        save, remove = st.columns(2)
        if save.button("Save rating", key=f"save_{label}"):
            profile[movie_id] = rating
        if remove.button("Remove rating", key=f"remove_{label}", disabled=movie_id not in profile):
            profile.pop(movie_id, None)
            st.session_state.pop(f"rating_{label}_{movie_id}", None)
            st.rerun()
    if profile:
        st.caption(f"{len(profile)} rated")
        for rated_id, rating in profile.items():
            st.text(f"{movies[rated_id].title} · {rating:g}/5")
    else:
        st.caption("Add a rating to get recommendations.")
    return profile


def show_pick(number: int, item: Recommendation, ranking_detail: str) -> None:
    with st.container(border=True):
        st.text(f"{number}. {item.movie.title}")
        st.caption(ranking_detail)
        st.write(f"Viewer A **{item.first.score:.2f}/5** · Viewer B **{item.second.score:.2f}/5**")
        with st.expander("What influenced the predictions?"):
            st.text(f"A · {item.first.evidence}")
            st.text(f"B · {item.second.evidence}")


def main() -> None:
    st.set_page_config(page_title="Movie Night for Two", page_icon="🎬", layout="wide")
    st.title("Movie Night for Two")
    st.write("Rate movies for each viewer, then see how three ranking methods choose a shared pick.")

    data_path = Path(environ.get("MOVIE_RECOMMEND_DATA", DATA_DIR)).expanduser().resolve()
    try:
        movies, popularity = load_catalog(data_path)
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        st.error("Could not load the MovieLens dataset. See the README for setup.")
        st.caption(str(error))
        st.stop()

    dataset_label = {
        "ml-latest-small": "MovieLens Latest Small",
        "ml-32m-demo": "MovieLens 32M · 10,000 training users",
    }.get(data_path.stem, data_path.stem)
    st.caption(f"Dataset · {dataset_label} · {len(movies):,} films")
    st.session_state.setdefault("profile_A", {})
    st.session_state.setdefault("profile_B", {})
    if EXAMPLE[0].keys() <= movies.keys() and st.button("Try example ratings", help="Replaces both viewers’ current ratings."):
        st.session_state.profile_A = EXAMPLE[0].copy()
        st.session_state.profile_B = EXAMPLE[1].copy()
        for label in ("A", "B"):
            st.session_state.pop(f"movie_{label}", None)

    movie_ids = sorted(movies, key=lambda movie_id: (movies[movie_id].title.casefold(), movie_id))
    title_counts = Counter(movie.title for movie in movies.values())
    movie_labels = {
        movie_id: (
            f"{movies[movie_id].title} · ID {movie_id}"
            if title_counts[movies[movie_id].title] > 1 else movies[movie_id].title
        )
        for movie_id in movie_ids
    }
    first_col, second_col = st.columns(2)
    with first_col:
        first = edit_profile("A", movies, movie_labels)
    with second_col:
        second = edit_profile("B", movies, movie_labels)
    if not first or not second:
        return

    st.divider()
    st.header("Recommendations")
    model_name = st.radio(
        "Prediction model",
        ("Genre", "Hybrid"),
        horizontal=True,
        help="Hybrid combines genre overlap with movie relationships learned from MovieLens ratings.",
    )
    balance = st.slider(
        "Weight on the lower predicted rating",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Applies to Balanced. At 0 it matches Average. At 1 only the lower prediction counts.",
    )

    average = score_catalog(data_path, first, second, model_name)
    if not average:
        st.info("No unrated movies remain.")
        return

    popular = sorted(average, key=lambda item: (
        -popularity[item.movie_id], item.movie.title.casefold(), item.movie_id
    ))
    balanced = sorted(average, key=lambda item: (
        -group_score(item.first.score, item.second.score, balance),
        item.movie.title.casefold(),
        item.movie_id,
    ))

    columns = st.columns(3)
    with columns[0]:
        st.subheader("Popularity")
        st.caption("Ranked by the number of MovieLens ratings.")
        for number, item in enumerate(popular[:TOP_K], 1):
            show_pick(number, item, f"{popularity[item.movie_id]:,} MovieLens ratings")
    with columns[1]:
        st.subheader("Average")
        st.caption("Ranked by the mean of both predictions.")
        for number, item in enumerate(average[:TOP_K], 1):
            show_pick(number, item, f"Mean prediction {item.score:.2f}/5")
    with columns[2]:
        st.subheader("Balanced")
        st.caption("Ranked by a blend of the mean and the lower prediction.")
        for number, item in enumerate(balanced[:TOP_K], 1):
            score = group_score(item.first.score, item.second.score, balance)
            show_pick(number, item, f"Balanced score {score:.2f}/5")


if __name__ == "__main__":
    main()
