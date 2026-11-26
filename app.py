"""Interactive comparison of movie rankings for two viewers."""

from collections import Counter
from pathlib import Path

import streamlit as st

from evaluate import load_ratings
from recommend import Movie, Recommendation, group_score, load_movies, recommend

DATA_DIR = Path(__file__).resolve().parent / "data" / "ml-latest-small"
EXAMPLE = ({1: 5.0, 2: 2.0}, {1: 3.0, 2: 5.0})
TOP_K = 5


@st.cache_data
def load_catalog(path: Path) -> tuple[dict[int, Movie], Counter[int]]:
    movies = load_movies(path / "movies.csv")
    profiles = load_ratings(path / "ratings.csv", movies)
    popularity = Counter(movie_id for profile in profiles.values() for movie_id in profile)
    return movies, popularity


@st.cache_data(max_entries=16)
def score_catalog(
    path: Path, first: dict[int, float], second: dict[int, float]
) -> list[Recommendation]:
    movies, _ = load_catalog(path)
    return recommend(movies, first, second, balance=0)


def edit_profile(
    label: str, movies: dict[int, Movie], movie_labels: dict[int, str]
) -> dict[int, float]:
    st.subheader(f"Viewer {label}")
    profile_key = f"profile_{label}"
    profile = st.session_state[profile_key]
    movie_id = st.selectbox(
        f"Find a movie for Viewer {label}",
        list(movie_labels),
        index=None,
        format_func=movie_labels.__getitem__,
        placeholder="Type a movie title",
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

    try:
        movies, popularity = load_catalog(DATA_DIR)
    except (OSError, ValueError, KeyError) as error:
        st.error("Download and extract MovieLens Latest Small under data/. See the README for setup.")
        st.caption(str(error))
        st.stop()

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
    balance = st.slider(
        "Weight on the lower predicted rating",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Applies to Balanced. At 0 it matches Average. At 1 only the lower prediction counts.",
    )

    average = score_catalog(DATA_DIR, first, second)
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
