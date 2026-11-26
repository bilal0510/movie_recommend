"""Interactive comparison of movie rankings for two viewers."""

from collections import Counter
from html import escape
from itertools import islice
from os import environ
from pathlib import Path
from zipfile import BadZipFile

import streamlit as st

from audit_metadata import open_csv
from evaluate import read_ratings
from hybrid import HybridModel, community_rating_prior
from recommend import Movie, Prediction, Recommendation, group_score, read_movies, recommend

DATA_DIR = Path(__file__).resolve().parent / "data" / "ml-latest-small"
EXAMPLE = ({1: 5.0, 2: 2.0}, {1: 3.0, 2: 5.0})
TOP_K = 5
STYLE = Path(__file__).resolve().parent / "assets" / "demo.css"


@st.cache_data
def load_catalog(path: Path) -> tuple[dict[int, Movie], Counter[int], dict[int, float]]:
    with open_csv(path, "movies.csv") as source:
        movies = read_movies(source)
    with open_csv(path, "ratings.csv") as source:
        profiles = read_ratings(source, movies)
    popularity, rating_prior = community_rating_prior(profiles.values())
    return movies, popularity, rating_prior


@st.cache_resource
def load_hybrid(path: Path) -> HybridModel:
    movies, _, _ = load_catalog(path)
    with open_csv(path, "ratings.csv") as source:
        return HybridModel(read_ratings(source, movies))


@st.cache_data(max_entries=4)
def score_catalog(
    path: Path, first: dict[int, float], second: dict[int, float], model_name: str
) -> list[Recommendation]:
    movies, _, _ = load_catalog(path)
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
        query = st.text_input(
            f"Search for Viewer {label}", key=f"search_{label}",
            placeholder="Enter at least two letters",
        ).strip().casefold()
        matches = []
        if len(query) >= 2:
            matches = list(islice(
                (option for option in options if query in movie_labels[option].casefold()), 101
            ))
            if not matches:
                st.caption("No matching titles.")
            elif len(matches) > 100:
                st.caption("Showing 100 matches. Enter more of the title to narrow the search.")
        options = list(dict.fromkeys([*matches[:100], *profile]))
    movie_id = st.selectbox(
        f"Choose a movie for Viewer {label}",
        options,
        index=None,
        format_func=movie_labels.__getitem__,
        placeholder="Choose a movie" if len(movie_labels) > 20_000 else "Type a movie title",
        key=f"movie_{label}",
        disabled=not options,
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
        if save.button("Save rating", key=f"save_{label}", type="primary"):
            profile[movie_id] = rating
        if remove.button("Remove rating", key=f"remove_{label}", disabled=movie_id not in profile):
            profile.pop(movie_id, None)
            st.session_state.pop(f"rating_{label}_{movie_id}", None)
            st.rerun()
    if profile:
        st.caption(f"Saved ratings · {len(profile)}")
        rows = "".join(
            f"<li><span>{escape(movies[rated_id].title)}</span><strong>{rating:g}/5</strong></li>"
            for rated_id, rating in profile.items()
        )
        st.html(f'<ul class="rating-list">{rows}</ul>')
    else:
        st.caption("Saved ratings will appear here.")
    return profile


def show_pick(method: str, number: int, item: Recommendation, ranking_detail: str) -> None:
    with st.container(border=True, key=f"pick_{method}_{number}"):
        st.html(f"""
            <div class="pick-title">
                <span class="pick-rank">{number}.</span>
                <span>{escape(item.movie.title)}</span>
            </div>
            <p class="pick-detail">{escape(ranking_detail)}</p>
            <div class="pick-scores">
                <div class="pick-score"><span>Viewer A</span><strong>{item.first.score:.2f}/5</strong></div>
                <div class="pick-score"><span>Viewer B</span><strong>{item.second.score:.2f}/5</strong></div>
            </div>
        """)
        with st.expander("Why this pick?"):
            st.html(f"""
                <p class="pick-evidence"><strong>Viewer A</strong><br>{escape(item.first.evidence)}</p>
                <p class="pick-evidence"><strong>Viewer B</strong><br>{escape(item.second.evidence)}</p>
            """)


def main() -> None:
    st.set_page_config(page_title="Movie Night for Two", page_icon="🎬", layout="wide")
    st.html(STYLE)

    data_path = Path(environ.get("MOVIE_RECOMMEND_DATA", DATA_DIR)).expanduser().resolve()
    try:
        movies, popularity, rating_prior = load_catalog(data_path)
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        st.error("Could not load the MovieLens dataset. See the README for setup.")
        st.caption(str(error))
        st.stop()

    dataset_label = {
        "ml-latest-small": "MovieLens Latest Small",
        "ml-32m-demo": "MovieLens 32M · 10,000 training users",
    }.get(data_path.stem, data_path.stem)
    st.html(f"""
        <header class="demo-hero">
            <div class="demo-dataset">{escape(dataset_label)} · {len(movies):,} films</div>
            <h1>Movie Night for Two</h1>
            <p>Rate a few films each and see which picks suit you both.</p>
        </header>
    """)
    st.session_state.setdefault("profile_A", {})
    st.session_state.setdefault("profile_B", {})
    ratings_heading, example_action = st.columns([3, 1], vertical_alignment="center")
    ratings_heading.header("Viewer ratings")
    with example_action:
        if EXAMPLE[0].keys() <= movies.keys() and st.button(
            "Try example ratings", type="primary", help="Replaces both viewers’ current ratings."
        ):
            st.session_state.profile_A = EXAMPLE[0].copy()
            st.session_state.profile_B = EXAMPLE[1].copy()
            for label in ("A", "B"):
                st.session_state.pop(f"movie_{label}", None)
                st.session_state.pop(f"search_{label}", None)

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
    with first_col, st.container(border=True, key="viewer_a"):
        first = edit_profile("A", movies, movie_labels)
    with second_col, st.container(border=True, key="viewer_b"):
        second = edit_profile("B", movies, movie_labels)
    if not first or not second:
        st.info("Save at least one rating per viewer, or try the example to see the recommendations.")
        return

    st.header("Recommendations")
    with st.container(border=True, key="ranking_controls"):
        model_column, balance_column = st.columns([1, 2], vertical_alignment="center")
        with model_column:
            model_name = st.radio(
                "Prediction model",
                ("Genre", "Hybrid", "Hybrid + rating prior"),
                index=2,
                horizontal=True,
                help="The rating prior uses a movie’s community rating and discounts it when few people rated the movie.",
            )
        with balance_column:
            balance = st.slider(
                "Balanced ranking · Weight on lower prediction",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                format="%d%%",
                help="At 0%, Balanced matches Average. At 100%, only the lower prediction counts.",
            ) / 100
    base_model = "Genre" if model_name == "Genre" else "Hybrid"
    average = score_catalog(data_path, first, second, base_model)
    if not average:
        st.info("No unrated movies remain.")
        return

    adjustments = rating_prior if model_name == "Hybrid + rating prior" else {}

    def ranking_score(item: Recommendation, balance_weight: float) -> float:
        return group_score(item.first.score, item.second.score, balance_weight) + adjustments.get(item.movie_id, 0)

    def ranking_key(item: Recommendation, balance_weight: float):
        return -ranking_score(item, balance_weight), item.movie.title.casefold(), item.movie_id

    popular = sorted(average, key=lambda item: (
        -popularity[item.movie_id], item.movie.title.casefold(), item.movie_id
    ))
    if adjustments:
        average.sort(key=lambda item: ranking_key(item, 0))
    balanced = sorted(average, key=lambda item: ranking_key(item, balance))

    columns = st.columns(3)
    with columns[0]:
        st.subheader("Popularity")
        st.caption("Ranked by the number of MovieLens ratings.")
        for number, item in enumerate(popular[:TOP_K], 1):
            show_pick("popularity", number, item, f"{popularity[item.movie_id]:,} MovieLens ratings")
    with columns[1]:
        st.subheader("Average")
        st.caption("Mean of both predictions" + (" plus the rating prior." if adjustments else "."))
        for number, item in enumerate(average[:TOP_K], 1):
            detail = f"Mean prediction {item.score:.2f}/5"
            if adjustments:
                detail += f" · Rating prior {adjustments.get(item.movie_id, 0):+.2f}"
            show_pick("average", number, item, detail)
    with columns[2]:
        st.subheader("Balanced")
        st.caption("Blend of the mean and lower prediction" + (" plus the rating prior." if adjustments else "."))
        for number, item in enumerate(balanced[:TOP_K], 1):
            score = group_score(item.first.score, item.second.score, balance)
            detail = f"Balanced prediction {score:.2f}/5"
            if adjustments:
                detail += f" · Rating prior {adjustments.get(item.movie_id, 0):+.2f}"
            show_pick("balanced", number, item, detail)


if __name__ == "__main__":
    main()
