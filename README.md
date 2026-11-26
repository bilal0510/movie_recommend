# Movie Night for Two

An AI lab project that recommends movies for two viewers. The experiment asks whether putting extra weight on the lower predicted rating increases the chance that both viewers enjoy a recommendation.

## Run the first prototype

Use Python 3.10 or later. Download [MovieLens Latest Small](https://mooc.grouplens.org/datasets/movielens/latest/) and extract it under `data/` so that `data/ml-latest-small/movies.csv` is available. Commands run from the repository root.

```bash
python3 recommend.py --search "Toy Story"
python3 recommend.py --viewer-a 1=5 2=2 --viewer-b 1=3 2=5
python3 recommend.py --viewer-a 1=5 2=2 --viewer-b 1=3 2=5 --balance 0
```

Use `--movies PATH` for a catalog stored elsewhere. Each viewer can rate any number of movies on a 0.5–5 scale. Results exclude movies rated by either viewer and show which rated movie contributed most to each prediction.

## Interactive demo

With the MovieLens files in place, run these commands from the project directory.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Use **Try example ratings** for an instant comparison. To enter your own ratings,
search for a title under each viewer, choose a score, and save it. Select a
previously rated movie to change or remove its rating. Ratings stay in the
current browser session.

Compare the five picks in each column. Use **Prediction model** to switch
between Genre and Hybrid with the same viewer ratings. Expand a pick to see
the genre evidence and, in Hybrid mode, which rating relationship mattered most. Move the slider to 0 to make **Balanced** match **Average**, or to
1 to rank by the lower predicted rating alone.

## Model

Each movie is represented by its genres. The model compares the candidate's genres with those of each rated movie using cosine similarity. It averages the rating deviations from 3, weighted by similarity, with an additional neutral vote to account for sparse evidence. A movie without matching genres receives a neutral prediction.

The group score blends the average of the two predicted ratings with the lower predicted rating. The default balance of 0.5 gives each term equal weight. Set it to 0 for the average or 1 for the lower rating alone.

## Lab experiment

Run the evaluation with the same MovieLens download. The evaluation also uses
`ratings.csv`.

```bash
python3 evaluate.py
python3 evaluate.py --seed 17
```

The default run samples 300 eligible viewer pairs with a fixed seed. Each pair
must share at least 20 rated movies and each viewer must have at least one
training rating left after the split. The evaluator hides 20 shared movies from
both profiles and ranks those same candidates for all three methods. The
popularity baseline ranks by number of ratings from other users. The average
and balanced methods use the same genre model with different group scores.

**Both liked** is the share of top-five picks that both viewers rated at least
4 stars. **Mean rating** averages both viewers' actual ratings across those picks.

| Method | Both liked (seed 42) | Mean rating (seed 42) | Both liked (seed 17) | Mean rating (seed 17) |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 51.5% | 4.005 | 50.5% | 3.995 |
| Average | 44.3% | 3.894 | 45.9% | 3.908 |
| Balanced | 44.6% | 3.890 | 46.5% | 3.900 |

Balanced ranking had a slightly higher joint approval rate and a slightly
lower mean rating than average ranking on both seeds. Popularity led on both
metrics. These results describe reranking movies that both viewers chose to
rate, which favors titles with broad visibility and limits what we can infer
about recommendations across the full catalog. The two runs describe observed
differences without estimating statistical uncertainty.

### How many ratings should each viewer provide?

```bash
python3 evaluate.py --profile-sizes 3 5 10
python3 evaluate.py --profile-sizes 3 5 10 --seed 17
```

This comparison requires at least 10 available ratings per viewer after hiding
20 shared movies. Each viewer's remaining ratings are shuffled once per sampled
pair. The first 3 ratings are included in the first 5, and the first 5 are
included in the first 10. The test movies and viewer pairs stay fixed within
each run. Popularity uses the same candidate lists and is reported once.

| Ranking | Ratings per viewer | Both liked (seed 42) | Mean rating (seed 42) | Both liked (seed 17) | Mean rating (seed 17) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | — | 51.8% | 3.999 | 50.3% | 3.991 |
| Average | 3 | 39.9% | 3.782 | 40.9% | 3.811 |
| Average | 5 | 39.5% | 3.791 | 39.3% | 3.796 |
| Average | 10 | 43.7% | 3.853 | 41.9% | 3.843 |
| Balanced | 3 | 39.7% | 3.773 | 41.0% | 3.813 |
| Balanced | 5 | 39.9% | 3.780 | 39.5% | 3.801 |
| Balanced | 10 | 43.4% | 3.869 | 41.8% | 3.838 |

Ten ratings per viewer gave the genre model its highest joint approval and mean
rating in both runs. Five showed mixed results relative to three, and popularity
still led. Ten is a reasonable target for the demo when viewers have enough
movies to rate. This experiment varies the number of personal ratings. A later
experiment can vary the amount of population data available to a collaborative
model.

### Does more population data improve the hybrid model?

```bash
python3 evaluate_hybrid.py
python3 evaluate_hybrid.py --seed 17
```

The hybrid combines genre predictions with adjusted cosine similarity between
movies. Each training user's ratings are centered around their own mean.
Movie pairs need at least three shared raters, and similarities with low
support are shrunk toward zero. The hybrid uses the genre estimate when there
is no collaborative evidence.

For each seed, the evaluator reserves 152 MovieLens users for testing and
uses nested samples of 114, 229, and 458 of the remaining users. Each run
tests 300 pairs with 10 known ratings per viewer and the same 20 shared
held-out movies across every method. The popularity baseline counts ratings
from the full training split.

| Method | Both liked (seed 42) | Mean rating (seed 42) | Both liked (seed 17) | Mean rating (seed 17) |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 49.5% | 3.937 | 52.1% | 3.997 |
| Genre Average | 37.7% | 3.774 | 38.7% | 3.770 |
| Genre Balanced | 37.5% | 3.767 | 38.8% | 3.765 |
| Hybrid 25% Average | 42.0% | 3.856 | 44.2% | 3.902 |
| Hybrid 25% Balanced | 41.7% | 3.849 | 43.1% | 3.885 |
| Hybrid 50% Average | 43.1% | 3.868 | 45.3% | 3.933 |
| Hybrid 50% Balanced | 42.1% | 3.860 | 46.3% | 3.931 |
| Hybrid 100% Average | 45.1% | 3.905 | 48.3% | 3.966 |
| Hybrid 100% Balanced | 45.5% | 3.899 | 47.5% | 3.954 |

Both hybrid rankings improved on both metrics as training size increased in
these runs. Popularity still led, and the hybrid rule with the higher joint
approval rate at full training size varied by seed. This experiment uses a
separate test-user split, so compare methods within each run. The Streamlit
demo offers both models and uses the full local MovieLens catalog of ratings
for Hybrid mode.

## Movie metadata coverage audit

The audit uses [MovieLens 32M](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)
and [Kaggle's TMDB + IMDb snapshot, version 1074](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates/versions/1074)
from November 26, 2026. Save the two ZIP downloads under the ignored `data/`
directory and run:

```bash
python3 audit_metadata.py --movielens data/ml-32m.zip --metadata data/tmdb-movies-daily-updates.zip
```

The join uses the TMDB ID in MovieLens `links.csv` first, then falls back to a
unique IMDb ID when there is no TMDB match. Duplicate keys and disagreements
between the IDs are excluded. Of 87,585 MovieLens titles, 86,020 matched by
TMDB ID and 477 by IMDb fallback. The audit found 189 conflicts and 899
unmatched titles.

| MovieLens cohort | Films | Matched | Director | Cast | Plot overview | All three |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All films | 87,585 | 98.8% | 98.2% | 95.2% | 98.5% | 94.6% |
| Released 2020 or later | 7,986 | 99.0% | 98.3% | 96.1% | 98.8% | 95.5% |
| Five or fewer ratings | 47,180 | 98.9% | 97.9% | 93.7% | 98.4% | 92.7% |
| More than five ratings | 40,405 | 98.6% | 98.5% | 97.0% | 98.6% | 96.9% |

Coverage counts nonempty fields as a share of all films in each cohort, including
unmatched films in the denominator. Release years come from MovieLens titles.
The rating-count cohorts use the full dataset and describe coverage only.
For the recommendation experiment, define sparse titles using training ratings
and evaluate director, principal cast, and plot separately against genres on
the same held-out movies. This audit measures availability of the fields.

## Does movie metadata improve recommendations?

```bash
python3 evaluate_metadata.py
python3 evaluate_metadata.py --seed 17
```

Each run samples 1,200 MovieLens 32M users and evaluates 500 eligible pairs.
Both viewers supply 10 ratings. Each method ranks the same 20 held-out movies
that both viewers rated. The popularity baseline counts ratings from users
outside the evaluation sample. Personalized methods rank by the average of
the two predicted scores.

Each enriched model blends genre cosine similarity equally with one extra
signal when that signal is available for both films. Director and the first
five listed cast members use cosine similarity over sets of names. Plot uses
TF-IDF cosine similarity over overview words. Missing features fall back to
genre similarity for that film comparison.

| Method | Both liked (seed 42) | Mean rating (seed 42) | Both liked (seed 17) | Mean rating (seed 17) |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 50.2% | 4.000 | 53.2% | 4.057 |
| Genres | 41.7% | 3.828 | 44.1% | 3.882 |
| Genres + director | 41.5% | 3.824 | 44.6% | 3.888 |
| Genres + cast | 41.3% | 3.828 | 44.4% | 3.884 |
| Genres + plot | 41.5% | 3.823 | 44.6% | 3.888 |

All three enriched methods had lower joint approval on seed 42 and higher
joint approval on seed 17. The results show no consistent gain using this
blend and test protocol. They describe reranking shared, previously rated
movies. No evaluated pair had more than five held-out films with at most
100 training ratings, so a separate top-five ranking for sparse films
cannot distinguish these methods.
The sparse prediction-error check had only two pair-film observations in
seed 42 and zero in seed 17. A test designed to sample these films directly
is needed to draw conclusions about them.
