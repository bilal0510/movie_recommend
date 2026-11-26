# Movie Night for Two

An AI lab project that recommends movies for two viewers. The experiment asks whether putting extra weight on the lower predicted rating increases the chance that both viewers enjoy a recommendation.

## Run the first prototype

Use Python 3.10 or later. Download [MovieLens Latest Small](https://mooc.grouplens.org/datasets/movielens/latest/) and extract it under `data/` so that `data/ml-latest-small/movies.csv` is available. Commands run from the repository root.

```bash
python3 recommend.py --search "Toy Story"
python3 recommend.py --viewer-a 1=5 2=2 --viewer-b 1=3 2=5
python3 recommend.py --viewer-a 1=5 2=2 --viewer-b 1=3 2=5 --balance 0
```

Use `--movies PATH` for a catalog stored elsewhere. Each viewer can rate any number of movies on a 0.5–5 scale. Results exclude movies rated by either viewer and display the strongest genre match that influenced each prediction.

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
