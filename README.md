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

Next, add a popularity baseline and compare it with both group rules on held-out movies rated by both members of each test pair. Measure how often both viewers gave a top recommendation at least 4 stars and report the mean held-out rating alongside that result. A small interactive demo can use the same scoring functions.
