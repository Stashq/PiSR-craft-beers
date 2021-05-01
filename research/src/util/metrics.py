import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm.auto import tqdm

from src.models.recommender import Recommender


def test_model(
    test_discretized_ratings: pd.DataFrame,
    model: Recommender,
    k: int = 20
) -> Tuple[pd.DataFrame, float]:

    recommendations: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    test_discretized_ratings = test_discretized_ratings.groupby("user_id")

    iterator = tqdm(test_discretized_ratings, desc="Calculating predictions")

    # ! TODO: multiprocessing? dask?
    for user_id, _ in iterator:
        pred_beers = model.predict_ratings(user_id)
        recommendations[user_id] = pred_beers

    mrr = get_mean_reciprocal_rank(test_discretized_ratings, recommendations)
    map_ = get_mean_average_precision(test_discretized_ratings, recommendations, k)
    ndcg = get_ndcg(test_discretized_ratings, recommendations, k)
    coverage = get_coverage(test_discretized_ratings, recommendations)
    rmse = get_rmse(test_discretized_ratings, recommendations)

    metrics = pd.concat([mrr, map_, ndcg, rmse])
    return metrics, coverage


def get_mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame,
    recommendations: Dict[int, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:

    ranks = []
    iterator = tqdm(test_discretized_ratings, desc="Calculating MRR")

    for user_id, user_ratings in iterator:
        pred_beers, _ = recommendations[user_id]

        test_beers = set(user_ratings.beer_id.values)
        test_liked_beers = set(user_ratings[user_ratings.liked].beer_id.values)

        pred_beers = [
            pred_beer in test_liked_beers
            for pred_beer in pred_beers
            if pred_beer in test_beers
        ]

        rank = _mean_reciprocal_rank(pred_beers)
        ranks.append(rank)

    mrr = pd.DataFrame(
        zip(
            [user_id for user_id, user_ratings in test_discretized_ratings],
            ["MRR"] * len(test_discretized_ratings),
            ranks
        ),
        columns=["user_id", "metric", "score"]
    )

    return mrr


def get_mean_average_precision(
    test_discretized_ratings: pd.DataFrame,
    recommendations: Dict[int, Tuple[np.ndarray, np.ndarray]],
    k: int = 20
) -> pd.DataFrame:
    """
    Calculate mean average precision.

    Parameters
    ----------
    test_discretized_ratings : pd.DataFrame
        Dataframe containing true users films ratings.
    model : Recommender
        Tested model.
    N : int
        Number of beers taken into account in recommendation for single user.

    Returns
    -------
    Tuple[float, List[float]]
        1. Mean average precision averaged for all users.
        2. List of average precision for all users separately.
    """
    ranks = []
    iterator = tqdm(test_discretized_ratings, desc="Calculating MAP")

    for user_id, user_ratings in iterator:
        pred_beers, _ = recommendations[user_id]

        test_beers = set(user_ratings.beer_id.values)
        test_liked_beers = set(user_ratings[user_ratings.liked].beer_id.values)

        pred_beers = [
            pred_beer in test_liked_beers
            for pred_beer in pred_beers
            if pred_beer in test_beers
        ]

        pred_relevancy = np.nonzero(pred_beers)[0]

        rank = _mean_average_precision(pred_relevancy, k)
        ranks.append(rank)

    map_ = pd.DataFrame(
        zip(
            [user_id for user_id, user_ratings in test_discretized_ratings],
            ["MAP"] * len(test_discretized_ratings),
            ranks
        ),
        columns=["user_id", "metric", "score"]
    )

    return map_


# ! TODO: znormalizować ocenę, bo wychodzi gówno
def get_ndcg(
    test_discretized_ratings: pd.DataFrame,
    recommendations: Dict[int, Tuple[np.ndarray, np.ndarray]],
    k: int = 20
) -> pd.DataFrame:

    ranks = []
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_beers, pred_ratings = recommendations[user_id]

        test_beers = set(user_ratings.beer_id.values)

        pred_ratings = [
            pred_rating
            for pred_beer, pred_rating in zip(pred_beers, pred_ratings)
            if pred_beer in test_beers
        ]

        user_ratings = user_ratings.sort_values(by="rating", ascending=False)
        test_ratings = user_ratings[user_ratings.beer_id.isin(test_beers)].rating.values

        ndcg = 1.0

        if test_ratings.size > 1:
            ndcg = ndcg_score([pred_ratings], [test_ratings])

        ranks.append(ndcg)

    ndcg = pd.DataFrame(
        zip(
            [user_id for user_id, user_ratings in test_discretized_ratings],
            ["NDCG"] * len(test_discretized_ratings),
            ranks
        ),
        columns=["user_id", "metric", "score"]
    )

    return ndcg


def get_coverage(
    test_discretized_ratings: pd.DataFrame,
    recommendations: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> float:

    all_beers = set(test_discretized_ratings.beer_id.values)

    pred_beers = [beers for beers, ratings in recommendations.values()]
    pred_beers = set(itertools.chain.from_iterable(pred_beers))

    coverage = len(pred_beers) / len(all_beers)
    return coverage


def get_rmse(
    test_discretized_ratings: pd.DataFrame,
    recommendations: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> float:

    scores: List[float] = []
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_beers, pred_ratings = recommendations[user_id]

        test_beers = set(user_ratings.beer_id.values)

        pred_ratings = [
            pred_rating
            for pred_beer, pred_rating in zip(pred_beers, pred_ratings)
            if pred_beer in test_beers
        ]

        user_ratings = user_ratings.sort_values(by="rating", ascending=False)
        test_ratings = user_ratings[user_ratings.beer_id.isin(test_beers)].rating.values

        e = pred_ratings - test_ratings
        se = e ** 2
        rmse = se.mean() ** 0.5
        scores.append(rmse)

    rmse = pd.DataFrame(
        zip(
            [user_id for user_id, user_ratings in test_discretized_ratings],
            ["RMSE"] * len(test_discretized_ratings),
            scores
        ),
        columns=["user_id", "metric", "score"]
    )

    return rmse


def _mean_reciprocal_rank(ranking: List[bool]) -> float:
    ranks = np.nonzero(ranking)[0]
    min_index = ranks.min() if ranks.size else float("inf")
    rank = 1 / (min_index + 1)
    return rank


def _mean_average_precision(ranking: List[bool], k: int) -> float:
    ranks = np.nonzero(ranking)[0][:k]

    scores = []

    for index, rank in enumerate(ranks):
        average_precision = (index + 1) / (rank + 1)
        scores.append(average_precision)

    score = 0

    if scores:
        score = np.mean(scores)

    return score
