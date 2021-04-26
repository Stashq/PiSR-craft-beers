import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm.auto import tqdm

from src.models.recommender import Recommender


def mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame, model: Recommender
) -> Tuple[float, List[float]]:
    ranks = []

    test_discretized_ratings = test_discretized_ratings.groupby("user_id")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_beers = model.predict(user_id)

        test_beers = set(user_ratings.beer_id.values)
        test_liked_beers = set(user_ratings[user_ratings.liked].beer_id.values)

        pred_beers = [
            pred_beer in test_liked_beers
            for pred_beer in pred_beers
            if pred_beer in test_beers
        ]

        rank = _mean_reciprocal_rank(pred_beers)
        ranks.append(rank)

    return np.mean(ranks), ranks


def mean_average_precision(
    test_discretized_ratings: pd.DataFrame, model: Recommender, N: int = 10
) -> Tuple[float, List[float]]:
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
    test_discretized_ratings = test_discretized_ratings.groupby("user_id")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_beers = model.predict(user_id)

        test_beers = set(user_ratings.beer_id.values)
        test_liked_beers = set(user_ratings[user_ratings.liked].beer_id.values)

        pred_beers = [
            pred_beer in test_liked_beers
            for pred_beer in pred_beers
            if pred_beer in test_beers
        ]

        pred_beers = pred_beers[:N]
        pred_relevancy = np.nonzero(pred_beers)[0]

        average_precisions = []

        for index, rank in enumerate(pred_relevancy):
            average_precision = (index + 1) / (rank + 1)
            average_precisions.append(average_precision)

        average_precision = 0

        if average_precisions:
            average_precision = np.mean(average_precisions)

        ranks.append(average_precision)

    return np.mean(ranks), ranks


def mean_ndcg(
    test_discretized_ratings: pd.DataFrame, model: Recommender
) -> Tuple[float, List[float]]:
    ranks = []

    test_discretized_ratings = test_discretized_ratings.groupby("user_id")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_beers, pred_ratings = model.predict_ratings(user_id)

        user_liked = user_ratings["liked"].values
        beer_ids = user_ratings["beer_id"].values

        beer_ids = np.where(np.isin(pred_beers, beer_ids))
        pred_ratings = pred_ratings[beer_ids]

        if (
            pred_ratings.size > 1
        ):  # If there is no enough data -then ignore score counting for that user
            ndcg = ndcg_score(user_liked.reshape(1, -1), [pred_ratings])
            ranks.append(ndcg)

    return np.mean(ranks), ranks


def coverage(
    test_discretized_ratings: pd.DataFrame,
    model: Recommender,
) -> float:

    predicted = []
    all_beers = pd.unique(test_discretized_ratings["beer_id"])
    test_discretized_ratings = test_discretized_ratings.groupby("user_id")

    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_beers = model.predict(user_id)
        pred_beers = set(pred_beers)
        user_beers = set(user_ratings["beer_id"].values)
        pred_beers = pred_beers & user_beers

        predicted.append(pred_beers)

    unique_predictions = set(itertools.chain.from_iterable(predicted))
    prediction_coverage = len(unique_predictions) / (len(all_beers))
    return prediction_coverage


def rmse(
    test_ratings: pd.DataFrame,
    model: Recommender,
) -> float:

    # ! TODO: wywalić jako metrykę do logowania w NN

    predicted = []

    iterator = tqdm(
        test_ratings.iterrows(),
        total=len(test_ratings),
        desc="Testing predictions"
    )

    for index, row in iterator:
        pred_beers = model.predict_rating(row.user_id, row.beer_id)
        predicted.append(pred_beers)

    e = np.array(predicted) - test_ratings.rating.values
    se = e ** 2
    rmse = se.mean() ** 0.5
    return rmse


def _mean_reciprocal_rank(ranking: List[bool]) -> float:
    ranks = np.nonzero(ranking)[0]
    min_index = ranks.min() if ranks.size else float("inf")
    rank = 1 / (min_index + 1)
    return rank


def _mean_average_precision(ranking: List[bool]) -> float:
    ranks = np.nonzero(ranking)[0]

    scores = []

    for index, rank in enumerate(ranks):
        average_precision = (index + 1) / (rank + 1)
        scores.append(average_precision)

    score = 0

    if scores:
        score = np.mean(scores)

    return score
