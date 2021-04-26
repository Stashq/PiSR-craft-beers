"""
Sources
MRR, MAP, NDCG:
https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832
"""

import numpy as np
import pandas as pd

from src.models.recommender import Recommender
from src.util import metrics


def test_mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame, model: Recommender
):
    mean, ranks = metrics.mean_reciprocal_rank(test_discretized_ratings, model)

    true_ranks = [1, 1, 0.5]
    assert isinstance(mean, float)
    assert mean == np.mean(true_ranks)

    assert ranks
    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == true_ranks


def test_mean_average_precision(
    test_discretized_ratings: pd.DataFrame, model: Recommender
):
    mean, ranks = metrics.mean_average_precision(test_discretized_ratings, model, N=10)

    true_ranks = [1, 1, 0.5]
    assert isinstance(mean, float)
    assert mean == np.mean(true_ranks)

    assert ranks
    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == true_ranks


def test_mean_ndcg(test_discretized_ratings: pd.DataFrame, model: Recommender):
    mean, ranks = metrics.mean_ndcg(test_discretized_ratings, model)

    true_ranks = [1, 1, 0.6309297535714573]
    assert isinstance(mean, float)
    assert mean == np.mean(true_ranks)

    assert ranks  # check if ranks is not empty list
    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == true_ranks


def test_coverage(test_discretized_ratings: pd.DataFrame, model: Recommender):
    coverage = metrics.coverage(test_discretized_ratings, model)

    assert isinstance(coverage, float)
    assert coverage == 1


def test_rmse(test_ratings: pd.DataFrame, model: Recommender):
    rmse = metrics.rmse(test_ratings, model)
    assert isinstance(rmse, float)
    assert rmse == 0.9405545476700737
