from typing import List, Tuple

import numpy as np
import pandas as pd
from pytest import fixture
from sklearn.preprocessing import LabelEncoder

from src.models.recommender import Recommender


class DummyRecommender(Recommender):

    def __init__(self, user_encoder: LabelEncoder, beer_encoder: LabelEncoder):

        super(DummyRecommender, self).__init__(user_encoder, beer_encoder)

        self.RECOMMENDATIONS = {
            1: {2: 5.0, 32: 4.5, 4121: 4.2, 5422: 4.1, 12: 3.5, 42: 2.5},
            2: {42: 4.7, 5534: 4.2, 11: 3.0},
            3: {31: 2.75, 16: 2.2},
        }

    def predict(self, user_id: int) -> List[int]:
        ranking, scores = self.predict_ratings(user_id)
        return ranking

    def predict_rating(self, user_id: int, beer_id: int) -> float:
        return self.RECOMMENDATIONS[user_id][beer_id]

    def predict_ratings(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        beers = np.array(list(self.RECOMMENDATIONS[user_id].keys()))
        ratings = np.array(list(self.RECOMMENDATIONS[user_id].values()))
        return beers, ratings


@fixture(scope="session")
def model() -> Recommender:
    return DummyRecommender(user_encoder=None, beer_encoder=None)


@fixture(scope="session")
def test_ratings() -> pd.DataFrame:
    ratings = {
        "user_id": [1, 1, 1, 2, 2, 3, 3],
        "beer_id": [32, 12, 42, 42, 11, 31, 16],
        "rating": [5.0, 4.0, 2.5, 5.0, 3.5, 2.5, 4.5]
    }

    return pd.DataFrame(ratings)


@fixture(scope="session")
def test_discretized_ratings(test_ratings: pd.DataFrame) -> pd.DataFrame:
    test_discretized_ratings = test_ratings.copy()
    test_discretized_ratings["liked"] = [True, True, False, True, False, False, True]

    return test_discretized_ratings
