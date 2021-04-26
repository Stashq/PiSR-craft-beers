from abc import ABC
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder


class Recommender(ABC):

    def __init__(self, user_encoder: LabelEncoder, beer_encoder: LabelEncoder):
        super(Recommender, self).__init__()
        self.user_encoder = user_encoder
        self.beer_encoder = beer_encoder

    def predict(self, user_id: int) -> List[int]:
        """
        Predicts ranking of beers unknown to the user.

        Parameters
        ----------
        user_id : int
            Real user's id from the data set.

        Returns
        -------
        List[int]
            List of beers real ids. Best recommendations first.
        """
        pass

    def predict_rating(self, user_id: int, beer_id: int) -> float:
        """
        Predicts rating for a given beer that a user would give.

        Parameters
        ----------
        user_id : int
            Real user's id from the data set.
        beer_id : int
            Real beer's id from the data set.

        Returns
        -------
        float
            Predicted beers's rating in range [0, 5].
        """
        pass

    def predict_ratings(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts ratings for all beers, that a user would give and haven't rated yet.

        Parameters
        ----------
        user_id : int
            User's id from the data set.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]:
            Ranked beers with their ratings. Highest ratings first.
        """
        pass
