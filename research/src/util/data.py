from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from lazy import lazy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.util.discretizer import RatingDiscretizer


@dataclass(frozen=True)
class Data:
    ratings_path: Path

    @lazy
    def ratings(self) -> pd.DataFrame:
        ratings = pd.read_csv(self.ratings_path)
        ratings = ratings.drop_duplicates(subset=["user_id", "beer_id"])
        return ratings

    @lazy
    def user_encoder(self) -> LabelEncoder:
        encoder = LabelEncoder()
        encoder.fit(self.ratings.user_id.values)
        return encoder

    @lazy
    def beer_encoder(self) -> LabelEncoder:
        encoder = LabelEncoder()
        encoder.fit(self.ratings.beer_id.values)
        return encoder

    @lazy
    def user_count(self) -> int:
        return self.user_encoder.classes_.size

    @lazy
    def beer_count(self) -> int:
        return self.beer_encoder.classes_.size

    @lazy
    def max_rating(self) -> float:
        return self.train_ratings.rating.values.max() 

    @lazy
    def train_ratings(self) -> pd.DataFrame:
        train_ratings, _, _ = self._train_val_test_ratings
        return train_ratings

    @lazy
    def val_ratings(self) -> pd.DataFrame:
        _, val_ratings, _ = self._train_val_test_ratings
        return val_ratings

    @lazy
    def test_ratings(self) -> pd.DataFrame:
        _, _, test_ratings = self._train_val_test_ratings
        return test_ratings

    @lazy
    def _train_val_test_ratings(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        beers, counts = np.unique(self.ratings.beer_id.values, return_counts=True)
        single_beers = set(beers[counts == 1])

        # ? beers with a single rating will just go to training set
        single_ratings = self.ratings.loc[self.ratings.beer_id.isin(single_beers)]
        ratings = self.ratings.loc[~self.ratings.beer_id.isin(single_beers)]

        train_ratings, val_test_ratings = train_test_split(
            ratings, test_size=0.4, stratify=ratings.beer_id.values, random_state=42
        )

        beers, counts = np.unique(val_test_ratings.beer_id.values, return_counts=True)
        single_val_beers = set(beers[counts == 1])

        single_val_ratings = val_test_ratings.loc[
            val_test_ratings.beer_id.isin(single_val_beers)
        ]

        val_test_ratings = val_test_ratings.loc[
            ~val_test_ratings.beer_id.isin(single_val_beers)
        ]

        val_ratings, test_ratings = train_test_split(
            val_test_ratings,
            test_size=0.5,
            stratify=val_test_ratings.beer_id.values,
            random_state=42
        )

        train_ratings = pd.concat([train_ratings, single_ratings])
        train_ratings = shuffle(train_ratings, random_state=42)

        val_ratings = pd.concat([val_ratings, single_val_ratings])
        val_ratings = shuffle(val_ratings, random_state=42)

        return train_ratings, val_ratings, test_ratings

    @lazy
    def train_discretized_ratings(self) -> pd.DataFrame:
        return self.rating_discretizer.transform(self.train_ratings)

    @lazy
    def val_discretized_ratings(self) -> pd.DataFrame:
        return self.rating_discretizer.transform(self.val_ratings)

    @lazy
    def test_discretized_ratings(self) -> pd.DataFrame:
        return self.rating_discretizer.transform(self.test_ratings)

    @lazy
    def rating_discretizer(self) -> RatingDiscretizer:
        rating_discretizer = RatingDiscretizer()
        rating_discretizer.fit_transform(self.train_ratings)
        return rating_discretizer

    @lazy
    def train_interactions(self) -> FloatTensor:
        interactions = self._interactions(self.train_ratings)
        interactions /= self.max_rating
        return FloatTensor(interactions)

    @lazy
    def val_interactions(self) -> FloatTensor:
        interactions = self._interactions(self.val_ratings)
        interactions /= self.max_rating
        return FloatTensor(interactions)

    @lazy
    def test_interaction(self) -> FloatTensor:
        interactions = self._interactions(self.test_ratings)
        interactions /= self.max_rating
        return FloatTensor(interactions)

    def _interactions(self, ratings: pd.DataFrame) -> np.ndarray:
        """
        Creates interaction matrix from ratings DataFrame.

        Parameters
        ----------
        ratings : pd.DataFrame
            Ratings DataFrame `[user_id, beer_id, rating]`.

        Returns
        -------
        np.ndarray
            Interactions matrix. Rows are users, columns are beers.
            Specific cell denotes the rating, how a certain user scored the beer.
            Interactions are encoded to handle continuity of indices.
        """

        users_encoded = self.user_encoder.transform(ratings.user_id.values)
        beers_encoded = self.beer_encoder.transform(ratings.beer_id.values)
        scores = ratings.rating

        user_dim = self.user_count
        beer_dim = self.beer_count
        # interactions = sp.sparse.csr_matrix((user_dim, beer_dim), dtype=float)

        interactions = np.zeros((user_dim, beer_dim), dtype=float)

        iterator = tqdm(
            zip(users_encoded, beers_encoded, scores),
            desc="Building interaction matrix",
            total=len(users_encoded),
        )

        for user_id, beer_id, score in iterator:
            interactions[user_id, beer_id] = score

        return interactions

    @classmethod
    def get_sparsity_factor(cls, array: np.ndarray) -> float:
        rows, _ = array.nonzero()
        sparsity_factor = len(rows) / array.size

        return sparsity_factor

    @lazy
    def user_train(self) -> np.ndarray:
        user_train = self.user_encoder.transform(self.train_ratings.user_id.values)
        return user_train

    @lazy
    def user_val(self) -> np.ndarray:
        user_val = self.user_encoder.transform(self.val_ratings.user_id.values)
        return user_val

    @lazy
    def user_test(self) -> np.ndarray:
        user_test = self.user_encoder.transform(self.test_ratings.user_id.values)
        return user_test

    @lazy
    def beer_train(self) -> np.ndarray:
        beer_train = self.beer_encoder.transform(self.train_ratings.beer_id.values)
        return beer_train

    @lazy
    def beer_val(self) -> np.ndarray:
        beer_val = self.beer_encoder.transform(self.val_ratings.beer_id.values)
        return beer_val

    @lazy
    def beer_test(self) -> np.ndarray:
        beer_test = self.beer_encoder.transform(self.test_ratings.beer_id.values)
        return beer_test

    @lazy
    def y_train(self) -> np.ndarray:
        return self.train_ratings.rating.values / self.max_rating

    @lazy
    def y_val(self) -> np.ndarray:
        return self.val_ratings.rating.values / self.max_rating

    @lazy
    def y_test(self) -> np.ndarray:
        return self.test_ratings.rating.values / self.max_rating

    @lazy
    def train_set(self) -> TensorDataset:
        user_train = LongTensor(self.user_train)
        beer_train = LongTensor(self.beer_train)
        y_train = FloatTensor(self.y_train)

        return TensorDataset(user_train, beer_train, y_train)

    @lazy
    def val_set(self) -> TensorDataset:
        user_val = LongTensor(self.user_val)
        beer_val = LongTensor(self.beer_val)
        y_val = FloatTensor(self.y_val)

        return TensorDataset(user_val, beer_val, y_val)

    @lazy
    def test_set(self) -> TensorDataset:
        user_test = LongTensor(self.user_test)
        beer_test = LongTensor(self.beer_test)
        y_test = FloatTensor(self.y_test)

        return TensorDataset(user_test, beer_test, y_test)

    def get_train_loader(self, *, batch_size: int) -> DataLoader:
        return DataLoader(self.train_set, batch_size=batch_size)

    def get_val_loader(self, *, batch_size: int) -> DataLoader:
        return DataLoader(self.val_set, batch_size=batch_size)

    def get_test_loader(self, *, batch_size: int) -> DataLoader:
        return DataLoader(self.test_set, batch_size=batch_size)
