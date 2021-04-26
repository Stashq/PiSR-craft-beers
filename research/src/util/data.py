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

from src.util.discretizer import RatingDiscretizer


@dataclass(frozen=True)
class Data:
    ratings_path: Path

    @lazy
    def ratings(self) -> pd.DataFrame:
        return pd.read_csv(self.ratings_path)

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
    def train_val_test_ratings(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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
    def train_ratings(self) -> pd.DataFrame:
        train_ratings, _, _ = self.train_val_test_ratings
        return train_ratings

    @lazy
    def val_ratings(self) -> pd.DataFrame:
        _, val_ratings, _ = self.train_val_test_ratings
        return val_ratings

    @lazy
    def test_ratings(self) -> pd.DataFrame:
        _, _, test_ratings = self.train_val_test_ratings
        return test_ratings

    @lazy
    def rating_discretizer(self) -> RatingDiscretizer:
        rating_discretizer = RatingDiscretizer()
        rating_discretizer.fit_transform(self.train_ratings)
        return rating_discretizer

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
        return self.train_ratings.rating.values

    @lazy
    def y_val(self) -> np.ndarray:
        return self.val_ratings.rating.values

    @lazy
    def y_test(self) -> np.ndarray:
        return self.test_ratings.rating.values

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
