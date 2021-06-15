from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder

from src.models.recommender import Recommender


class Collaborative(LightningModule, Recommender):

    def __init__(
        self,
        interactions: torch.FloatTensor,
        user_encoder: LabelEncoder,
        beer_encoder: LabelEncoder,
        max_rating: float
    ):
        super(Collaborative, self).__init__(user_encoder, beer_encoder, max_rating)
        self.interactions = interactions
        self.predict_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    # @abstractmethod
    # def init(self):
    #     # ? define architecture here
    #     pass

    @abstractmethod
    def forward(
        self,
        users: torch.LongTensor,
        beers: torch.LongTensor
    ) -> torch.FloatTensor:
        # ? define forward pass here
        pass

    def set_predict_device(self, device: torch.device = None):
        if not device:
            device = self.predict_device

        self.predict_device = device
        self.to(self.predict_device)

    def training_step(self, batch, batch_idx):
        users, beers, y = batch
        y_pred = self.forward(users, beers)

        mse_loss = F.mse_loss(y_pred, y)
        rmse_loss = torch.sqrt(mse_loss)

        self.log("train/mse", mse_loss, on_step=False, on_epoch=True)
        self.log("train/rmse_loss", rmse_loss, on_step=False, on_epoch=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        users, beers, y = batch
        y_pred = self.forward(users, beers)

        mse_loss = F.mse_loss(y_pred, y)
        rmse_loss = torch.sqrt(mse_loss)

        self.log("val/mse", mse_loss, on_step=False, on_epoch=True)
        self.log("val/rmse", rmse_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        users, beers, y = batch
        y_pred = self.forward(users, beers)

        mse_loss = F.mse_loss(y_pred, y)
        rmse_loss = torch.sqrt(mse_loss)

        self.log("test/mse", mse_loss, on_step=False, on_epoch=True)
        self.log("test/rmse", rmse_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/mse",
        }

    def predict(self, user_id: int) -> List[int]:
        beers, rating = self.predict_ratings(user_id)
        return list(beers)

    @torch.no_grad()
    def predict_rating(self, user_id: int, beer_id: int) -> float:
        user_id = self.user_encoder.transform([user_id])[0]
        beer_id = self.movie_encoder.transform([beer_id])[0]

        user_id = torch.LongTensor([user_id]).to(self.predict_device)
        beer_id = torch.LongTensor([beer_id]).to(self.predict_device)

        rating = self.forward(user_id, beer_id)
        rating = rating.cpu().item()

        # rating = self.ratings_encoder.inverse_transform([rating])[0]
        rating *= self.MAX_RATING

        return rating

    @torch.no_grad()
    def predict_ratings(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        user_id = self.user_encoder.transform([user_id])[0]

        beers = set(range(self.beer_encoder.classes_.size))
        beers_drinked = self.interactions[user_id].nonzero()

        if len(beers_drinked):
            beers_drinked = set(beers_drinked[0])
        else:
            beers_drinked = set()

        beers -= beers_drinked
        beers = list(beers)

        beers = torch.LongTensor(beers).to(self.predict_device)
        user = torch.LongTensor([user_id] * len(beers)).to(self.predict_device)

        ratings = self.forward(user, beers)
        ratings = ratings.cpu().numpy()
        beers = beers.cpu().numpy()

        ranking = pd.DataFrame(zip(beers, ratings), columns=["beer", "rating"])
        ranking = ranking.sort_values(by="rating", ascending=False)

        beers = ranking.beer.values
        ratings = ranking.rating.values

        beers = self.beer_encoder.inverse_transform(beers)
        ratings *= self.MAX_RATING
        # ratings = self.rating_encoder.inverse_transform(ratings)

        return beers, ratings
