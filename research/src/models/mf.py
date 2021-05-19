import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from src.models.collaborative import Collaborative


class MatrixFactorization(Collaborative):

    def __init__(
        self,
        user_dim: int,
        beer_dim: int,
        n_factors: int,
        interactions: torch.FloatTensor,
        user_encoder: LabelEncoder,
        beer_encoder: LabelEncoder,
        max_rating: float,
        learning_rate: float = 1e-3,
        weight_decay: float = 0
    ):

        super(MatrixFactorization, self).__init__(
            interactions,
            user_encoder,
            beer_encoder,
            max_rating,
        )

        self.save_hyperparameters(
            "user_dim",
            "beer_dim",
            "n_factors",
            "learning_rate",
            "weight_decay"
        )

        self.init()

    def init(self):

        self.user_embedding = nn.Embedding(
            self.hparams.user_dim,
            self.hparams.n_factors
        )

        self.beer_embedding = nn.Embedding(
            self.hparams.beer_dim,
            self.hparams.n_factors
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        users: torch.LongTensor,
        beers: torch.LongTensor
    ) -> torch.FloatTensor:

        users = self.user_embedding(users)
        beers = self.beer_embedding(beers)
        rating = users * beers
        rating = rating.sum(1, keepdim=True)
        rating = rating.squeeze()
        rating = self.sigmoid(rating)

        return rating
