import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from src.models.collaborative import Collaborative


class MLP(Collaborative):

    def __init__(
        self,
        user_dim: int,
        beer_dim: int,
        n_factors: int,
        n_layers: int,
        interactions: torch.FloatTensor,
        user_encoder: LabelEncoder,
        beer_encoder: LabelEncoder,
        max_rating: float,
        learning_rate: float = 1e-3,
        weight_decay: float = 0
    ):

        super(MLP, self).__init__(
            interactions,
            user_encoder,
            beer_encoder,
            max_rating,
        )

        self.save_hyperparameters(
            "user_dim",
            "beer_dim",
            "n_factors",
            "n_layers",
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

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.linears.append(
            nn.Linear(self.hparams.n_factors * 2, self.hparams.n_factors)
        )

        for _ in range(self.hparams.n_layers):
            in_features = self.linears[-1].out_features
            out_features = self.linears[-1].out_features // 2
            linear = nn.Linear(in_features, out_features)
            self.linears.append(linear)

        for _ in range(len(self.linears)):
            # dropout = nn.Dropout(p=0.0)
            dropout = nn.Dropout(p=0.2)
            self.dropouts.append(dropout)

        self.linear_n = nn.Linear(self.linears[-1].out_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        users: torch.LongTensor,
        beers: torch.LongTensor
    ) -> torch.FloatTensor:

        users = self.user_embedding(users)
        beers = self.beer_embedding(beers)

        x = torch.cat([users, beers], dim=1)

        for linear, dropout in zip(self.linears, self.dropouts):
            x = linear(x)
            x = dropout(x)
            x = F.relu(x)

        x = self.linear_n(x)
        x = self.sigmoid(x)

        return x.squeeze()
