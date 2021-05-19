import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from src.models.collaborative import Collaborative


class MatrixFactorization2(Collaborative):

    def __init__(
        self,
        user_dim: int,
        beer_dim: int,
        n_factors: int,
        embedding_rescaler: float,
        interactions: torch.FloatTensor,
        user_encoder: LabelEncoder,
        beer_encoder: LabelEncoder,
        max_rating: float,
        use_mlp: bool = False,
        n_layers: int = 0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0
    ):

        super(MatrixFactorization2, self).__init__(
            interactions,
            user_encoder,
            beer_encoder,
            max_rating,
        )

        self.save_hyperparameters(
            "user_dim",
            "beer_dim",
            "n_factors",
            "embedding_rescaler",
            "use_mlp",
            "n_layers",
            "learning_rate",
            "weight_decay"
        )

        self.init()

    def init(self):

        mask = self.interactions > 0

        ratings_sum = self.interactions.sum()
        ratings_count = mask.sum()
        mean_rating = ratings_sum / ratings_count

        user_ratings_sum = self.interactions.sum(dim=1)
        user_ratings_count = mask.sum(dim=1)
        user_mean_rating = user_ratings_sum / user_ratings_count
        user_mean_rating[user_mean_rating.isnan()] = mean_rating

        beer_ratings_sum = self.interactions.sum(dim=0)
        beer_ratings_count = mask.sum(dim=0)
        beer_mean_rating = beer_ratings_sum / beer_ratings_count
        beer_mean_rating[beer_mean_rating.isnan()] = mean_rating

        # self.global_bias = torch.FloatTensor([mean_rating])
        self.global_bias = mean_rating
        self.user_bias = nn.Embedding(self.hparams.user_dim, 1)
        self.beer_bias = nn.Embedding(self.hparams.beer_dim, 1)

        self.user_bias.weight.data = torch.zeros(self.user_bias.weight.shape)
        self.beer_bias.weight.data = torch.zeros(self.beer_bias.weight.shape)

        self.user_bias.weight.data = user_mean_rating - mean_rating
        self.beer_bias.weight.data = beer_mean_rating - mean_rating

        self.user_bias.weight.requires_grad = False
        self.beer_bias.weight.requires_grad = False

        self.user_embedding = nn.Embedding(
            self.hparams.user_dim,
            self.hparams.n_factors
        )

        self.beer_embedding = nn.Embedding(
            self.hparams.beer_dim,
            self.hparams.n_factors
        )

        # ! dlaczego skalujemy embeddingi a nie bias
        self.user_embedding.weight.data *= self.hparams.embedding_rescaler
        self.beer_embedding.weight.data *= self.hparams.embedding_rescaler

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if self.hparams.use_mlp:

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

        users_embedded = self.user_embedding(users)
        beers_embedded = self.beer_embedding(beers)
        rating = users_embedded * beers_embedded

        rating = rating.sum(1, keepdim=True).squeeze()
        rating += self.global_bias
        rating += self.user_bias(users).squeeze()
        rating += self.beer_bias(beers).squeeze()

        if self.hparams.use_mlp:

            x = torch.cat([users_embedded, beers_embedded], dim=1)

            for linear, dropout in zip(self.linears, self.dropouts):
                x = linear(x)
                x = dropout(x)
                x = F.relu(x)

            x = self.linear_n(x)
            rating += x.squeeze()

        rating = self.sigmoid(rating)
        return rating
