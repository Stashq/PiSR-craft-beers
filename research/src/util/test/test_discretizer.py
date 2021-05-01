import pandas as pd
from pytest import fixture

from src.util.discretizer import RatingDiscretizer


@fixture(scope="function")
def rating_discretizer() -> RatingDiscretizer:
    return RatingDiscretizer()


def test_rating_discretizer(
    test_ratings: pd.DataFrame,
    test_discretized_ratings: pd.DataFrame,
    rating_discretizer: RatingDiscretizer,
):
    discretized_ratings = rating_discretizer.fit_transform(test_ratings)

    assert isinstance(discretized_ratings, pd.DataFrame)
    assert len(test_ratings) == len(discretized_ratings)

    liked = discretized_ratings.liked.values
    test_liked = test_discretized_ratings.liked.values

    assert liked.dtype == bool
    assert (liked == test_liked).all()
