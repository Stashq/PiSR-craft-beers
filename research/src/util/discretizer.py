import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RatingDiscretizer(TransformerMixin, BaseEstimator):
    """
    Constructs a sklearn based transformer working on pandas dataframes
    transforming user ratings to +1 or -1 if a rating is above or below
    user average. If a user has only a single rating then
    global average rating is used.

    Parameters
    ----------
    squeeze_movie_indexes : bool, default=True
        Indicate if movie indexes should be reindexed
        from 0 to len(ratings.movieId.unique()).

    Example
    --------
    >>> import pandas as pd
    >>> from src.util.discretizer import RatingDiscretizer
    >>> transformer = RatingDiscretizer()
    >>> X = pd.read_csv('ratings_small.csv')
    >>> X_train, X_test = sklearn.model_selection.train_test_split(X)
    >>> transformer.fit(X)
    >>> transformer.transform(X_test)
            userId  movieId  rating   timestamp
    38501      282      195    -1.0  1111494823
    99428      665     3110     1.0   995232733
    76284      529      959    -1.0   960052682
    ...        ...      ...     ...         ...
    """

    def fit(self, X, y=None):
        """
        Fit transformer by checking X.
        If ``validate`` is ``True``, ``X`` will be checked.
        Parameters
        ----------
        X : pandas dataframe containing columns ['userId', 'movieId', 'rating']
        Returns
        -------
        self
        """
        # X = self._check_input(X)  # in future version

        user_stats = X.groupby("user_id").agg({"rating": ["mean", "count"]})
        user_stats.columns = user_stats.columns.get_level_values(1)

        self.means = user_stats["mean"]

        # for users with a single rating just assume global mean
        global_mean = X.rating.mean()
        self.means.update(
            pd.Series(global_mean, self.means[user_stats["count"] == 1].index)
        )

        return self

    def transform(self, X):
        """
        Transform X using the forward function.
        Parameters
        ----------
        X : pandas dataframe containing columns ['userId', 'movieId', 'rating']
        Returns
        -------
        X_out : pandas dataframe containing columns ['userId','rating']
            where rating equals 1 or -1 if user likes the movie or not
            or NaN if user was not present while fitting the RatingDiscretizer
            if more columns were passed to the transform function,
            only 'rating' column is modified and the rest is returned unchanged
        """
        x_copy = X.copy()
        valid_user_ids = set(X["user_id"]) & set(self.means.index)
        ratings = X.set_index("user_id").rating
        # liked = np.sign(ratings - self.means[valid_user_ids]).values
        liked = ratings - self.means.loc[valid_user_ids]
        liked = liked > 0
        # x_copy.rating.update(pd.Series(ratings, X.rating.index))
        x_copy["liked"] = liked.values

        return x_copy
