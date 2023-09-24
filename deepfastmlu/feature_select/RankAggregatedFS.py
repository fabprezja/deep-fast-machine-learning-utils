import numpy as np
from typing import List, Optional, Any, Union

class RankAggregatedFS:
    """
    RankAggregatedFS is a feature selector that aggregates the rankings of features from multiple feature selection
    methods. It combines the scores or rankings of features from different methods to provide a unified ranking of
    features. This approach can be useful when there's uncertainty about which feature selection method to use, as
    it combines the strengths of multiple methods.

    Attributes:
        ranking_ (np.ndarray): After fitting, provides the indices of features ranked based on their aggregated scores.
        aggregated_scores_ (np.ndarray): After fitting, provides the aggregated scores of features.

    Args:
        methods (List[Any], optional): A list of feature selection methods to aggregate. Each method should have a
        `fit` method and either provide a `scores_` attribute after fitting or be compatible with the `transform`
        method. Defaults to an empty list.

        k (int): The number of top-ranked features to select after aggregation. Features are ranked based on their
        aggregated scores, and the top `k` features are selected.

        weights (Optional[List[float]], optional): If provided, assigns weights to the feature selection methods. This
        can be used to give more importance to certain methods over others. If None, all methods are equally weighted.

    ...
    """

    def __init__(self, methods: Optional[List[Any]] = None, k: int = 3, weights: Optional[List[float]] = None):
        """
        Initialize the RankAggregatedFS object.

        Args:
            methods (List[Any], optional): Feature selection methods to aggregate. Defaults to an empty list.
            k (int): Number of top-ranked features to select.
            weights (Optional[List[float]]): Weights for each method. If None, all methods are equally weighted.
        """
        self.methods = methods if methods is not None else []
        self.k = k
        self.weights = weights
        self.ranking_ = np.array([])
        self.aggregated_scores_ = np.array([])

        if self.weights is not None and len(self.weights) != len(self.methods):
            raise ValueError("The number of weights must match the number of methods.")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'RankAggregatedFS':
        """
        Fit the RankAggregatedFS to the data.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray, optional): Target values. Defaults to None.
        """
        scores = []

        for method in self.methods:
            method.fit(X, y)
            if hasattr(method, 'scores_'):
                scores.append(method.scores_)
            else:
                scores.append(np.ones(X.shape[1]))

        if self.weights is None:
            self.aggregated_scores_ = np.mean(scores, axis=0)
        else:
            self.aggregated_scores_ = np.average(scores, axis=0, weights=self.weights)

        self.ranking_ = np.argsort(self.aggregated_scores_)[::-1][:self.k]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to retain only the selected features.

        Args:
            X (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        return X[:, self.ranking_]

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray, optional): Target values. Defaults to None.

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_rankings(self) -> np.ndarray:
        """
        Return features ranked by their aggregated scores.

        Returns:
            np.ndarray: Features ranked by their aggregated scores.
        """
        return np.argsort(self.aggregated_scores_)[::-1]