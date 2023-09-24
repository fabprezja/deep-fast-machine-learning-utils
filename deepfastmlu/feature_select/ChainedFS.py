import numpy as np
from typing import List, Optional, Any

class ChainedFS:
    """
    ChainedFS is a feature selector that sequentially applies a list of feature selection methods. This class allows
    for the chaining of multiple feature selection methods, where the output of one method becomes the input for the
    next. This can be particularly useful when one wants to combine the strengths of different feature selection
    techniques or when a sequence of operations is required to refine the feature set.

    For instance, one might first want to use a variance threshold to remove low-variance features and then apply
    a more computationally intensive method on the reduced set.

    Attributes:
        methods (List[Any]): A list of feature selection methods to be applied in sequence. Each method should have
        `fit` and `transform` methods. The order of the list determines the order of application of the methods.

        indices_ (np.ndarray): After fitting, provides the indices of features retained after all methods are applied.

    Args:
        methods (List[Any], optional): Feature selection methods to apply in sequence. Defaults to an empty list.

    ...
    """

    def __init__(self, methods: Optional[List[Any]] = None):
        """Initialize the ChainedFS object.

        Args:
            methods (List[Any], optional): Feature selection methods to apply in sequence. Defaults to an empty list.
        """
        self.methods = methods if methods is not None else []
        self.indices_ = np.array([])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the ChainedFS to the data.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray, optional): Target values. Defaults to None.
        """
        X_new = X
        self.indices_ = np.arange(X.shape[1])

        for method in self.methods:
            method.fit(X_new, y)
            if hasattr(method, 'get_support'):
                support = method.get_support()
                X_new = X_new[:, support]
                self.indices_ = self.indices_[support]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data based on the sequence of methods applied.

        Args:
            X (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        X_new = X
        for method in self.methods:
            X_new = method.transform(X_new)
        return X_new

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray, optional): Target values. Defaults to None.

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)