import numpy as np
from sklearn.feature_selection import VarianceThreshold

class AdaptiveVarianceThreshold:
    """
    AdaptiveVarianceThreshold is a feature selector that dynamically determines a variance threshold based on the
    provided percentile of the feature variances. Features with a variance below this threshold are dropped.
    Traditional variance-based feature selection uses a fixed threshold, which is not optimal for all datasets.

    Attributes:
        percentile (float): The percentile of the feature variances used to determine the threshold. A higher
        percentile will result in a higher threshold and potentially more features being dropped.

        variances (np.ndarray, optional): Holds the variances of each feature in the dataset. Calculated during the
        `fit` method.

        threshold (float, optional): The calculated variance threshold. Features with a variance below this value
        will be dropped during the `transform` method.

        indices_to_drop (np.ndarray, optional): After fitting and transforming, provides the indices of features
        that were dropped due to their variance.

        selector (VarianceThreshold, optional): An instance of Scikit-learn's VarianceThreshold class, used to
        perform the feature selection based on the calculated threshold.

        verbose (bool): If set to True, prints additional information during the processing, such as the calculated
        variance threshold.

    Args:
        percentile (float, optional): The desired percentile of feature variances to use for determining the
        threshold. Defaults to 1.5.

        verbose (bool, optional): Whether to print additional information during processing. Defaults to False.

    ...
    """

    def __init__(self, percentile=1.5, verbose=False):
        """Initialize the AdaptiveVarianceThreshold object.

        Args:
            percentile (float, optional): The variances percentile for determining the threshold. Defaults to 1.5.
            verbose (bool, optional): Whether to print additional information during processing. Defaults to False.
        """
        self.percentile = percentile
        self.variances = None
        self.threshold = None
        self.indices_to_drop = None
        self.selector = None
        self.verbose = verbose

    def fit(self, features):
        """Calculate the variances and threshold, and fit the selector.

        Args:
            features (np.array): The feature set.
        """
        self.variances = np.var(features, axis=0)
        self.threshold = np.percentile(self.variances, self.percentile)
        if self.verbose:
            print(f"Variance threshold: {self.threshold}")
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(features)

    def transform(self, features):
        """Apply the selector to the features and store the indices of the dropped features.

        Args:
            features (np.array): The feature set.

        Returns:
            np.array: The transformed feature set.
        """
        features_transformed = self.selector.transform(features)
        self.indices_to_drop = np.where(self.selector.variances_ < self.threshold)
        return features_transformed

    def fit_transform(self, features, y=None):
        """
        Call the fit and transform methods successively.

        Args:
            features (np.array): The feature set.
            y (np.array, optional): Target values. This parameter is included for compatibility with scikit-learn's
            transformer API but is not used in this method.

        Returns:
            np.array: The transformed feature set.
        """
        self.fit(features)
        return self.transform(features)
