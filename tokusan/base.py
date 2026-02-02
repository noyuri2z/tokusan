"""
Core LIME algorithm implementation.

This module contains the LimeBase class which implements the fundamental
LIME (Local Interpretable Model-Agnostic Explanations) algorithm. The
algorithm works by:

1. Creating perturbations of the input around the point of interest
2. Getting predictions from the black-box model for these perturbations
3. Weighting samples by their proximity to the original input
4. Fitting a simple interpretable model (linear regression) to the
   weighted samples
5. Using the coefficients of this simple model as feature importances

The key insight is that even complex models behave approximately linearly
in small local regions, so a linear model can faithfully explain individual
predictions.

Reference:
    Ribeiro, M.T., Singh, S. and Guestrin, C., 2016.
    "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
    In Proceedings of the 22nd ACM SIGKDD International Conference on
    Knowledge Discovery and Data Mining (pp. 1135-1144). ACM.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state


class LimeBase:
    """
    Core class for learning locally linear sparse models from perturbed data.

    This class implements the heart of the LIME algorithm: fitting a simple
    linear model to approximate the black-box model's behavior in a local
    neighborhood around a specific input.

    The key components are:
    - Kernel function: Weights samples by distance from the original input
    - Feature selection: Chooses which features to include in explanation
    - Linear model: Ridge regression to find feature importances

    Attributes:
        kernel_fn: Function that converts distances to similarity weights.
        verbose: Whether to print debugging information.
        random_state: Random state for reproducibility.

    Example:
        >>> def kernel_fn(distances):
        ...     return np.sqrt(np.exp(-(distances ** 2) / 25 ** 2))
        >>> base = LimeBase(kernel_fn)
        >>> intercept, exp, score, local_pred = base.explain_instance_with_data(
        ...     data, labels, distances, label=1, num_features=10
        ... )
    """

    def __init__(
        self,
        kernel_fn: Callable[[np.ndarray], np.ndarray],
        verbose: bool = False,
        random_state=None
    ):
        """
        Initialize the LimeBase explainer.

        Args:
            kernel_fn: Function that takes an array of distances and returns
                      an array of proximity weights in the range (0, 1).
                      Higher weights mean the sample is more important.
            verbose: If True, print intermediate values during explanation.
            random_state: Integer or numpy.RandomState for reproducibility.
                         If None, uses numpy's default random state.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(
        weighted_data: np.ndarray,
        weighted_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the LARS (Least Angle Regression) path for LASSO regularization.

        LARS efficiently computes the entire regularization path for LASSO,
        showing how coefficients change as regularization strength varies.
        This is used for automatic feature selection.

        Args:
            weighted_data: Feature matrix, already weighted by kernel.
            weighted_labels: Target values, already weighted by kernel.

        Returns:
            Tuple of (alphas, coefs):
                - alphas: Array of regularization parameter values
                - coefs: Array of coefficient values at each alpha
        """
        alphas, _, coefs = lars_path(
            weighted_data,
            weighted_labels,
            method='lasso',
            verbose=False
        )
        return alphas, coefs

    def forward_selection(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        num_features: int
    ) -> np.ndarray:
        """
        Select features using greedy forward selection.

        Iteratively adds features that most improve the model's R^2 score.
        This is more accurate than other methods but slower for large
        numbers of features.

        Args:
            data: Binary feature matrix (num_samples, num_features).
            labels: Target values for each sample.
            weights: Sample weights from kernel function.
            num_features: Maximum number of features to select.

        Returns:
            np.ndarray: Indices of selected features in order of selection.
        """
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features: List[int] = []

        # Iterate to select features one by one
        for _ in range(min(num_features, data.shape[1])):
            max_score = -float('inf')
            best_feature = 0

            # Try adding each unused feature
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue

                # Fit model with current features plus candidate
                candidate_features = used_features + [feature]
                clf.fit(
                    data[:, candidate_features],
                    labels,
                    sample_weight=weights
                )
                score = clf.score(
                    data[:, candidate_features],
                    labels,
                    sample_weight=weights
                )

                # Track the best feature to add
                if score > max_score:
                    best_feature = feature
                    max_score = score

            used_features.append(best_feature)

        return np.array(used_features)

    def feature_selection(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        num_features: int,
        method: str
    ) -> np.ndarray:
        """
        Select features using the specified method.

        Different methods offer different trade-offs between accuracy and
        computational cost:
        - 'forward_selection': Most accurate, slowest
        - 'highest_weights': Fast, uses weighted feature importance
        - 'lasso_path': Uses LASSO regularization path
        - 'auto': Chooses based on num_features
        - 'none': Uses all features

        Args:
            data: Binary feature matrix.
            labels: Target values.
            weights: Sample weights.
            num_features: Maximum features to select.
            method: Selection method name.

        Returns:
            np.ndarray: Indices of selected features.
        """
        if method == 'none':
            # Use all features
            return np.array(range(data.shape[1]))

        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)

        elif method == 'highest_weights':
            # Fit model and select features with highest weighted importance
            clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.issparse(data):
                # Handle sparse data efficiently
                coef = sp.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()

                # Handle case where data is sparser than requested features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((
                        indices,
                        np.zeros(num_to_pad, dtype=indices.dtype)
                    ))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                # Dense data: sort by weighted importance
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True
                )
                return np.array([x[0] for x in feature_weights[:num_features]])

        elif method == 'lasso_path':
            # Use LASSO regularization path for feature selection
            weighted_data = (
                (data - np.average(data, axis=0, weights=weights))
                * np.sqrt(weights[:, np.newaxis])
            )
            weighted_labels = (
                (labels - np.average(labels, weights=weights))
                * np.sqrt(weights)
            )
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)

            # Find the point in the path with desired number of features
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            return nonzero

        elif method == 'auto':
            # Choose method based on number of features requested
            # Forward selection is better for small feature counts
            if num_features <= 6:
                return self.feature_selection(
                    data, labels, weights, num_features, 'forward_selection'
                )
            else:
                return self.feature_selection(
                    data, labels, weights, num_features, 'highest_weights'
                )

        # Default fallback
        return np.array(range(data.shape[1]))

    def explain_instance_with_data(
        self,
        neighborhood_data: np.ndarray,
        neighborhood_labels: np.ndarray,
        distances: np.ndarray,
        label: int,
        num_features: int,
        feature_selection: str = 'auto',
        model_regressor=None
    ) -> Tuple[float, List[Tuple[int, float]], float, np.ndarray]:
        """
        Generate an explanation from perturbed data and predictions.

        This is the core explanation method. Given:
        - Perturbed versions of the input
        - Black-box model predictions for those perturbations
        - Distances from the original input

        It fits a weighted linear model and returns feature importances.

        Args:
            neighborhood_data: Binary matrix (num_samples, num_features).
                              First row is the original instance (all 1s).
                              Other rows have some features set to 0.
            neighborhood_labels: Predictions for each perturbed sample.
                                Shape: (num_samples, num_classes).
            distances: Distance from each sample to the original input.
            label: Which class label to explain.
            num_features: Maximum features in the explanation.
            feature_selection: Method for selecting features. Options:
                - 'forward_selection': Greedy feature addition
                - 'highest_weights': Use weighted coefficients
                - 'lasso_path': LASSO regularization path
                - 'none': Use all features
                - 'auto': Choose based on num_features
            model_regressor: sklearn regressor to use. Must support
                            sample_weight in fit() and have coef_ attribute.
                            Defaults to Ridge regression.

        Returns:
            Tuple of (intercept, exp, score, local_pred):
                - intercept: Linear model intercept
                - exp: List of (feature_id, weight) sorted by |weight|
                - score: R^2 score of the local model
                - local_pred: Prediction of local model on original input
        """
        # Convert distances to similarity weights using kernel function
        weights = self.kernel_fn(distances)

        # Extract the column for the label we're explaining
        labels_column = neighborhood_labels[:, label]

        # Select which features to include in the explanation
        used_features = self.feature_selection(
            neighborhood_data,
            labels_column,
            weights,
            num_features,
            feature_selection
        )

        # Set up the interpretable linear model
        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1,
                fit_intercept=True,
                random_state=self.random_state
            )
        easy_model = model_regressor

        # Fit the weighted linear model
        easy_model.fit(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights
        )

        # Evaluate how well the linear model approximates the black-box
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights
        )

        # Get the local model's prediction for the original instance
        local_pred = easy_model.predict(
            neighborhood_data[0, used_features].reshape(1, -1)
        )

        # Print debug info if verbose
        if self.verbose:
            print(f'Intercept: {easy_model.intercept_}')
            print(f'Prediction_local: {local_pred}')
            print(f'Right: {neighborhood_labels[0, label]}')

        # Create explanation as sorted (feature_id, weight) pairs
        # Sort by absolute weight value (most important first)
        explanation = sorted(
            zip(used_features, easy_model.coef_),
            key=lambda x: np.abs(x[1]),
            reverse=True
        )

        return (
            easy_model.intercept_,
            explanation,
            prediction_score,
            local_pred
        )
