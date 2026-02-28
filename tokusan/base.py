"""Core LIME algorithm implementation for locally linear sparse models."""

from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state


class LimeBase:
    """Fits weighted local linear models to approximate black-box predictions."""

    def __init__(
        self,
        kernel_fn: Callable[[np.ndarray], np.ndarray],
        verbose: bool = False,
        random_state=None
    ):
        """Initialize with a kernel function that converts distances to weights."""
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(
        weighted_data: np.ndarray,
        weighted_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the LARS/LASSO regularization path for feature selection."""
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
        """Select features by greedily adding the one that most improves R^2."""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features: List[int] = []

        for _ in range(min(num_features, data.shape[1])):
            max_score = -float('inf')
            best_feature = 0

            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue

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
        """Select features using the specified method (auto, forward_selection, highest_weights, lasso_path, none)."""
        if method == 'none':
            return np.array(range(data.shape[1]))

        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)

        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.issparse(data):
                coef = sp.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()

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
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True
                )
                return np.array([x[0] for x in feature_weights[:num_features]])

        elif method == 'lasso_path':
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

            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            return nonzero

        elif method == 'auto':
            if num_features <= 6:
                return self.feature_selection(
                    data, labels, weights, num_features, 'forward_selection'
                )
            else:
                return self.feature_selection(
                    data, labels, weights, num_features, 'highest_weights'
                )

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
        """Fit a weighted linear model to perturbed data and return feature importances."""
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]

        used_features = self.feature_selection(
            neighborhood_data,
            labels_column,
            weights,
            num_features,
            feature_selection
        )

        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1,
                fit_intercept=True,
                random_state=self.random_state
            )
        easy_model = model_regressor

        easy_model.fit(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights
        )

        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights
        )

        local_pred = easy_model.predict(
            neighborhood_data[0, used_features].reshape(1, -1)
        )

        if self.verbose:
            print(f'Intercept: {easy_model.intercept_}')
            print(f'Prediction_local: {local_pred}')
            print(f'Right: {neighborhood_labels[0, label]}')

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
