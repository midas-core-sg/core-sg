from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover - exercised only on older sklearn
    validate_data = None

from .core_sg import CoreSG

ProgressCallback = Callable[[str, float, dict[str, Any]], None]


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


class CoreSGClusterer(ClusterMixin, BaseEstimator):
    """
    Scikit-learn-style estimator wrapper around the native :class:`CoreSG` API.

    `CoreSGClusterer` keeps `k_max` as a constructor parameter so it can be
    introspected, cloned, and tuned by scikit-learn utilities. The native
    :class:`CoreSG` object is built only once, on the first `fit(...)` call.
    Later `fit(...)` calls reuse that object and extract a new hierarchy for
    the requested `k`.
    """

    def __init__(
        self,
        k_max: int,
        metric: str = "euclidean",
        p: int = 2,
        algorithm: str = "core-sg",
        no_noise: bool = True,
        noise_label_strategy: str = "mst_label_propagation",
        random_state: int | np.random.RandomState | None = None,
        approx_knn_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        progress_callback: ProgressCallback | None = None,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
        match_reference_implementation: bool = False,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        max_cluster_size: int = 0,
        cluster_selection_epsilon_max: float = float("inf"),
    ) -> None:
        self.k_max = k_max
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.no_noise = no_noise
        self.noise_label_strategy = noise_label_strategy
        self.random_state = random_state
        self.approx_knn_kwargs = approx_knn_kwargs
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon_max = cluster_selection_epsilon_max

    def _validate_X(self, X: Any) -> np.ndarray:
        check_params = {
            "accept_sparse": False,
            "ensure_2d": True,
            "ensure_min_samples": 2,
            "dtype": [np.float64, np.float32],
        }
        if validate_data is not None:
            return validate_data(self, X, y=None, reset=True, **check_params)

        X_checked = check_array(X, **check_params)
        self.n_features_in_ = X_checked.shape[1]
        return X_checked

    def _validate_k_max(self, n_samples: int) -> int:
        if not _is_integer(self.k_max):
            raise ValueError("k_max must be an integer.")

        k_max = int(self.k_max)
        if k_max < 2:
            raise ValueError("k_max must be >= 2.")
        if k_max >= n_samples:
            raise ValueError("k_max must satisfy 2 <= k_max <= n_samples - 1.")
        return k_max

    @staticmethod
    def _validate_k(k: int | None, *, k_max: int) -> int:
        if k is None:
            return k_max
        if not _is_integer(k):
            raise ValueError("k must be an integer or None.")

        k_value = int(k)
        if k_value < 2:
            raise ValueError("k must be >= 2.")
        if k_value > k_max:
            raise ValueError("k must satisfy 2 <= k <= k_max.")
        return k_value

    def _core_sg_kwargs(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "p": self.p,
            "algorithm": self.algorithm,
            "no_noise": self.no_noise,
            "noise_label_strategy": self.noise_label_strategy,
            "random_state": self.random_state,
            "approx_knn_kwargs": (
                None if self.approx_knn_kwargs is None else dict(self.approx_knn_kwargs)
            ),
            "verbose": self.verbose,
            "progress_callback": self.progress_callback,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "match_reference_implementation": self.match_reference_implementation,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "cluster_selection_persistence": self.cluster_selection_persistence,
            "max_cluster_size": self.max_cluster_size,
            "cluster_selection_epsilon_max": self.cluster_selection_epsilon_max,
        }

    def _has_core_sg(self) -> bool:
        return isinstance(getattr(self, "core_sg_", None), CoreSG)

    def _sync_current_outputs(self) -> None:
        core_sg = self.core_sg_
        self.labels_ = core_sg.labels_
        self.probabilities_ = core_sg.probabilities_
        self.cluster_persistence_ = core_sg.cluster_persistence_
        self.condensed_tree_ = core_sg.condensed_tree_
        self.single_linkage_tree_ = core_sg.single_linkage_tree_
        self.minimum_spanning_tree_ = core_sg.minimum_spanning_tree_

    def fit(self, X: Any, y: Any = None, *, k: int | None = None) -> "CoreSGClusterer":
        """
        Build Core-SG once and expose clustering artifacts for `k`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense feature matrix used only when the internal native `CoreSG`
            object does not exist yet. After the first fit, subsequent calls
            reuse `core_sg_` and do not rebuild the support graph.
        y : ignored, default=None
            Accepted for scikit-learn compatibility.
        k : int or None, keyword-only, default=None
            Neighborhood size to extract. If None, `k_max` is used.

        Returns
        -------
        CoreSGClusterer
            The fitted estimator itself.
        """
        del y
        if not self._has_core_sg():
            if self.metric == "precomputed":
                raise ValueError(
                    "CoreSGClusterer does not support metric='precomputed'. "
                    "Use dense feature input with this wrapper."
                )
            X_checked = self._validate_X(X)
            k_max = self._validate_k_max(X_checked.shape[0])
            self.core_sg_ = CoreSG(**self._core_sg_kwargs())
            self.core_sg_.fit(X_checked, k_max=k_max)
            self.k_max_ = k_max

        k_value = self._validate_k(k, k_max=self.k_max_)
        self.core_sg_.extract_hierarchy_from_core_sg(k_value)
        self.k_ = k_value
        self._sync_current_outputs()
        return self

    def fit_predict(self, X: Any, y: Any = None, *, k: int | None = None) -> np.ndarray:
        """
        Fit the estimator and return the labels for the fitted `k`.
        """
        return self.fit(X, y=y, k=k).labels_

    def get_fitted_core_sg(self) -> CoreSG:
        """
        Return the fitted native CoreSG object.
        """
        check_is_fitted(self, attributes=["core_sg_"])
        return self.core_sg_
