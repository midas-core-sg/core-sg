from __future__ import annotations

from time import time
from typing import Any, Callable
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .edges import add_mst_edges_to_metric_edges, build_knng_vectors
from .hdbscan_adapter import (
    mst_to_single_linkage_tree,
    reference_mst_original_distance,
    wrap_condensed_tree,
    wrap_minimum_spanning_tree,
    wrap_single_linkage_tree,
)
from .hdbscan_adapter import (
    tree_to_labels as hdbscan_tree_to_labels,
)
from .knn import knn_from_precomputed
from .mst_kruskal import kruskal_mst
from .noise_handler import build_noise_handler
from .reweight import reweight_core_sg_mutual_reachability, sort_core_sg
from .score_sg import build_score_sg_from_data, is_graph_connected

ProgressCallback = Callable[[str, float, dict[str, Any]], None]


def _emit_progress(
    event: str,
    elapsed: float,
    *,
    verbose: int = 0,
    progress_callback: ProgressCallback | None = None,
    message: str | None = None,
    **info: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(event, elapsed, dict(info))
    if verbose:
        print(message or f"{event} done in {elapsed:.2f}s")


def build_core_sg_from_data(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str = "euclidean",
    p: int = 2,
    pairwise_dtype=np.float64,
    _round_distances: bool = False,
):
    """
    Gets X (n, d), calculates pairwise distances D (n, n) and build:
      - core distances
      - kNNG vetorized
      - metric_edges
      - Includes MST edges in metric_edges
      - Concatenate kNNG with MST MRD, building core-sg

    Returns:
      core_sg, metric_edges, core_k, mst_orig, D
    """
    X = np.asarray(X)
    n = X.shape[0]

    if n <= 1:
        raise ValueError("X needs to have shape > 1")
    if k_max <= 0 or k_max >= n:
        raise ValueError("k_max invalid (1 <= k_max <= n-1).")
    if k_max < 2:
        raise ValueError("k_max must be >= 2 to use HDBSCAN.")
    # ---- pairwise distances dentro da função ----
    if metric == "minkowski":
        D = pairwise_distances(X, metric=metric, p=p)
    elif metric == "arccos":
        D = pairwise_distances(X, metric="cosine")
    else:
        D = pairwise_distances(X, metric=metric)

    D = np.ascontiguousarray(D, dtype=pairwise_dtype)
    np.fill_diagonal(D, 0.0)

    if _round_distances:
        D = D.round(4)

    # ------------------------------------------------------------------
    # Separação correta dos papéis:
    # - min_samples_k: parâmetro do HDBSCAN
    # - graph_knn_k: número de vizinhos no kNN graph do Core-SG
    # - core_k: core-distance compatível com HDBSCAN
    # ------------------------------------------------------------------
    min_samples_k = k_max
    graph_knn_k = min_samples_k

    # kNN graph do Core-SG usa k original
    idxs_graph, dists_graph = knn_from_precomputed(
        D,
        k=graph_knn_k,
        include_self=False,
    )

    metric_edges, knng_to_insert = build_knng_vectors(
        idxs_graph,
        dists_graph,
        knng_size=n,
        k_max=graph_knn_k,
    )

    # min_samples conta o próprio ponto, então com diagonal 0
    # o índice correto é min_samples_k - 1
    # core_k = np.partition(D, kth=min_samples_k - 1, axis=1)[:, min_samples_k - 1]
    # core_k = np.ascontiguousarray(core_k, dtype=np.float64)

    core_k_list = np.partition(D, kth=min_samples_k - 1, axis=1)[:, :min_samples_k]
    core_k_list = np.sort(core_k_list, axis=1)

    # MST da mutual reachability com k_max
    hdb_obj, mst_orig = reference_mst_original_distance(D, k_max=k_max)

    u = mst_orig[:, 0].astype(np.int64, copy=False)
    v = mst_orig[:, 1].astype(np.int64, copy=False)

    # metric_edges deve sempre guardar a distância original D[u,v]
    mst_for_metric_edges = np.empty((mst_orig.shape[0], 3), dtype=np.float64)
    mst_for_metric_edges[:, 0] = u
    mst_for_metric_edges[:, 1] = v
    mst_for_metric_edges[:, 2] = D[u, v]

    metric_edges = add_mst_edges_to_metric_edges(
        metric_edges,
        mst_for_metric_edges,
        n_nodes=n,
    )

    u_min = np.minimum(u, v)
    v_max = np.maximum(u, v)

    mst_tmp = np.empty((mst_orig.shape[0], 3), dtype=np.float64)
    mst_tmp[:, 0] = u_min
    mst_tmp[:, 1] = v_max
    mst_tmp[:, 2] = -1.0  # placeholder

    core_sg = np.vstack([knng_to_insert, mst_tmp])
    core_sg = sort_core_sg(core_sg)

    return core_sg, metric_edges, core_k_list, D, hdb_obj


def mst_from_core_sg(
    core_sg: np.ndarray,
    metric_edges: np.ndarray,
    core_k_list: np.ndarray,
    n_nodes: int,
    k: int,
    *,
    verbose: int = 0,
    progress_callback: ProgressCallback | None = None,
):
    """
    Reweight -> Kruskal ->
    Retorna:
      mst_arr: (n-1,3) [u,v,w] ordenado por w
    """

    if k <= 0 or k >= n_nodes:
        raise ValueError("k_max inválido (precisa 1 <= k_max <= n-1).")
    if k < 2:
        raise ValueError("k_max deve ser >= 2 para reproduzir min_samples do HDBSCAN.")

    t0 = time()
    core_k = core_k_list[:, k - 1]
    core_k = np.ascontiguousarray(core_k, dtype=np.float64)
    weighted = reweight_core_sg_mutual_reachability(
        core_sg=core_sg,
        core_k=core_k,
        metric_edges=metric_edges,
        n_nodes=n_nodes,
    )

    t1 = time()
    _emit_progress(
        "reweight",
        t1 - t0,
        verbose=verbose,
        progress_callback=progress_callback,
        message=f"REWEIGHT Core_SG = {k} done in {t1 - t0:.2f}s",
        k=k,
    )

    t0 = time()
    mst_rec = kruskal_mst(weighted, n_nodes=n_nodes)
    mst_arr = np.column_stack([mst_rec.u, mst_rec.v, mst_rec.distance]).astype(
        np.float64, copy=False
    )
    mst_arr = mst_arr[np.argsort(mst_arr[:, 2], kind="mergesort")]
    t1 = time()
    _emit_progress(
        "kruskal",
        t1 - t0,
        verbose=verbose,
        progress_callback=progress_callback,
        message=f"KRUSKAL Core_SG = {k} done in {t1 - t0:.2f}s",
        k=k,
    )

    return mst_arr


def core_sg_mutual_reachability_distance(
    core_sg: np.ndarray,
    metric_edges: np.ndarray,
    core_k_list: np.ndarray,
    n_nodes: int,
    k_max: int,
    k: int,
):
    """
    Reweight CoreSG
    Returns:
      core_sg: (n-1,3) [u,v,w] order by w
    """

    if k <= 0 or k > k_max:
        raise ValueError("k invalid (1 <= k <= k_max).")
    if k < 2:
        raise ValueError("k must be >= 2.")

    core_k = core_k_list[:, k - 1]
    core_k = np.ascontiguousarray(core_k, dtype=np.float64)
    weighted = reweight_core_sg_mutual_reachability(
        core_sg=core_sg,
        core_k=core_k,
        metric_edges=metric_edges,
        n_nodes=n_nodes,
    )

    return weighted


def tree_to_labels(
    obj: "CoreSG",
    single_linkage_tree: np.ndarray,
    min_spanning_tree: np.ndarray,
):
    """
    Convert a single linkage tree into the standard HDBSCAN outputs.

    Only the parameters explicitly provided in `obj.hdbscan_kwargs` and
    supported by `_tree_to_labels` are forwarded. Any missing parameter
    falls back to the default defined by the internal HDBSCAN function.

    Parameters
    ----------
    obj : CoreSG
        Fitted CoreSG instance containing the data required by
        `_tree_to_labels(...)` and the keyword arguments originally provided at
        initialization.
    single_linkage_tree : np.ndarray
        Single linkage hierarchy built from the minimum spanning tree.
    min_spanning_tree : np.ndarray
        Minimum spanning tree associated with the hierarchy.

    Returns
    -------
    tuple
        A tuple with:
        (
            labels,
            probabilities,
            cluster_persistence,
            condensed_tree,
            single_linkage_tree,
            min_spanning_tree,
        )
    """

    tree_kwargs = obj._get_tree_to_labels_kwargs()

    return hdbscan_tree_to_labels(
        obj._tree_to_labels_data_,
        single_linkage_tree,
        tree_kwargs=tree_kwargs,
        min_spanning_tree=min_spanning_tree,
    )


class CoreSG:
    """
    Core-SG wrapper for reusing HDBSCAN computations across different values of k.

    The class builds the Core-SG once for a reference `k_max`, stores the
    corresponding HDBSCAN outputs, and allows reconstructing the MST and the
    hierarchy for other values `k <= k_max`.

    Notes
    -----
    - All `hdbscan_kwargs` are forwarded to `build_core_sg_from_data(...)`.
    - Only a filtered subset of these arguments is forwarded to
      the internal HDBSCAN adapter used for tree-to-label conversion.
    - HDBSCAN-specific integration is centralized in `hdbscan_adapter`.
    """

    _TREE_TO_LABELS_KEYS = {
        "cluster_selection_method",
        "allow_single_cluster",
        "match_reference_implementation",
        "cluster_selection_epsilon",
        "cluster_selection_persistence",
        "max_cluster_size",
        "cluster_selection_epsilon_max",
    }

    def __init__(
        self,
        metric: str = "euclidean",
        p: int = 2,
        verbose: int = 0,
        progress_callback: ProgressCallback | None = None,
        no_noise: bool = True,
        noise_label_strategy: str = "mst_label_propagation",
        algorithm: str = "core-sg",
        random_state: int | np.random.RandomState | None = None,
        approx_knn_kwargs: dict[str, Any] | None = None,
        **hdbscan_kwargs: Any,
    ) -> None:
        """
        Initialize the CoreSG object.

        Parameters
        ----------
        metric : str, default="euclidean"
            Distance metric used to build the pairwise distances / Core-SG.
        p : int, default=2
            Power parameter for metrics such as Minkowski.
        verbose : int, default=0
            Verbosity level. If greater than zero, progress messages are
            printed during the main computational steps.
        progress_callback : callable, default=None
            Optional callback called as
            `progress_callback(event, elapsed, info)` after timed steps.
        no_noise : bool, default=True
            If True, points labeled as `-1` after hierarchy extraction are
            optionally reassigned through a post-processing strategy. This
            affects only `labels_`.
        noise_label_strategy : str, default="mst_label_propagation"
            Name of the post-processing strategy used when `no_noise=True`.
            The current implementation supports only
            `"mst_label_propagation"`, inspired by the density-connectivity
            label propagation view described by Gertrudes et al. (2019),
            "A unified view of density-based methods for semi-supervised
            clustering and classification".
        **hdbscan_kwargs : Any
            Extra keyword arguments passed to `build_core_sg_from_data(...)`.
            A filtered subset is also reused in `_tree_to_labels(...)`.
        """
        if not isinstance(no_noise, bool):
            raise TypeError("no_noise must be a boolean value.")
        if not isinstance(noise_label_strategy, str):
            raise TypeError("noise_label_strategy must be a string.")
        if algorithm not in {"core-sg", "score-sg"}:
            raise ValueError("algorithm must be one of {'core-sg', 'score-sg'}.")
        if approx_knn_kwargs is not None and not isinstance(approx_knn_kwargs, dict):
            raise TypeError("approx_knn_kwargs must be a dictionary or None.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be an integer greater than or equal to 0.")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None.")

        self.metric = metric
        self.p = p
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.no_noise = no_noise
        self.noise_label_strategy = noise_label_strategy
        self.algorithm = algorithm
        self.random_state = random_state
        self.approx_knn_kwargs = (
            None if approx_knn_kwargs is None else dict(approx_knn_kwargs)
        )
        self.hdbscan_kwargs = dict(hdbscan_kwargs)
        build_noise_handler(self.noise_label_strategy, c=5)

        # Global fit metadata
        self.n_samples_ = None
        self.k_max_ = None
        self._raw_data_ = None

        # Cached artifacts for k_max
        self._condensed_tree_k_max_array_ = None
        self.labels_k_max_ = None
        self.probabilities_k_max_ = None
        self.cluster_persistence_k_max_ = None
        self._single_linkage_tree_k_max_array_ = None
        self._min_spanning_tree_k_max_array_ = None

        # Shared Core-SG structures
        self.support_graph_ = None
        self.metric_edges_ = None
        self.core_distances_ = None
        self._dense_distance_matrix_ = None
        self._tree_to_labels_data_ = None
        self._score_sg_anti_hubs = None

        # Current artifacts for an extracted k
        self.labels_ = None
        self.probabilities_ = None
        self.cluster_persistence_ = None
        self._condensed_tree_array_ = None
        self._single_linkage_tree_array_ = None
        self._min_spanning_tree_array_ = None

    def get_core_sg_mutual_reachability_distance(self, k: int):
        self._validate_k(k)
        return core_sg_mutual_reachability_distance(
            self.support_graph_,
            self.metric_edges_,
            self.core_distances_,
            self.n_samples_,
            self.k_max_,
            k,
        )

    def get_core_distance(self, k):
        self._validate_k(k)
        core_k = self.core_distances_[:, k - 1]
        core_k = np.ascontiguousarray(core_k, dtype=np.float64)
        return core_k

    def _raise_algorithm_specific_attribute_error(
        self, attribute_name: str, *, algorithm: str
    ) -> None:
        raise AttributeError(
            f"Attribute '{attribute_name}' is available only when "
            f"algorithm='{algorithm}'. Current algorithm is '{self.algorithm}'."
        )

    @property
    def distance_matrix_(self):
        if self.algorithm != "core-sg":
            self._raise_algorithm_specific_attribute_error(
                "distance_matrix_", algorithm="core-sg"
            )
        return self._dense_distance_matrix_

    @distance_matrix_.setter
    def distance_matrix_(self, value):
        self._dense_distance_matrix_ = value

    @property
    def anti_hubs_(self):
        if self.algorithm != "score-sg":
            self._raise_algorithm_specific_attribute_error(
                "anti_hubs_", algorithm="score-sg"
            )
        return self._score_sg_anti_hubs

    @anti_hubs_.setter
    def anti_hubs_(self, value):
        self._score_sg_anti_hubs = value

    def _get_tree_to_labels_kwargs(self) -> dict[str, Any]:
        """
        Extract only the keyword arguments supported by `_tree_to_labels`.

        Returns
        -------
        dict[str, Any]
            Dictionary containing only the supported keys that were explicitly
            provided and whose value is not None.
        """
        return {
            key: value
            for key, value in self.hdbscan_kwargs.items()
            if key in self._TREE_TO_LABELS_KEYS and value is not None
        }

    def _store_k_max_outputs(self, hdb_obj: Any) -> None:
        """
        Store the HDBSCAN outputs obtained during the reference fit at k_max.

        Parameters
        ----------
        hdb_obj : Any
            HDBSCAN-like fitted object returned by `build_core_sg_from_data`.
        """
        self.condensed_tree_k_max_ = hdb_obj._condensed_tree
        self.labels_k_max_ = hdb_obj.labels_
        self.probabilities_k_max_ = hdb_obj.probabilities_
        self.cluster_persistence_k_max_ = hdb_obj.cluster_persistence_
        self.single_linkage_tree_k_max_ = hdb_obj._single_linkage_tree
        self.minimum_spanning_tree_k_max_ = hdb_obj._min_spanning_tree

    def _store_k_max_outputs_from_arrays(
        self,
        *,
        labels: np.ndarray,
        probabilities: np.ndarray,
        cluster_persistence: np.ndarray,
        condensed_tree: np.ndarray,
        single_linkage_tree: np.ndarray,
        min_spanning_tree: np.ndarray,
    ) -> None:
        self.labels_k_max_ = labels
        self.probabilities_k_max_ = probabilities
        self.cluster_persistence_k_max_ = cluster_persistence
        self.condensed_tree_k_max_ = condensed_tree
        self.single_linkage_tree_k_max_ = single_linkage_tree
        self.minimum_spanning_tree_k_max_ = min_spanning_tree

    def _build_by_algorithm(
        self, X: np.ndarray, k_max: int, *, _round_distances: bool = False
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any | None, np.ndarray | None
    ]:
        if self.algorithm == "core-sg":
            return build_core_sg_from_data(
                X,
                k_max=k_max,
                metric=self.metric,
                p=self.p,
                _round_distances=_round_distances,
            ) + (None,)
        if self.algorithm == "score-sg":
            core_sg, metric_edges, core_k_list, tree_to_labels_data, anti_hubs = (
                build_score_sg_from_data(
                    X,
                    k_max=k_max,
                    metric=self.metric,
                    p=self.p,
                    random_state=self.random_state,
                    approx_knn_kwargs=self.approx_knn_kwargs,
                )
            )
            return (
                core_sg,
                metric_edges,
                core_k_list,
                tree_to_labels_data,
                None,
                anti_hubs,
            )

        raise ValueError("algorithm must be one of {'core-sg', 'score-sg'}.")

    def _extract_score_sg_k_max_outputs(self) -> None:
        if not is_graph_connected(self.support_graph_, self.n_samples_):
            raise ValueError(
                "score-sg support graph is disconnected for k_max, so an MST "
                "cannot be extracted from the constructed support graph."
            )

        mst_k_max = mst_from_core_sg(
            core_sg=self.support_graph_,
            metric_edges=self.metric_edges_,
            core_k_list=self.core_distances_,
            n_nodes=self.n_samples_,
            k=self.k_max_,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
        )
        single_linkage_tree = mst_to_single_linkage_tree(mst_k_max)
        (
            labels,
            probabilities,
            cluster_persistence,
            condensed_tree,
            single_linkage_tree,
            min_spanning_tree,
        ) = tree_to_labels(self, single_linkage_tree, mst_k_max)
        self._store_k_max_outputs_from_arrays(
            labels=labels,
            probabilities=probabilities,
            cluster_persistence=cluster_persistence,
            condensed_tree=condensed_tree,
            single_linkage_tree=single_linkage_tree,
            min_spanning_tree=min_spanning_tree,
        )

    def _should_apply_noise_handler(self) -> bool:
        return bool(
            self.no_noise
            and self.labels_ is not None
            and np.any(np.asarray(self.labels_) == -1)
        )

    def _ensure_fitted(self) -> None:
        if self.k_max_ is None or self.n_samples_ is None:
            raise AttributeError("CoreSG is not fitted yet. Run fit first.")

    def _validate_k(self, k: int) -> None:
        self._ensure_fitted()
        if k <= 0 or k > self.k_max_:
            raise ValueError("k invalid (1 <= k <= k_max).")
        if k < 2:
            raise ValueError("k must be >= 2.")

    @staticmethod
    def _validate_noise_handler_c(c: int) -> None:
        if not isinstance(c, int) or c < 1:
            raise ValueError("c must be an integer greater than or equal to 1.")

    def _apply_noise_handler(self, *, c: int) -> None:
        if not self._should_apply_noise_handler():
            return

        handler = build_noise_handler(self.noise_label_strategy, c=c)
        self.labels_ = handler.reassign(
            labels=self.labels_,
            min_spanning_tree=self._min_spanning_tree_array_,
            n_samples=self.n_samples_,
        )

    @staticmethod
    def _mst_to_dataframe(mst: np.ndarray) -> pd.DataFrame:
        """
        Convert an MST array into a typed pandas DataFrame.

        Parameters
        ----------
        mst : np.ndarray
            MST stored as an array with columns [to, from, weight].

        Returns
        -------
        pd.DataFrame
            DataFrame with typed columns: `to`, `from`, `weight`.
        """
        return pd.DataFrame(
            mst,
            columns=["to", "from", "weight"],
        ).astype(
            {
                "to": "int64",
                "from": "int64",
                "weight": "float64",
            }
        )

    @property
    def condensed_tree_(self):
        """
        Return the current extracted condensed tree wrapped as an HDBSCAN object.

        Returns
        -------
        CondensedTree
            Wrapped condensed tree for the current extracted hierarchy.

        Raises
        ------
        AttributeError
            If no current condensed tree is available.
        """

        if self._condensed_tree_array_ is not None:
            return wrap_condensed_tree(self._condensed_tree_array_, self.labels_)

        raise AttributeError(
            "No condensed tree was generated for the current k; "
            "try running extract_hierarchy_from_core_sg first."
        )

    @condensed_tree_.setter
    def condensed_tree_(self, value):
        self._condensed_tree_array_ = value

    @property
    def condensed_tree_k_max_(self):
        """
        Return the fit-time condensed tree wrapped as an HDBSCAN object.

        Returns
        -------
        CondensedTree
            Wrapped condensed tree saved during `fit`.

        Raises
        ------
        AttributeError
            If no fit-time condensed tree is available.
        """

        if self._condensed_tree_k_max_array_ is not None:
            return wrap_condensed_tree(
                self._condensed_tree_k_max_array_, self.labels_k_max_
            )

        raise AttributeError(
            "No condensed tree was saved from fit; try running fit first."
        )

    @condensed_tree_k_max_.setter
    def condensed_tree_k_max_(self, value):
        self._condensed_tree_k_max_array_ = value

    @property
    def single_linkage_tree_(self):
        """
        Return the current extracted single linkage tree wrapped as an HDBSCAN object.

        Returns
        -------
        SingleLinkageTree
            Wrapped single linkage tree for the current extracted hierarchy.

        Raises
        ------
        AttributeError
            If no current single linkage tree is available.
        """

        if self._single_linkage_tree_array_ is not None:
            return wrap_single_linkage_tree(self._single_linkage_tree_array_)

        raise AttributeError(
            "No single linkage tree was generated for the current k; "
            "try running extract_hierarchy_from_core_sg first."
        )

    @single_linkage_tree_.setter
    def single_linkage_tree_(self, value):
        self._single_linkage_tree_array_ = value

    def set_verbose(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("verbose must be an integer greater than or equal to 0.")
        self.verbose = value

    @property
    def single_linkage_tree_k_max_(self):
        """
        Return the fit-time single linkage tree wrapped as an HDBSCAN object.

        Returns
        -------
        SingleLinkageTree
            Wrapped single linkage tree saved during `fit`.

        Raises
        ------
        AttributeError
            If no fit-time single linkage tree is available.
        """

        if self._single_linkage_tree_k_max_array_ is not None:
            return wrap_single_linkage_tree(self._single_linkage_tree_k_max_array_)

        raise AttributeError(
            "No single linkage tree was saved from fit; try running fit first."
        )

    @single_linkage_tree_k_max_.setter
    def single_linkage_tree_k_max_(self, value):
        self._single_linkage_tree_k_max_array_ = value

    @property
    def minimum_spanning_tree_(self):
        """
        Return the current extracted MST wrapped as an HDBSCAN object.

        Returns
        -------
        MinimumSpanningTree or None
            Wrapped MST for the current extracted hierarchy. If no raw feature
            data is available, returns None and emits a warning.

        Raises
        ------
        AttributeError
            If no current MST is available.
        """

        if self._min_spanning_tree_array_ is None:
            raise AttributeError(
                "No minimum spanning tree was generated for the current k; "
                "try running extract_hierarchy_from_core_sg first."
            )

        if self._raw_data_ is not None:
            return wrap_minimum_spanning_tree(
                self._min_spanning_tree_array_, self._raw_data_
            )

        warn(
            "No raw data is available; this may be due to using a "
            "precomputed metric matrix. No minimum spanning tree object "
            "will be provided without raw data."
        )
        return None

    @minimum_spanning_tree_.setter
    def minimum_spanning_tree_(self, value):
        self._min_spanning_tree_array_ = value

    @property
    def minimum_spanning_tree_k_max_(self):
        """
        Return the fit-time MST wrapped as an HDBSCAN object.

        Returns
        -------
        MinimumSpanningTree or None
            Wrapped MST saved during `fit`. If no raw feature data is
            available, returns None and emits a warning.

        Raises
        ------
        AttributeError
            If no fit-time MST is available.
        """

        if self._min_spanning_tree_k_max_array_ is None:
            raise AttributeError(
                "No minimum spanning tree was saved from fit; try running fit first."
            )

        if self._raw_data_ is not None:
            return wrap_minimum_spanning_tree(
                self._min_spanning_tree_k_max_array_, self._raw_data_
            )

        warn(
            "No raw data is available; this may be due to using a "
            "precomputed metric matrix. No minimum spanning tree object "
            "will be provided without raw data."
        )
        return None

    @minimum_spanning_tree_k_max_.setter
    def minimum_spanning_tree_k_max_(self, value):
        self._min_spanning_tree_k_max_array_ = value

    def get_fitted_hdbscan_objects(self, wrapped: bool = True) -> dict[str, Any]:
        """
        Return the HDBSCAN artifacts saved during the Core-SG `fit` at `k_max`.

        Parameters
        ----------
        wrapped : bool, default=True
            If True, returns the tree artifacts wrapped using the same object
            types exposed by HDBSCAN (`CondensedTree`, `SingleLinkageTree`,
            `MinimumSpanningTree` when possible). If False, returns the raw
            internal arrays.

        Returns
        -------
        dict[str, Any]
            Dictionary with the artifacts saved during `fit`, namely:
            `labels_`, `probabilities_`, `cluster_persistence_`,
            `condensed_tree_`, `single_linkage_tree_`, and
            `minimum_spanning_tree_`.

        Raises
        ------
        AttributeError
            If the object has not been fitted yet.
        """
        if self.k_max_ is None:
            raise AttributeError("CoreSG is not fitted yet. Run fit first.")

        if wrapped:
            return {
                "labels_": self.labels_k_max_,
                "probabilities_": self.probabilities_k_max_,
                "cluster_persistence_": self.cluster_persistence_k_max_,
                "condensed_tree_": self.condensed_tree_k_max_,
                "single_linkage_tree_": self.single_linkage_tree_k_max_,
                "minimum_spanning_tree_": self.minimum_spanning_tree_k_max_,
            }

        return {
            "labels_": self.labels_k_max_,
            "probabilities_": self.probabilities_k_max_,
            "cluster_persistence_": self.cluster_persistence_k_max_,
            "condensed_tree_": self._condensed_tree_k_max_array_,
            "single_linkage_tree_": self._single_linkage_tree_k_max_array_,
            "minimum_spanning_tree_": self._min_spanning_tree_k_max_array_,
        }

    def fit(self, X: np.ndarray, k_max: int) -> "CoreSG":
        """
        Build the Core-SG support for a reference value `k_max`.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        k_max : int
            Maximum neighborhood size used to build the reusable Core-SG
            support.

        Returns
        -------
        CoreSG
            The fitted instance itself.
        """
        return self._fit(X, k_max, _round_distances=False)

    def _fit_for_tests(self, X: np.ndarray, k_max: int) -> "CoreSG":
        return self._fit(X, k_max, _round_distances=True)

    def _fit(
        self, X: np.ndarray, k_max: int, *, _round_distances: bool = False
    ) -> "CoreSG":
        """
        Build the Core-SG for a reference value `k_max`.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        k_max : int
            Maximum neighborhood size used to build the Core-SG.

        Returns
        -------
        CoreSG
            The fitted instance itself.
        """
        self.n_samples_ = X.shape[0]
        self.k_max_ = k_max

        # Segue a lógica do HDBSCAN: o wrapper do MST depende dos dados crus.
        self._raw_data_ = X

        t0 = time()
        (
            self.support_graph_,
            self.metric_edges_,
            self.core_distances_,
            tree_to_labels_data,
            hdb_obj,
            anti_hubs,
        ) = self._build_by_algorithm(X, k_max, _round_distances=_round_distances)
        self._tree_to_labels_data_ = tree_to_labels_data
        self._dense_distance_matrix_ = (
            tree_to_labels_data if self.algorithm == "core-sg" else None
        )
        self.anti_hubs_ = anti_hubs
        t1 = time()

        _emit_progress(
            "build",
            t1 - t0,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            message=f"Core-SG build done in {t1 - t0:.2f}s",
            k_max=k_max,
            algorithm=self.algorithm,
        )

        if self.algorithm == "core-sg":
            self._store_k_max_outputs(hdb_obj)
        else:
            self._extract_score_sg_k_max_outputs()
        return self

    def extract_mst_from_core_sg(self, k: int, toDF: bool = False):
        """
        Extract the minimum spanning tree for a given `k` from the Core-SG.

        Parameters
        ----------
        k : int
            Neighborhood size for which the MRD-based MST should be extracted.
        toDF : bool, default=False
            If True, returns the MST as a pandas DataFrame. Otherwise returns
            the raw NumPy array.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The extracted MST, either as a NumPy array or as a DataFrame.
        """
        self._validate_k(k)
        if self.k_max_ == k:
            if toDF:
                return self._mst_to_dataframe(self._min_spanning_tree_k_max_array_)
            return self._min_spanning_tree_k_max_array_

        t0 = time()
        try:
            mst_core = mst_from_core_sg(
                core_sg=self.support_graph_,
                metric_edges=self.metric_edges_,
                core_k_list=self.core_distances_,
                n_nodes=self.n_samples_,
                k=k,
                verbose=self.verbose,
                progress_callback=self.progress_callback,
            )
        except ValueError as exc:
            if self.algorithm == "score-sg" and "Disconex Graph" in str(exc):
                raise ValueError(
                    "score-sg support graph is disconnected, so MST extraction "
                    "is not possible for this fit."
                ) from exc
            raise
        t1 = time()

        _emit_progress(
            "extract_mst",
            t1 - t0,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            message=f"Core-SG MST K = {k} (Kruskal) done in {t1 - t0:.2f}s",
            k=k,
        )

        if toDF:
            return self._mst_to_dataframe(mst_core)

        return mst_core

    def extract_hierarchy_from_core_sg(self, k: int, c: int = 5) -> None:
        """
        Reconstruct the HDBSCAN hierarchy for a given `k` from the Core-SG.

        This method computes the MST for the requested `k`, converts it into
        a single linkage tree, and runs the HDBSCAN tree post-processing to
        populate the current attributes:
        `labels_`, `probabilities_`, `cluster_persistence_`,
        `condensed_tree_`, `single_linkage_tree_`, `minimum_spanning_tree_`.

        Parameters
        ----------
        k : int
            Neighborhood size for which the hierarchy should be extracted.
        c : int, default=5
            Number of largest edge weights retained in the top-`c` path
            signature used by the `mst_label_propagation` noise reassignment
            strategy. This affects only the optional post-processing step and
            only updates `labels_`.

        Returns
        -------
        None
            The method updates the instance attributes in place.
        """
        self._validate_k(k)
        self._validate_noise_handler_c(c)

        if self.k_max_ == k:
            (
                self.labels_,
                self.probabilities_,
                self.cluster_persistence_,
                condensed_tree,
                single_linkage_tree,
                min_spanning_tree,
            ) = self.get_fitted_hdbscan_objects(wrapped=False).values()

            self._condensed_tree_array_ = condensed_tree
            self._single_linkage_tree_array_ = single_linkage_tree
            self._min_spanning_tree_array_ = min_spanning_tree
            self._apply_noise_handler(c=c)

            return None

        min_spanning_tree = self.extract_mst_from_core_sg(k)
        single_linkage_tree = mst_to_single_linkage_tree(min_spanning_tree)

        t0 = time()
        (
            self.labels_,
            self.probabilities_,
            self.cluster_persistence_,
            condensed_tree,
            single_linkage_tree,
            min_spanning_tree,
        ) = tree_to_labels(self, single_linkage_tree, min_spanning_tree)
        t1 = time()

        self._condensed_tree_array_ = condensed_tree
        self._single_linkage_tree_array_ = single_linkage_tree
        self._min_spanning_tree_array_ = min_spanning_tree
        self._apply_noise_handler(c=c)

        _emit_progress(
            "extract_hierarchy",
            t1 - t0,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            message=f"FOSC K = {k} done in {t1 - t0:.2f}s",
            k=k,
        )

        return None
