from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers import make_sample_X


class _WrappedTree:
    def __init__(self, raw, aux=None):
        self.raw = np.asarray(raw)
        self.aux = aux

    def to_pandas(self):
        return pd.DataFrame(self.raw)


class FakeHDBSCAN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, D):
        D = np.asarray(D, dtype=np.float64)
        n = D.shape[0]
        chain_u = np.arange(n - 1, dtype=np.int64)
        chain_v = np.arange(1, n, dtype=np.int64)
        chain_w = D[chain_u, chain_v]
        mst = np.column_stack([chain_u, chain_v, chain_w]).astype(np.float64)
        self._min_spanning_tree = mst
        self._single_linkage_tree = mst.copy()
        self._condensed_tree = np.column_stack(
            [
                np.arange(n, dtype=np.float64),
                np.arange(n, dtype=np.float64),
                np.ones(n, dtype=np.float64),
                np.full(n, 2.0, dtype=np.float64),
            ]
        )
        self.labels_ = np.arange(n) % 2
        self.probabilities_ = np.linspace(0.5, 1.0, n)
        self.cluster_persistence_ = np.array([0.6, 0.8], dtype=np.float64)
        return self


@pytest.fixture()
def fake_hdbscan_modules(monkeypatch):
    fake_root = types.ModuleType("hdbscan")
    fake_hdbscan_mod = types.ModuleType("hdbscan.hdbscan_")
    fake_linkage_mod = types.ModuleType("hdbscan._hdbscan_linkage")
    fake_plots_mod = types.ModuleType("hdbscan.plots")

    def fake_tree_to_labels(D, single_linkage_tree, **kwargs):
        n = np.asarray(D).shape[0]
        labels = np.arange(n) % 2
        probabilities = np.linspace(0.5, 1.0, n)
        cluster_persistence = np.array([0.7, 0.9], dtype=np.float64)
        condensed = np.column_stack(
            [
                np.arange(n, dtype=np.float64),
                np.arange(n, dtype=np.float64),
                np.ones(n, dtype=np.float64),
                np.full(n, 2.0, dtype=np.float64),
            ]
        )
        return (
            labels,
            probabilities,
            cluster_persistence,
            condensed,
            np.asarray(single_linkage_tree),
        )

    fake_hdbscan_mod._tree_to_labels = fake_tree_to_labels
    fake_linkage_mod.label = lambda mst: np.asarray(mst, dtype=np.float64)
    fake_plots_mod.MinimumSpanningTree = _WrappedTree
    fake_plots_mod.SingleLinkageTree = _WrappedTree
    fake_plots_mod.CondensedTree = _WrappedTree
    fake_root.HDBSCAN = FakeHDBSCAN

    monkeypatch.setitem(sys.modules, "hdbscan", fake_root)
    monkeypatch.setitem(sys.modules, "hdbscan.hdbscan_", fake_hdbscan_mod)
    monkeypatch.setitem(sys.modules, "hdbscan._hdbscan_linkage", fake_linkage_mod)
    monkeypatch.setitem(sys.modules, "hdbscan.plots", fake_plots_mod)

    for name in list(sys.modules):
        if name == "core_sg" or name.startswith("core_sg."):
            del sys.modules[name]

    return {
        "root": fake_root,
        "hdbscan_": fake_hdbscan_mod,
        "linkage": fake_linkage_mod,
        "plots": fake_plots_mod,
    }


@pytest.fixture()
def core_sg_module(fake_hdbscan_modules):
    return importlib.import_module("core_sg.core_sg")


@pytest.fixture()
def noise_handler_module(fake_hdbscan_modules):
    return importlib.import_module("core_sg.noise_handler")


@pytest.fixture()
def sample_X() -> np.ndarray:
    return make_sample_X()
