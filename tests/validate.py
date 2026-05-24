from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass(frozen=True)
class CoreSGValidationReport:
    n: int
    k_max: int
    ok: bool
    missing_in_core: int


def validate_mst_in_core_sg(
    core_sg: np.ndarray, mst_hdb: np.ndarray, n: int, k: int
) -> CoreSGValidationReport:
    """
    Valida Core-SG vs HDBSCAN referência (MST exata).
    - O Core-SG calcula pairwise distances internamente.
    - O HDBSCAN referência roda com metric="precomputed" sobre a mesma matriz D.
    - Compara MST de mutual reachability (arestas + pesos).

    Retorna CoreSGValidationReport.
    """
    # --- Comparação MST: arestas + pesos ---

    d_hdb = {}
    a = 0
    ok = True

    for core in core_sg:
        max_val = max(core[:2])
        min_val = min(core[:2])
        if max_val not in d_hdb:
            d_hdb[max_val] = {}
        d_hdb[max_val][min_val] = [core[2]]
    for hdb in mst_hdb:
        max_val = max(hdb[:2])
        min_val = min(hdb[:2])
        try:
            d_hdb[max_val][min_val].append(hdb[2])
        except KeyError:
            a += 1
            ok = False

    return CoreSGValidationReport(
        n=int(n),
        k_max=int(k),
        ok=bool(ok),
        missing_in_core=int(a),
    )


def validate_mst_from_core_sg(
    mst_core: np.ndarray, mst_hdb: np.ndarray, n: int, k: int
) -> CoreSGValidationReport:
    """
    Valida Core-SG vs HDBSCAN referência (MST exata).
    - O Core-SG calcula pairwise distances internamente.
    - O HDBSCAN referência roda com metric="precomputed" sobre a mesma matriz D.
    - Compara MST de mutual reachability (arestas + pesos).

    Retorna CoreSGValidationReport.
    """
    # --- Comparação MST: arestas + pesos ---
    mst_core_copy = mst_core.copy()
    mst_hdb_copy = mst_hdb.copy()
    ok = True
    a = 0

    d_hdb = {}
    for hdb in mst_hdb_copy:
        max_hdb, min_hdb, weight_hdb = int(max(hdb[:2])), int(min(hdb[:2])), hdb[2]

        if max_hdb not in d_hdb:
            d_hdb[max_hdb] = {}
        d_hdb[max_hdb][min_hdb] = weight_hdb

    for core in mst_core_copy:
        max_core, min_core, weight_core = (
            int(max(core[:2])),
            int(min(core[:2])),
            core[2],
        )

        if max_core not in d_hdb:
            # print("Maximo")
            a += 1
        elif min_core not in d_hdb[max_core]:
            # print("Minimo")
            # print(max_core,min_core)
            a += 1
        elif abs(d_hdb[max_core][min_core] - weight_core) > 0.001:
            # print("Distancia")
            a += 1

    if float(a) / n > 0.01:
        ok = False

    return CoreSGValidationReport(
        n=int(n),
        k_max=int(k),
        ok=bool(ok),
        missing_in_core=int(a),
    )


def validate_weights_in_core_sg(
    core_sg: np.ndarray,
    mst_hdb: np.ndarray,
    D: np.ndarray,
    core_k: np.ndarray,
    n: int,
    k: int,
) -> CoreSGValidationReport:
    """
    Valida Core-SG vs HDBSCAN referência (MST exata).
    - O Core-SG calcula pairwise distances internamente.
    - O HDBSCAN referência roda com metric="precomputed" sobre a mesma matriz D.
    - Compara MST de mutual reachability (arestas + pesos).

    Retorna CoreSGValidationReport.
    """
    # --- Comparação MST: arestas + pesos ---

    d_hdb = {}
    a = 0
    ok = True

    for core in core_sg:
        max_val = max(core[:2])
        min_val = min(core[:2])
        if max_val not in d_hdb:
            d_hdb[max_val] = {}
        d_hdb[max_val][min_val] = [core[2]]

    for hdb in mst_hdb:
        max_val = max(hdb[:2])
        min_val = min(hdb[:2])
        try:
            if d_hdb[max_val][min_val] != hdb[2]:
                # print(f"Para o HDBSCAN MRD = {hdb[2]}")
                # print(f"Distancia NxN = {D[int(max_val)][int(min_val)]} || Core-Distance_{max_val} = {core_k[int(max_val)]} || Core-Distance_{min_val} = {core_k[int(min_val)]}")
                a += 1
                ok = False
        except Exception:
            a += 1
            ok = False

    return CoreSGValidationReport(
        n=int(n),
        k_max=int(k),
        ok=bool(ok),
        missing_in_core=int(a),
    )


def validate_core_sg_atributtes(
    atribute: Any, n: int, k: int, condensed: bool = False
) -> CoreSGValidationReport:
    """
    Valida Core-SG vs HDBSCAN referência (MST exata).
    - O Core-SG calcula pairwise distances internamente.
    - O HDBSCAN referência roda com metric="precomputed" sobre a mesma matriz D.
    - Compara MST de mutual reachability (arestas + pesos).

    Retorna CoreSGValidationReport.
    """
    # --- Comparação MST: arestas + pesos ---

    a = 0
    ok = True

    try:
        _ = atribute.to_pandas()

        if not condensed:
            assert _.shape[0] == n - 1
        else:
            assert _.shape[0] >= n
    except Exception:
        ok = False

    return CoreSGValidationReport(
        n=int(n),
        k_max=int(k),
        ok=bool(ok),
        missing_in_core=int(a),
    )
