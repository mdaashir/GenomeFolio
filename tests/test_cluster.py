import numpy as np
import pandas as pd

from src.cluster import select_constrained_k, stability_ari
from src.config import load_config

CFG = load_config().cluster
N = 12


def _block_distance(n=N, within=0.1, between=0.9):
    """Distance matrix with two clear blocks of size n/2."""
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    D = np.where(labels[:, None] == labels[None, :], within, between)
    np.fill_diagonal(D, 0.0)
    cols = [f"S{i}" for i in range(n)]
    return pd.DataFrame(D, index=cols, columns=cols)


def test_recovers_block_structure():
    D = _block_distance()
    res = select_constrained_k(D, CFG)
    assert res.n_clusters == 2
    assert res.min_size_ok
    sizes = sorted(len(v) for v in res.clusters.values())
    assert sizes == [N // 2, N // 2]
    assert res.silhouette > 0.5


def test_k_never_exceeds_cap():
    D = _block_distance()
    res = select_constrained_k(D, CFG)
    assert res.n_clusters <= CFG.k_cap(N)


def test_no_undersized_clusters_when_possible():
    D = _block_distance()
    res = select_constrained_k(D, CFG)
    assert all(len(v) >= CFG.min_cluster_size for v in res.clusters.values())


def test_small_universe_falls_back_without_multiple_clusters():
    D = _block_distance(n=4)
    res = select_constrained_k(D, CFG)
    # k_cap(4) = 1 < k_min -> single cluster, flagged
    assert res.n_clusters == 1
    assert not res.min_size_ok


def test_stability_ari_diagonal_is_one():
    from src.cluster import ClusteringResult

    labels = pd.Series([0, 0, 1, 1], index=list("abcd"))
    r = ClusteringResult(labels=labels, n_clusters=2, silhouette=0.5, min_size_ok=True)
    M = stability_ari({2: r, 5: r})
    assert np.allclose(np.diag(M.to_numpy()), 1.0)


def test_warns_when_no_valid_k_exists():
    # 6 stocks with one outlier: k_cap=2, and the only cut splits 1 vs 5 -> undersized
    n = 6
    D = np.full((n, n), 0.1)
    D[0, :] = D[:, 0] = 1.0
    np.fill_diagonal(D, 0.0)
    cols = [f"S{i}" for i in range(n)]
    frame = pd.DataFrame(D, index=cols, columns=cols)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = select_constrained_k(frame, CFG)
    assert not res.min_size_ok
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
