"""Consensus (band-robust) clustering tests."""

import numpy as np

from genomefolio.cluster import consensus_clustering, consensus_distance, stability_ari
from genomefolio.config import load_config

CFG = load_config()


def test_consensus_distance_is_a_valid_matrix(separable_symbols):
    sym = separable_symbols(n_per_group=5, length=100)
    D = consensus_distance(sym, CFG.align, CFG.align.band_sweep)
    A = D.to_numpy()
    assert np.allclose(A, A.T)
    assert (A >= -1e-12).all()
    assert np.allclose(np.diag(A), 0.0)


def test_consensus_clustering_recovers_blocks(separable_symbols):
    sym = separable_symbols(n_per_group=5, length=120)
    res = consensus_clustering(sym, CFG.align, CFG.cluster, CFG.align.band_sweep)
    assert res.n_clusters == 2
    groups = {frozenset(v) for v in res.clusters.values()}
    assert groups == {frozenset(f"S{i}" for i in range(5)), frozenset(f"S{i}" for i in range(5, 10))}


def test_stability_ari_matches_consensus_expectation(separable_symbols):
    from genomefolio.cluster import band_sweep

    sym = separable_symbols(n_per_group=5, length=120)
    sweep = band_sweep(sym, CFG.align, CFG.cluster, CFG.align.band_sweep)
    ari = stability_ari(sweep)
    # identical block structure at every band -> bands agree
    off_diag = ari.to_numpy()[~np.eye(len(sweep), dtype=bool)]
    assert (off_diag > 0.9).all()
