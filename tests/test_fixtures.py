"""Toy-fixture tests: a known two-block structure through distance, clustering and the gate."""

import numpy as np

from genomefolio.align import distance_matrix
from genomefolio.cluster import select_constrained_k
from genomefolio.config import load_config
from genomefolio.null_test import evaluate_gate

CFG = load_config()


def test_block_distance_fixture_clusters_into_two(block_distance):
    res = select_constrained_k(block_distance(), CFG.cluster, CFG.cluster.linkage)
    assert res.n_clusters == 2
    assert res.min_size_ok


def test_separable_symbols_cluster_into_the_right_two_groups(separable_symbols):
    sym = separable_symbols(n_per_group=5, length=120)
    D = distance_matrix(sym, CFG.align)
    res = select_constrained_k(D, CFG.cluster, CFG.cluster.linkage)
    assert res.n_clusters == 2
    groups = {frozenset(v) for v in res.clusters.values()}
    assert groups == {frozenset(f"S{i}" for i in range(5)), frozenset(f"S{i}" for i in range(5, 10))}


def test_gate_passes_on_known_block_symbols(separable_symbols):
    sym = separable_symbols(n_per_group=5, length=80)
    res = evaluate_gate(
        sym, CFG.align, CFG.cluster, CFG.null_test, n_draws=40, rng=np.random.default_rng(CFG.seed)
    )
    assert res.passed
    assert res.real_silhouette > res.threshold
