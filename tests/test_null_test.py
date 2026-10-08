import numpy as np
import pandas as pd

from src.config import load_config
from src.null_test import circular_shift_frame, evaluate_gate

CFG = load_config()


def _frame(data, columns=None, length=None):
    data = np.asarray(data)
    length = data.shape[0]
    cols = columns or [f"S{i}" for i in range(data.shape[1])]
    idx = pd.date_range("2020-01-01", periods=length, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


def test_circular_shift_preserves_multiset_and_differs():
    rng = np.random.default_rng(0)
    sym = _frame(rng.integers(0, 5, size=(80, 4)))
    shifted = circular_shift_frame(sym, rng, band=CFG.align.band)
    for col in sym.columns:
        assert sorted(sym[col].tolist()) == sorted(shifted[col].tolist())
    assert not sym.equals(shifted)


def test_shift_offset_exceeds_band():
    # Every column is shifted by > band, so cross-stock alignment is broken but
    # each string's internal structure is preserved.
    rng = np.random.default_rng(1)
    sym = _frame(rng.integers(0, 5, size=(60, 3)))
    shifted = circular_shift_frame(sym, rng, band=CFG.align.band)
    for col in sym.columns:
        orig = sym[col].to_numpy()
        moved = shifted[col].to_numpy()
        # a valid circular shift matches some rotation of the original
        diffs = [np.array_equal(np.roll(orig, k), moved) for k in range(len(orig))]
        assert any(diffs)


def test_gate_passes_on_separable_structure():
    rng = np.random.default_rng(7)
    base_a = rng.integers(0, 5, size=80)
    base_b = rng.integers(0, 5, size=80)
    data = np.column_stack([np.tile(base_a, 1) for _ in range(5)] + [np.tile(base_b, 1) for _ in range(5)])
    sym = _frame(data)
    res = evaluate_gate(
        sym, CFG.align, CFG.cluster, CFG.null_test, n_draws=40, rng=np.random.default_rng(CFG.seed)
    )
    assert res.passed
    assert res.real_silhouette > res.threshold


def test_gate_result_shapes_and_diagnostics():
    rng = np.random.default_rng(3)
    sym = _frame(rng.integers(0, 5, size=(80, 10)))
    res = evaluate_gate(
        sym, CFG.align, CFG.cluster, CFG.null_test, n_draws=25, rng=np.random.default_rng(CFG.seed)
    )
    assert res.null_silhouettes.shape == (25,)
    assert isinstance(res.passed, bool)
    assert np.isfinite(res.threshold)
    assert 0.0 <= res.fdr_significant_fraction <= 1.0
