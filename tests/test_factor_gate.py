"""Factor-model null gate tests (M5)."""

import numpy as np

from genomefolio.config import load_config
from genomefolio.data import align_to_common_days, load_fixture, to_returns
from genomefolio.null_test import evaluate_gate, factor_neutral_symbols
from genomefolio.symbolize import symbolize

CFG = load_config()


def _window():
    r = align_to_common_days(to_returns(load_fixture(CFG)))
    return r.iloc[-CFG.window.train_days :]


def test_factor_neutral_symbols_differ_from_raw():
    win = _window()
    raw = symbolize(win, CFG.symbolize.cutoffs)
    neutral = factor_neutral_symbols(win, CFG.symbolize.cutoffs)
    assert not raw.equals(neutral)


def test_factor_gate_returns_valid_result_and_is_stricter():
    win = _window()
    raw = symbolize(win, CFG.symbolize.cutoffs)
    neutral = factor_neutral_symbols(win, CFG.symbolize.cutoffs)
    g_raw = evaluate_gate(raw, CFG.align, CFG.cluster, CFG.null_test, 40, np.random.default_rng(CFG.seed))
    g_neutral = evaluate_gate(neutral, CFG.align, CFG.cluster, CFG.null_test, 40, np.random.default_rng(CFG.seed))
    assert isinstance(g_neutral.passed, bool)
    # removing the shared market factor cannot increase cross-stock similarity
    assert g_neutral.real_silhouette <= g_raw.real_silhouette + 1e-9
