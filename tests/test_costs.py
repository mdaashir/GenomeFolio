"""Transaction-cost model tests (M8)."""

import pytest

from genomefolio.backtest import run_backtest
from genomefolio.config import load_config
from genomefolio.costs import transaction_cost
from genomefolio.data import align_to_common_days, load_fixture, to_returns

CFG = load_config()


def test_transaction_cost_zero_and_linear():
    assert transaction_cost(0.0) == 0.0
    assert transaction_cost(1.0, base_bps=10.0) == pytest.approx(10.0 / 1e4)


def test_spread_and_impact_add_cost():
    flat = transaction_cost(0.5, 10.0, 0.0, 0.0)
    assert transaction_cost(0.5, 10.0, 5.0, 0.0) > flat
    assert transaction_cost(0.5, 10.0, 0.0, 20.0) > flat


def test_impact_model_reduces_backtest_returns():
    from dataclasses import replace

    r = align_to_common_days(to_returns(load_fixture(CFG))).iloc[-400:, :10]
    base = run_backtest(r, CFG, method="equal_weight", cost_bps=10.0)
    impact_cfg = replace(CFG, costs=replace(CFG.costs, impact_coeff=50.0))
    impacted = run_backtest(r, impact_cfg, method="equal_weight", cost_bps=10.0)
    assert impacted.daily_returns.sum() < base.daily_returns.sum()
