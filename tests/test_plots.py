"""Plot-helper tests."""

import pandas as pd

from genomefolio.plots import cumulative_returns_plot, turnover_bar_plot


def test_plots_write_non_empty_pngs(tmp_path):
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    a = pd.Series(0.001, index=idx)
    b = pd.Series(-0.0005, index=idx)
    p1 = cumulative_returns_plot({"a": a, "b": b}, tmp_path / "cum.png")
    assert p1.exists() and p1.stat().st_size > 0
    p2 = turnover_bar_plot(pd.Series([0.1, 0.2, 0.3]), tmp_path / "turn.png")
    assert p2.exists() and p2.stat().st_size > 0
