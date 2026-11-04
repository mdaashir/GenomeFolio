"""Figure tests: every phase produces a non-empty matplotlib figure."""

import matplotlib.pyplot as plt

from src import plots
from src.config import load_config
from src.data import align_to_common_days, load_fixture, to_returns
from src.pipeline import run_pipeline

CFG = load_config()


def _window_and_result():
    returns = align_to_common_days(to_returns(load_fixture(CFG)))
    window = returns.iloc[-CFG.window.train_days :]
    return window, run_pipeline(window, CFG)


def test_every_phase_produces_a_figure():
    window, result = _window_and_result()
    figures = [
        plots.cumulative_returns(window),
        plots.symbol_heatmap(result.symbols),
        plots.symbol_distribution(result.symbols),
        plots.score_heatmap(result.scores),
        plots.distance_heatmap(result.distance),
        plots.dendrogram(result.distance, CFG.cluster.linkage),
        plots.silhouette_curve(result.distance, CFG.cluster),
        plots.cluster_sizes(result.clusters),
        plots.forecast_panels(
            result.composite_returns, result.forecasts, CFG.window.forecast_horizon
        ),
    ]
    for fig in figures:
        assert fig.axes
        plt.close(fig)


def test_band_stability_returns_figure_and_tables():
    _window, result = _window_and_result()
    fig, summary, ari = plots.band_stability(
        result.symbols, CFG.align, CFG.cluster, CFG.align.band_sweep
    )
    assert fig.axes
    assert set(summary.columns) == {"k", "silhouette"}
    assert list(ari.columns) == list(CFG.align.band_sweep)
    plt.close(fig)
