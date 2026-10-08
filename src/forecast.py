"""Per-cluster return forecasting.

Forecast each cluster's equal-weight composite **cumulative** return over the next
~``forecast_horizon`` trading days (the monthly rebalance horizon), not a 1-day-ahead
figure. Daily returns are only weakly predictable, so the output is intended as a
*modest* weight tilt, never a strong signal.

ARIMA (default):
- ``d`` is fixed by a stationarity test (ADF) on the training window only (0 or 1);
- ``(p, q)`` chosen by AIC over a small grid with ``p, q <= max_p/max_q``;
- if no fit converges, fall back to the training-window historical mean.

Spline (alternative): extrapolate a smoothing spline of the window's cumulative return.

No look-ahead: the caller passes the current training window only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ForecastConfig


@dataclass
class ClusterForecast:
    """Forecast for one cluster over the rebalance horizon."""

    expected_return: float
    model: str
    order: tuple[int, int, int] | None = None
    fallback: bool = False


def _stationarity_d(series: np.ndarray) -> int:
    """ADF test on the training window: d = 0 if already stationary else 1."""
    if series.size < 20 or np.allclose(series, series[0]):
        return 0
    try:
        from statsmodels.tsa.stattools import adfuller

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pvalue = adfuller(series, autolag="AIC")[1]
        return 0 if pvalue < 0.05 else 1
    except Exception:
        return 0


def _arima_forecast(series: np.ndarray, cfg: ForecastConfig, horizon: int) -> ClusterForecast:
    from statsmodels.tsa.arima.model import ARIMA

    mean_fallback = ClusterForecast(
        expected_return=float(np.mean(series) * horizon),
        model="historical_mean",
        fallback=True,
    )
    if series.size < 30 or np.allclose(series, series[0]):
        return mean_fallback

    d = _stationarity_d(series)
    best = None
    best_aic = np.inf
    best_order = None
    for p in range(cfg.max_p + 1):
        for q in range(cfg.max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(series, order=(p, d, q)).fit()
                if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                    best, best_aic, best_order = fitted, fitted.aic, (p, d, q)
            except Exception:
                continue
    if best is None:
        return mean_fallback
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            steps = best.forecast(steps=horizon)
        return ClusterForecast(
            expected_return=float(np.sum(np.asarray(steps, dtype=float))),
            model="arima",
            order=best_order,
            fallback=False,
        )
    except Exception:
        return mean_fallback


def _spline_forecast(series: np.ndarray, horizon: int) -> ClusterForecast:
    from scipy.interpolate import UnivariateSpline

    mean_fallback = ClusterForecast(
        expected_return=float(np.mean(series) * horizon), model="historical_mean", fallback=True
    )
    if series.size < 30 or np.allclose(series, series[0]):
        return mean_fallback
    try:
        cum = np.cumsum(series)
        x = np.arange(cum.size, dtype=float)
        spline = UnivariateSpline(x, cum, k=3, s=len(x))
        future = np.arange(cum.size, cum.size + horizon, dtype=float)
        step = spline(future) - spline(np.array([cum.size - 1.0]))
        return ClusterForecast(expected_return=float(step[-1]), model="spline", fallback=False)
    except Exception:
        return mean_fallback


def forecast_series(series, cfg: ForecastConfig, horizon: int) -> ClusterForecast:
    """Forecast one composite return series over ``horizon`` days."""
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return ClusterForecast(expected_return=0.0, model="empty", fallback=True)
    if cfg.model == "spline":
        return _spline_forecast(arr, horizon)
    return _arima_forecast(arr, cfg, horizon)


def forecast_clusters(
    composite_returns: pd.DataFrame, cfg: ForecastConfig, horizon: int
) -> dict[int, ClusterForecast]:
    """Forecast each column (cluster composite) of ``composite_returns``."""
    return {col: forecast_series(composite_returns[col], cfg, horizon) for col in composite_returns.columns}
