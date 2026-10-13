"""Statistical evaluation helpers.

- **Probabilistic / deflated Sharpe** (Bailey & López de Prado): PSR is the probability the
  true Sharpe exceeds a benchmark given the sample length and the skew/kurtosis of returns;
  DSR is PSR evaluated at the benchmark implied by having searched over several trials, so
  it shrinks a Sharpe that came out of a large specification search.
- **Stationary bootstrap** (Politis & Romano): resamples blocks with geometric lengths, an
  alternative to the fixed/automatic block bootstrap for dependent series.

Non-annualised Sharpe is used inside PSR/DSR (per the original formulation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

PERIODS_PER_YEAR = 252
_EULER = 0.5772156649015329


def probabilistic_sharpe_ratio(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true SR > 0) for an observed non-annualised ``sr`` over ``n`` observations."""
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    denom = np.sqrt(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(norm.cdf(sr * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """E[max SR] under the null across ``n_trials`` independent trials (Gumbel approx)."""
    if n_trials < 2 or var_trials <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trials) * ((1.0 - _EULER) * a + _EULER * b))


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float) -> float:
    """PSR evaluated at ``sr_benchmark`` — the selection-adjusted Sharpe hurdle."""
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    denom = np.sqrt(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe_from_returns(daily: pd.Series, n_trials: int = 1, var_trials: float | None = None) -> dict:
    """Compute PSR, the multiple-testing hurdle and DSR for a daily return series."""
    daily = daily.dropna()
    n = len(daily)
    if n < 3:
        return {"psr": np.nan, "sr_benchmark": np.nan, "dsr": np.nan, "sharpe": np.nan}
    sd = float(daily.std(ddof=0))
    sr = float(daily.mean() / sd) if sd > 0 else 0.0
    skew = float(daily.skew())
    kurt = float(daily.kurtosis()) + 3.0  # pandas kurtosis is excess
    # variance of the trial Sharpe ratios: unavailable for a single run -> use sampling
    # variance 1/n as a conservative default.
    var_t = (1.0 / n) if var_trials is None else float(var_trials)
    sr0 = expected_max_sharpe(n_trials, var_t)
    return {
        "sharpe": sr * np.sqrt(PERIODS_PER_YEAR),
        "psr": probabilistic_sharpe_ratio(sr, n, skew, kurt),
        "sr_benchmark": sr0,
        "dsr": deflated_sharpe_ratio(sr, n, skew, kurt, sr0),
    }


def stationary_bootstrap_indices(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index draw (geometric blocks, mean length 1/p)."""
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be inside (0, 1]")
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        for k in range(length):
            if i >= n:
                break
            idx[i] = (start + k) % n
            i += 1
    return idx


def _sharpe(daily: np.ndarray) -> float:
    sd = float(np.std(daily, ddof=0))
    return float(np.mean(daily) / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 1e-12 else np.nan


def paired_stationary_bootstrap(
    method_daily: pd.Series,
    baseline_daily: pd.Series,
    n_resamples: int,
    mean_block_length: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict:
    """Paired stationary-bootstrap CIs on the Sharpe difference (same blocks both series)."""
    common = method_daily.index.intersection(baseline_daily.index)
    m = method_daily.reindex(common).to_numpy(dtype=float)
    b = baseline_daily.reindex(common).to_numpy(dtype=float)
    n = len(m)
    if n < 10:
        nan = (np.nan, np.nan)
        return {"sharpe_diff": nan}
    p = 1.0 / max(1, mean_block_length)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = stationary_bootstrap_indices(n, p, rng)
        diffs[r] = _sharpe(m[idx]) - _sharpe(b[idx])
    lo = (1.0 - confidence) / 2.0 * 100.0
    hi = (1.0 + confidence) / 2.0 * 100.0
    return {"sharpe_diff": (float(np.nanpercentile(diffs, lo)), float(np.nanpercentile(diffs, hi)))}
