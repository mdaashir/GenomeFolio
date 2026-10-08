"""Price loading and return construction.

Inputs: a :class:`~src.config.Config`, or explicit tickers/date-range/paths.
Outputs: a wide ``prices`` frame (dates x tickers, adjusted close) and a wide
``returns`` frame of simple daily returns on the common trading days.

Assumptions:
- prices are adjusted close (``auto_adjust=True``), so splits/dividends are handled
  at source;
- the universe is fixed, so survivorship bias is documented rather than hidden.

No look-ahead: a return at date ``t`` uses prices at ``t`` and ``t-1`` only. All
windowing (training windows, tune/report split) is applied downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config, REPO_ROOT

_FIXTURE_TICKERS = ("AAPL", "MSFT", "INTC", "CSCO", "JPM", "JNJ", "PG", "XOM", "KO", "DIS")


def _normalise_close(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Return a dates x tickers adjusted-close frame from a yfinance download."""
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = list(tickers)[: close.shape[1]]
    close = close.reindex(columns=list(tickers))
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    return close


def download_prices(
    tickers: tuple[str, ...], start: str, end: str
) -> pd.DataFrame:
    """Download adjusted close for ``tickers`` between ``start`` and ``end``."""
    import yfinance as yf

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    return _normalise_close(raw, tickers)


def get_prices(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Load adjusted close from the on-disk cache, downloading once if needed.

    The cache lives at ``universe.cache_dir / {start}_{end}.csv`` and is gitignored;
    only the small fixture is committed.
    """
    u = cfg.universe
    cache_dir = REPO_ROOT / u.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{u.start}_{u.end}.csv"
    if cache_path.exists() and not force:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices.reindex(columns=list(u.tickers)).sort_index()
    prices = download_prices(u.tickers, u.start, u.end)
    prices.to_csv(cache_path)
    return prices


def load_fixture(cfg: Config) -> pd.DataFrame:
    """Load the committed offline sample (>= 8 stocks x >= 3 years)."""
    path = REPO_ROOT / cfg.universe.fixture_path
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index()


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, dropping the first (all-NaN) row."""
    return prices.pct_change().iloc[1:]


def align_to_common_days(returns: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates where every stock trades, so all strings share an index."""
    return returns.dropna(how="any")


def time_split(
    returns: pd.DataFrame, tune_fraction: float, enabled: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (tune, report) by time.

    Returns the whole series as the tune set and an empty report set when
    ``enabled`` is False (the fixture path). The split is chronological, so the
    report period is untouched by tuning.
    """
    if not enabled:
        return returns, returns.iloc[0:0]
    cut = int(len(returns) * tune_fraction)
    return returns.iloc[:cut], returns.iloc[cut:]
