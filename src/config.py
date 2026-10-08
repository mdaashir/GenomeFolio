"""Typed configuration loader.

All parameters live in ``config/params.yaml`` (and ``config/sectors.yaml``); this
module turns them into frozen dataclasses so nothing downstream is hardcoded.

Inputs: a path to a YAML file (defaults to ``config/params.yaml`` relative to the
repo root).
Outputs: a :class:`Config` (and, for :func:`load_sectors`, a ticker -> group map).
Assumptions: the YAML schema matches the dataclasses below; values are validated
here so bad config fails fast and loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"
DEFAULT_SECTORS_PATH = REPO_ROOT / "config" / "sectors.yaml"


def _tuple(values: Any) -> tuple:
    return tuple(values) if values is not None else ()


@dataclass(frozen=True)
class UniverseConfig:
    tickers: tuple[str, ...]
    start: str
    end: str
    cache_dir: str
    fixture_path: str
    min_history_years: int


@dataclass(frozen=True)
class SplitConfig:
    enabled: bool
    tune_fraction: float


@dataclass(frozen=True)
class SymbolizeConfig:
    cutoffs: tuple[float, ...]
    cutoffs_ablation: tuple[float, ...]
    market_neutral: bool
    quantile_scope: str = "per_stock"


@dataclass(frozen=True)
class AlignConfig:
    match_score: float
    gap_open: float
    gap_extend: float
    band: int
    band_sweep: tuple[int, ...]
    use_numba: bool


@dataclass(frozen=True)
class ClusterConfig:
    linkage: str
    k_min: int
    k_max: int
    k_cap_divisor: int
    min_cluster_size: int

    def k_cap(self, n: int) -> int:
        """Largest k allowed for ``n`` stocks: ``min(k_max, floor(n / divisor))``."""
        return min(self.k_max, n // self.k_cap_divisor)


@dataclass(frozen=True)
class NullTestConfig:
    gate_percentile: float
    n_draws_tuning: int
    n_draws_report: int
    fdr_alpha: float


@dataclass(frozen=True)
class WindowConfig:
    train_days: int
    rebalance_days: int
    forecast_horizon: int


@dataclass(frozen=True)
class ForecastConfig:
    model: str
    max_p: int
    max_q: int
    seasonal: bool
    fallback: str


@dataclass(frozen=True)
class PortfolioConfig:
    cluster_method: str
    within_cluster: str
    cap_floor: float
    cap_factor: float
    forecast_tilt_strength: float

    def cap(self, n: int) -> float:
        """Per-stock weight cap ``max(cap_floor, cap_factor / n)``.

        The floor alone (e.g. 10%) is infeasible for small universes: 8 x 10% = 80%
        leaves no fully invested portfolio, so the factor term lifts it when needed.
        """
        return max(self.cap_floor, self.cap_factor / n)


@dataclass(frozen=True)
class CostsConfig:
    base_bps: float
    sweep_bps: tuple[float, ...]


@dataclass(frozen=True)
class BootstrapConfig:
    block_length: int
    use_politis_white: bool
    n_resamples: int
    confidence: float


@dataclass(frozen=True)
class BacktestConfig:
    gate_passed_only_descriptive: bool


@dataclass(frozen=True)
class Config:
    seed: int
    universe: UniverseConfig
    split: SplitConfig
    symbolize: SymbolizeConfig
    align: AlignConfig
    cluster: ClusterConfig
    null_test: NullTestConfig
    window: WindowConfig
    forecast: ForecastConfig
    portfolio: PortfolioConfig
    costs: CostsConfig
    bootstrap: BootstrapConfig
    backtest: BacktestConfig
    source_path: str = field(default="", compare=False)


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in data:
        raise KeyError(f"Missing required config section: '{key}'")
    return data[key]


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate parameters from YAML.

    Returns a :class:`Config`. Raises ``ValueError`` on out-of-range or inconsistent
    values so misconfiguration surfaces at startup rather than mid-backtest.
    """
    path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    u = _section(raw, "universe")
    s = _section(raw, "split")
    sy = _section(raw, "symbolize")
    al = _section(raw, "align")
    cl = _section(raw, "cluster")
    nt = _section(raw, "null_test")
    w = _section(raw, "window")
    fo = _section(raw, "forecast")
    po = _section(raw, "portfolio")
    co = _section(raw, "costs")
    bo = _section(raw, "bootstrap")
    bt = _section(raw, "backtest")

    cfg = Config(
        seed=int(raw["seed"]),
        universe=UniverseConfig(
            tickers=tuple(u["tickers"]),
            start=str(u["start"]),
            end=str(u["end"]),
            cache_dir=str(u["cache_dir"]),
            fixture_path=str(u["fixture_path"]),
            min_history_years=int(u["min_history_years"]),
        ),
        split=SplitConfig(enabled=bool(s["enabled"]), tune_fraction=float(s["tune_fraction"])),
        symbolize=SymbolizeConfig(
            cutoffs=tuple(float(c) for c in sy["cutoffs"]),
            cutoffs_ablation=tuple(float(c) for c in sy["cutoffs_ablation"]),
            market_neutral=bool(sy["market_neutral"]),
            quantile_scope=str(sy.get("quantile_scope", "per_stock")),
        ),
        align=AlignConfig(
            match_score=float(al["match_score"]),
            gap_open=float(al["gap_open"]),
            gap_extend=float(al["gap_extend"]),
            band=int(al["band"]),
            band_sweep=tuple(int(b) for b in al["band_sweep"]),
            use_numba=bool(al["use_numba"]),
        ),
        cluster=ClusterConfig(
            linkage=str(cl["linkage"]),
            k_min=int(cl["k_min"]),
            k_max=int(cl["k_max"]),
            k_cap_divisor=int(cl["k_cap_divisor"]),
            min_cluster_size=int(cl["min_cluster_size"]),
        ),
        null_test=NullTestConfig(
            gate_percentile=float(nt["gate_percentile"]),
            n_draws_tuning=int(nt["n_draws_tuning"]),
            n_draws_report=int(nt["n_draws_report"]),
            fdr_alpha=float(nt["fdr_alpha"]),
        ),
        window=WindowConfig(
            train_days=int(w["train_days"]),
            rebalance_days=int(w["rebalance_days"]),
            forecast_horizon=int(w["forecast_horizon"]),
        ),
        forecast=ForecastConfig(
            model=str(fo["model"]),
            max_p=int(fo["max_p"]),
            max_q=int(fo["max_q"]),
            seasonal=bool(fo["seasonal"]),
            fallback=str(fo["fallback"]),
        ),
        portfolio=PortfolioConfig(
            cluster_method=str(po["cluster_method"]),
            within_cluster=str(po["within_cluster"]),
            cap_floor=float(po["cap_floor"]),
            cap_factor=float(po["cap_factor"]),
            forecast_tilt_strength=float(po["forecast_tilt_strength"]),
        ),
        costs=CostsConfig(
            base_bps=float(co["base_bps"]),
            sweep_bps=tuple(float(b) for b in co["sweep_bps"]),
        ),
        bootstrap=BootstrapConfig(
            block_length=int(bo["block_length"]),
            use_politis_white=bool(bo["use_politis_white"]),
            n_resamples=int(bo["n_resamples"]),
            confidence=float(bo["confidence"]),
        ),
        backtest=BacktestConfig(
            gate_passed_only_descriptive=bool(bt["gate_passed_only_descriptive"]),
        ),
        source_path=str(path),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not cfg.universe.tickers:
        raise ValueError("universe.tickers must not be empty")
    for name, cut in (
        ("symbolize.cutoffs", cfg.symbolize.cutoffs),
        ("symbolize.cutoffs_ablation", cfg.symbolize.cutoffs_ablation),
    ):
        if len(cut) != 4:
            raise ValueError(f"{name} must have exactly 4 cutoffs for the 5-symbol alphabet")
        if list(cut) != sorted(cut) or not all(0.0 < c < 1.0 for c in cut):
            raise ValueError(f"{name} must be strictly increasing and inside (0, 1)")
    if cfg.cluster.k_min < 2:
        raise ValueError("cluster.k_min must be >= 2")
    if cfg.cluster.k_max < cfg.cluster.k_min:
        raise ValueError("cluster.k_max must be >= cluster.k_min")
    if cfg.cluster.k_cap_divisor < 1:
        raise ValueError("cluster.k_cap_divisor must be >= 1")
    if cfg.cluster.min_cluster_size < 2:
        raise ValueError("cluster.min_cluster_size must be >= 2")
    if cfg.align.band < 0:
        raise ValueError("align.band must be >= 0")
    if not (0.0 < cfg.split.tune_fraction < 1.0):
        raise ValueError("split.tune_fraction must be inside (0, 1)")
    if not (0.0 < cfg.null_test.gate_percentile < 100.0):
        raise ValueError("null_test.gate_percentile must be inside (0, 100)")
    if cfg.portfolio.cluster_method not in {"equal", "risk_parity", "forecast_tilt"}:
        raise ValueError("portfolio.cluster_method must be equal|risk_parity|forecast_tilt")
    if cfg.symbolize.quantile_scope not in {"per_stock", "pooled"}:
        raise ValueError("symbolize.quantile_scope must be per_stock|pooled")
    if cfg.portfolio.within_cluster not in {"equal", "inverse_vol"}:
        raise ValueError("portfolio.within_cluster must be equal|inverse_vol")
    if cfg.costs.base_bps < 0:
        raise ValueError("costs.base_bps must be >= 0")
    if not (0.0 < cfg.bootstrap.confidence < 1.0):
        raise ValueError("bootstrap.confidence must be inside (0, 1)")


def load_sectors(path: str | Path | None = None) -> dict[str, tuple[str, ...]]:
    """Load the ticker -> sector-group mapping used by the sector-cluster control.

    Returns a dict of group name -> tuple of tickers. Raises ``ValueError`` if any
    group has fewer than 3 stocks or there are not 3-6 groups, so the control obeys
    the same size rules as the alignment clusters.
    """
    path = Path(path) if path is not None else DEFAULT_SECTORS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    sectors = _section(raw, "sectors")
    groups = {str(k): tuple(v) for k, v in sectors.items()}
    if not (3 <= len(groups) <= 6):
        raise ValueError(f"sectors.yaml must define 3-6 groups, got {len(groups)}")
    for name, members in groups.items():
        if len(members) < 3:
            raise ValueError(f"sector group '{name}' has {len(members)} stocks (< 3)")
    return groups
