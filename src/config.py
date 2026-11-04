"""Typed configuration loader.

All parameters live in ``config/params.yaml``; this module turns them into frozen
dataclasses so nothing downstream is hardcoded.

The pipeline is data -> symbolize -> align -> cluster -> forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"

_SCIPY_LINKAGES = {"single", "complete", "average", "weighted", "centroid", "median", "ward"}


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
class WindowConfig:
    train_days: int
    forecast_horizon: int


@dataclass(frozen=True)
class ForecastConfig:
    model: str
    max_p: int
    max_q: int
    seasonal: bool
    fallback: str


@dataclass(frozen=True)
class Config:
    seed: int
    universe: UniverseConfig
    split: SplitConfig
    symbolize: SymbolizeConfig
    align: AlignConfig
    cluster: ClusterConfig
    window: WindowConfig
    forecast: ForecastConfig
    source_path: str = field(default="", compare=False)


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in data:
        raise KeyError(f"Missing required config section: '{key}'")
    return data[key]


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate parameters from YAML.

    Returns a :class:`Config`. Raises ``ValueError`` on out-of-range or inconsistent
    values so misconfiguration surfaces at startup rather than mid-run.
    """
    path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    u = _section(raw, "universe")
    s = _section(raw, "split")
    sy = _section(raw, "symbolize")
    al = _section(raw, "align")
    cl = _section(raw, "cluster")
    w = _section(raw, "window")
    fo = _section(raw, "forecast")

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
        window=WindowConfig(
            train_days=int(w["train_days"]),
            forecast_horizon=int(w["forecast_horizon"]),
        ),
        forecast=ForecastConfig(
            model=str(fo["model"]),
            max_p=int(fo["max_p"]),
            max_q=int(fo["max_q"]),
            seasonal=bool(fo["seasonal"]),
            fallback=str(fo["fallback"]),
        ),
        source_path=str(path),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not cfg.universe.tickers:
        raise ValueError("universe.tickers must not be empty")
    if len(set(cfg.universe.tickers)) != len(cfg.universe.tickers):
        raise ValueError("universe.tickers must be unique")
    try:
        start, end = date.fromisoformat(cfg.universe.start), date.fromisoformat(cfg.universe.end)
    except ValueError as exc:
        raise ValueError("universe.start/end must be ISO dates (YYYY-MM-DD)") from exc
    if start >= end:
        raise ValueError("universe.start must be before universe.end")
    for name, cut in (
        ("symbolize.cutoffs", cfg.symbolize.cutoffs),
        ("symbolize.cutoffs_ablation", cfg.symbolize.cutoffs_ablation),
    ):
        if len(cut) != 4:
            raise ValueError(f"{name} must have exactly 4 cutoffs for the 5-symbol alphabet")
        if list(cut) != sorted(cut) or not all(0.0 < c < 1.0 for c in cut):
            raise ValueError(f"{name} must be strictly increasing and inside (0, 1)")
    if cfg.symbolize.quantile_scope not in {"per_stock", "pooled"}:
        raise ValueError("symbolize.quantile_scope must be per_stock|pooled")
    if cfg.align.match_score <= 0:
        raise ValueError("align.match_score must be > 0")
    if cfg.align.band < 0:
        raise ValueError("align.band must be >= 0")
    if cfg.align.band_sweep and min(cfg.align.band_sweep) < 0:
        raise ValueError("align.band_sweep entries must be >= 0")
    if cfg.cluster.linkage not in _SCIPY_LINKAGES:
        raise ValueError(f"cluster.linkage must be one of {sorted(_SCIPY_LINKAGES)}")
    if cfg.cluster.k_min < 2:
        raise ValueError("cluster.k_min must be >= 2")
    if cfg.cluster.k_max < cfg.cluster.k_min:
        raise ValueError("cluster.k_max must be >= cluster.k_min")
    if cfg.cluster.k_cap_divisor < 1:
        raise ValueError("cluster.k_cap_divisor must be >= 1")
    if cfg.cluster.min_cluster_size < 2:
        raise ValueError("cluster.min_cluster_size must be >= 2")
    if not (0.0 < cfg.split.tune_fraction < 1.0):
        raise ValueError("split.tune_fraction must be inside (0, 1)")
    if cfg.window.train_days < 1 or cfg.window.forecast_horizon < 1:
        raise ValueError("window.train_days and forecast_horizon must be >= 1")
    if cfg.forecast.model not in {"arima", "spline"}:
        raise ValueError("forecast.model must be arima|spline")
