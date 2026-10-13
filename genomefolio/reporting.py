"""Reporting helpers: run metadata and series persistence.

``write_run_meta`` records everything needed to reproduce a run (resolved config, its
hash, the git commit, the uv.lock hash, interpreter/platform, key package versions) so
every results file in ``reports/`` is traceable to the exact code and config that made it.

``save_series`` persists daily return series so sensitivity analyses (block length,
costs, subsets) can reuse them instead of re-running the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

import pandas as pd

from genomefolio.config import REPO_ROOT, Config

_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "numba", "yfinance")


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return None


def _lock_hash() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    if lock.exists():
        return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    return None


def _versions() -> dict[str, str]:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def config_to_dict(cfg: Config) -> dict:
    """Resolved config as a plain dict (tuples stringified by ``default=str`` on dump)."""
    return asdict(cfg)


def config_hash(cfg: Config) -> str:
    """Stable 16-hex hash of the resolved config."""
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_run_meta(path: str | Path, cfg: Config, extra: Mapping | None = None) -> Path:
    """Write reproducibility metadata to ``path`` (JSON) and return it."""
    meta = {
        "config": config_to_dict(cfg),
        "config_hash": config_hash(cfg),
        "seed": cfg.seed,
        "git_commit": _git_commit(),
        "uv_lock_hash": _lock_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _versions(),
    }
    if extra:
        meta.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def save_series(series_by_name: Mapping[str, pd.Series], path: str | Path) -> Path:
    """Persist a set of named daily return series to a CSV."""
    frame = pd.DataFrame(dict(series_by_name))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return path
