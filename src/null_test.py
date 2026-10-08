"""Circular-shift null gate.

Large-caps share a market factor, so nearly every pair beats an i.i.d. shuffle and a
"fraction of pairs passing" gate would always pass. We therefore:

1. Build a null by **independently circular-shifting each stock's string by a random
   offset larger than the band ``w``**. Circular shifts preserve each string's internal
   autocorrelation and volatility clustering and break only the cross-stock alignment —
   unlike i.i.d. shuffling, which destroys both and makes the null too easy to beat.
2. Run each null through the **identical constrained k selection** used for the real
   run (``src.cluster.select_constrained_k``), so the null is not handicapped by a fixed
   k while the real run takes a maximum over k.
3. **Gate**: the real silhouette must beat the configured percentile (95th) of the null
   silhouettes. The pass fraction across rebalances is later read against the ~5% chance
   rate implied by that threshold.

Per-pair FDR (Benjamini-Hochberg) is reported as a **diagnostic only**; it does not gate.

On gate failure the caller should fall back to equal weight for that rebalance and log it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.align import distance_matrix
from src.cluster import ClusteringResult, select_constrained_k
from src.config import AlignConfig, ClusterConfig, NullTestConfig


@dataclass
class NullGateResult:
    """Outcome of the null gate for one rebalance."""

    real_result: ClusteringResult
    threshold: float
    n_draws: int
    passed: bool
    null_silhouettes: np.ndarray
    fdr_significant_fraction: float = float("nan")
    pair_pvalues: np.ndarray | None = field(default=None, repr=False)

    @property
    def real_silhouette(self) -> float:
        return self.real_result.silhouette


def _shift_offset(rng: np.random.Generator, n: int, band: int) -> int:
    """Random offset strictly larger than the band, so shifts break cross-alignment."""
    lo, hi = band + 1, n - band - 1
    if hi <= lo:
        raise ValueError("string too short to circular-shift beyond the band")
    return int(rng.integers(lo, hi))


def circular_shift_frame(
    symbols: pd.DataFrame, rng: np.random.Generator, band: int
) -> pd.DataFrame:
    """Independently circular-shift every column by a random offset > band."""
    n = len(symbols)
    shifted = {}
    for col in symbols.columns:
        off = _shift_offset(rng, n, band)
        shifted[col] = np.roll(symbols[col].to_numpy(), off)
    return pd.DataFrame(shifted, index=symbols.index)


def _benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Return a boolean mask of rejected hypotheses (BH step-up)."""
    flat = pvals.ravel()
    m = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    if not below.any():
        mask = np.zeros(flat.shape, dtype=bool)
    else:
        k_max = np.max(np.nonzero(below)[0])
        cutoff = ranked[k_max]
        mask = flat <= cutoff
    return mask.reshape(pvals.shape)


def evaluate_gate(
    symbols: pd.DataFrame,
    align_cfg: AlignConfig,
    cluster_cfg: ClusterConfig,
    null_cfg: NullTestConfig,
    n_draws: int,
    rng: np.random.Generator,
) -> NullGateResult:
    """Gate the real clustering against circular-shift null clusterings."""
    real_D = distance_matrix(symbols, align_cfg)
    real = select_constrained_k(real_D, cluster_cfg, cluster_cfg.linkage)
    real_dist = real_D.to_numpy()

    n = len(symbols.columns)
    exceed = np.zeros((n, n))  # count of null draws with dist <= real dist (per pair)
    null_sils = np.empty(n_draws, dtype=float)

    for d in range(n_draws):
        shifted = circular_shift_frame(symbols, rng, align_cfg.band)
        null_D = distance_matrix(shifted, align_cfg)
        null_res = select_constrained_k(null_D, cluster_cfg, cluster_cfg.linkage)
        null_sils[d] = null_res.silhouette
        exceed += (null_D.to_numpy() <= real_dist).astype(float)

    threshold = float(np.percentile(null_sils, null_cfg.gate_percentile))
    passed = bool(real.silhouette > threshold)

    # Diagnostic-only per-pair FDR (upper triangle, excluding diagonal).
    iu = np.triu_indices(n, k=1)
    pvals = (exceed[iu] + 1.0) / (n_draws + 1.0)
    sig = _benjamini_hochberg(pvals, null_cfg.fdr_alpha)
    pair_pvalues = np.full((n, n), np.nan)
    pair_pvalues[iu] = pvals
    pair_pvalues[(iu[1], iu[0])] = pvals
    return NullGateResult(
        real_result=real,
        threshold=threshold,
        n_draws=n_draws,
        passed=passed,
        null_silhouettes=null_sils,
        fdr_significant_fraction=float(sig.mean()) if sig.size else float("nan"),
        pair_pvalues=pair_pvalues,
    )
