"""Transaction-cost model.

Flat basis points are an idealisation. This adds a half-spread term and a concave
square-root **market-impact** term:

    cost_fraction = (base_bps + spread_bps)/1e4 * turnover
                  + impact_coeff/1e4 * turnover**1.5

``turnover`` is ``sum |dw|`` (total traded weight, buys plus sells). ``impact_coeff`` is
in "bps per turnover^0.5" units; the default 0 reduces to the flat linear model.
"""

from __future__ import annotations


def transaction_cost(
    turnover: float, base_bps: float = 10.0, spread_bps: float = 0.0, impact_coeff: float = 0.0
) -> float:
    """Fractional cost of trading ``turnover``."""
    if turnover <= 0.0:
        return 0.0
    linear = (base_bps + spread_bps) / 1e4 * turnover
    impact = impact_coeff / 1e4 * turnover**1.5
    return float(linear + impact)


def cost_sweep(turnovers, base_bps: float, sweep_bps) -> dict[float, float]:
    """Mean fractional cost at each ``base_bps`` level in ``sweep_bps``."""
    import numpy as np

    turnovers = np.asarray(list(turnovers), dtype=float)
    return {float(bps): float(np.mean([transaction_cost(t, bps) for t in turnovers])) for bps in sweep_bps}
