"""Evaluate a locked pre-registration spec once (M10).

Loads ``reports/preregistration_<name>.json`` (written by ``preregister.py``), runs the
frozen arms, and reports the primary comparison with a paired CI and a verdict per the
spec's decision rule. One shot — do not tweak and re-run.

Usage::

    uv run python scripts/run_preregistered.py reports/preregistration_smoke.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_power_study as ps  # noqa: E402  (shared arm builders and constants)

from genomefolio.backtest import paired_block_bootstrap, performance_metrics, seed_for  # noqa: E402
from genomefolio.config import load_config  # noqa: E402
from genomefolio.data import align_to_common_days, get_prices, load_fixture, to_returns  # noqa: E402
from genomefolio.reporting import save_series, write_run_meta  # noqa: E402

REPORTS = Path("reports")


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    base = load_config()
    us = spec["universe_spec"]
    uni = replace(
        base.universe,
        tickers=tuple(us["tickers"]),
        start=us["start"],
        end=us["end"],
        cache_dir=us["cache_dir"],
    )
    cfg = replace(base, universe=uni)
    prices = load_fixture(cfg) if us.get("fixture") else get_prices(cfg)
    returns = align_to_common_days(to_returns(prices))

    daily = {}
    arm_params = {"per_stock": (False, "per_stock"), "pooled_mnl": (True, "pooled")}
    for arm in spec["arms"]:
        mn, scope = arm_params[arm]
        daily[arm] = ps._alignment_arm(returns, cfg, mn, scope)
    daily["equal_weight"] = ps._equal_arm(returns, cfg)

    primary, reference = spec["primary"], spec["reference"]
    ci = paired_block_bootstrap(
        daily[primary], daily[reference],
        base.bootstrap.n_resamples, base.bootstrap.block_length,
        base.bootstrap.confidence, np.random.default_rng(seed_for(spec["name"], base.seed)),
        base.bootstrap.use_politis_white,
    )
    lo, hi = ci["sharpe_diff"]
    if spec["comparison"] == "higher_sharpe":
        if lo > 0:
            verdict = "SUPPORTED"
        elif hi < 0:
            verdict = "REFUTED (significantly worse)"
        else:
            verdict = "NOT SUPPORTED (CI includes 0)"
    else:
        verdict = "UNKNOWN COMPARISON"

    result = {
        "spec": spec,
        "metrics": {name: performance_metrics(s) for name, s in daily.items()},
        "primary_ci_vs_reference": ci,
        "verdict": verdict,
    }
    name = spec["name"]
    (REPORTS / f"preregistration_{name}_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    save_series(daily, REPORTS / f"preregistration_{name}_daily.csv")
    write_run_meta(REPORTS / f"preregistration_{name}_meta.json", cfg)
    print(f"{primary} vs {reference}: sharpe_diff={ci['sharpe_diff']} -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1])
