"""Console entry point (``genomefolio``).

Runs the shortened pipeline: data -> symbolize -> align -> cluster -> forecast. There is
no null gate and no portfolio/backtest downstream.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from src.config import load_config
from src.data import align_to_common_days, get_prices, load_fixture, to_returns
from src.pipeline import run_pipeline

logger = logging.getLogger("genomefolio")


@click.command()
@click.option("--config", "config_path", default=None, help="Path to params.yaml")
@click.option("--use-fixture", is_flag=True, help="Use the committed offline sample")
@click.option("--out", "out_path", default=None, help="Write clusters + forecasts to this JSON file")
@click.option("--verbose", is_flag=True, help="Debug-level logging")
def cli(config_path, use_fixture, out_path, verbose):
    """Cluster stocks by return pattern and forecast each cluster."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        force=True,
    )
    cfg = load_config(config_path)
    prices = load_fixture(cfg) if use_fixture else get_prices(cfg)
    returns = align_to_common_days(to_returns(prices))
    if len(returns) < cfg.window.train_days:
        raise click.ClickException(
            f"need at least {cfg.window.train_days} days of returns, got {len(returns)}"
        )

    window = returns.iloc[-cfg.window.train_days :]
    result = run_pipeline(window, cfg)

    logger.info(
        "Universe: %d stocks | window=%d days | clusters=%d | silhouette=%.3f",
        window.shape[1],
        len(window),
        result.n_clusters,
        result.silhouette,
    )
    if not result.min_size_ok:
        logger.info("Note: no k satisfied the minimum cluster size; using the best-silhouette cut.")

    for cid in sorted(result.clusters):
        members = result.clusters[cid]
        forecast = result.forecasts[cid]
        logger.info(
            "\nCluster %d (%d stocks) forecast=%+.4f over %d days [%s]",
            cid,
            len(members),
            forecast.expected_return,
            cfg.window.forecast_horizon,
            forecast.model,
        )
        logger.info("  %s", ", ".join(members))

    if out_path:
        payload = {
            "n_clusters": result.n_clusters,
            "silhouette": result.silhouette,
            "min_size_ok": result.min_size_ok,
            "forecast_horizon": cfg.window.forecast_horizon,
            "clusters": {str(cid): list(members) for cid, members in result.clusters.items()},
            "forecasts": {
                str(cid): {
                    "expected_return": fc.expected_return,
                    "model": fc.model,
                    "order": list(fc.order) if fc.order else None,
                    "fallback": fc.fallback,
                }
                for cid, fc in result.forecasts.items()
            },
        }
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("\nWrote %s", path)


if __name__ == "__main__":
    cli()
