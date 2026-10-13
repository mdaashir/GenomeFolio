"""Console entry point (``genomefolio``).

Thin wrapper over :func:`genomefolio.backtest.main`.
"""

from __future__ import annotations

import logging
import sys

import click

from genomefolio.backtest import main
from genomefolio.config import load_config

_METHODS = "alignment,equal_weight,min_variance,mean_variance,correlation,random,sector"


@click.command()
@click.option("--config", "config_path", default=None, help="Path to params.yaml")
@click.option("--use-fixture", is_flag=True, help="Use the committed offline sample")
@click.option("--draws", "n_draws", default=None, type=int, help="Null-gate draws")
@click.option("--cost-bps", default=None, type=float, help="Override transaction cost (bps)")
@click.option("--methods", default=_METHODS, help="Comma-separated methods to compare")
@click.option("--band-only", is_flag=True, help="Only print the band-sweep stability")
@click.option("--verbose", is_flag=True, help="Debug-level logging")
def cli(config_path, use_fixture, n_draws, cost_bps, methods, band_only, verbose):
    """Run the GenomeFolio walk-forward backtest."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        force=True,
    )
    cfg = load_config(config_path)
    main(
        config_path,
        use_fixture,
        n_draws if n_draws is not None else cfg.null_test.n_draws_report,
        cost_bps,
        tuple(m.strip() for m in methods.split(",") if m.strip()),
        band_only,
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
