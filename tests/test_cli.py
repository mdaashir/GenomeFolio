"""CLI smoke test."""

from click.testing import CliRunner

from genomefolio.cli import cli


def test_cli_band_only_on_fixture():
    result = CliRunner().invoke(cli, ["--use-fixture", "--draws", "5", "--band-only"])
    assert result.exit_code == 0
    assert "Band-sweep stability" in result.output
    assert "w= 5" in result.output


def test_cli_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "walk-forward backtest" in result.output
