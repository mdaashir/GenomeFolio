import pytest
import yaml

from genomefolio.config import Config, load_config, load_sectors, validate_sector_coverage


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert len(cfg.symbolize.cutoffs) == 4
    assert cfg.cluster.k_min >= 2
    assert cfg.window.train_days > 0


def test_k_cap_formula():
    cfg = load_config()
    # k <= min(k_max, floor(N / divisor))
    assert cfg.cluster.k_cap(20) == 6
    assert cfg.cluster.k_cap(10) == 3
    assert cfg.cluster.k_cap(8) == 2


def test_cap_formula():
    cfg = load_config()
    # cap = max(cap_floor, cap_factor / N)
    assert cfg.portfolio.cap(20) == pytest.approx(0.10)
    assert cfg.portfolio.cap(8) == pytest.approx(0.1875)


def test_portfolio_optimiser_params_present_and_valid():
    p = load_config().portfolio
    assert p.mv_risk_aversion >= 0
    assert 0.0 <= p.mean_shrink < 1.0


def test_sector_groups_obey_size_rules():
    sectors = load_sectors()
    assert 3 <= len(sectors) <= 6
    for members in sectors.values():
        assert len(members) >= 3


def test_sectors_cover_universe():
    cfg = load_config()
    sectors = load_sectors()
    covered = {t for members in sectors.values() for t in members}
    assert set(cfg.universe.tickers) == covered


def test_sector_coverage_validation_passes_and_fails():
    sectors = load_sectors()
    validate_sector_coverage(sectors, load_config().universe.tickers)  # no raise
    with pytest.raises(ValueError, match="does not cover"):
        validate_sector_coverage(sectors, ["AAPL", "ZZZZ"])


def test_invalid_cutoffs_rejected(tmp_path):
    data = yaml.safe_load(open("config/params.yaml", encoding="utf-8"))
    data["symbolize"]["cutoffs"] = [0.5, 0.4, 0.6, 0.8]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_section_rejected(tmp_path):
    data = yaml.safe_load(open("config/params.yaml", encoding="utf-8"))
    del data["cluster"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(path)


def test_bad_split_fraction_rejected(tmp_path):
    data = yaml.safe_load(open("config/params.yaml", encoding="utf-8"))
    data["split"]["tune_fraction"] = 1.5
    path = tmp_path / "badsplit.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_quantile_scope_default_and_validation(tmp_path):
    cfg = load_config()
    assert cfg.symbolize.quantile_scope in {"per_stock", "pooled"}
    data = yaml.safe_load(open("config/params.yaml", encoding="utf-8"))
    data["symbolize"]["quantile_scope"] = "bogus"
    path = tmp_path / "badscope.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["window"].__setitem__("train_days", 0),
        lambda d: d["window"].__setitem__("rebalance_days", -1),
        lambda d: d["window"].__setitem__("forecast_horizon", 0),
        lambda d: d["align"].__setitem__("match_score", 0.0),
        lambda d: d["align"].__setitem__("band_sweep", [2, -1]),
        lambda d: d["null_test"].__setitem__("n_draws_report", 0),
        lambda d: d["bootstrap"].__setitem__("n_resamples", 0),
        lambda d: d["bootstrap"].__setitem__("block_length", 0),
        lambda d: d["costs"].__setitem__("sweep_bps", []),
        lambda d: d["costs"].__setitem__("base_bps", -1.0),
        lambda d: d["portfolio"].__setitem__("cap_floor", 0.0),
        lambda d: d["portfolio"].__setitem__("cap_factor", 0.0),
        lambda d: d["portfolio"].__setitem__("forecast_tilt_strength", -1.0),
        lambda d: d["portfolio"].__setitem__("mv_risk_aversion", -1.0),
        lambda d: d["portfolio"].__setitem__("mean_shrink", 1.5),
        lambda d: d["forecast"].__setitem__("model", "bogus"),
        lambda d: d["cluster"].__setitem__("linkage", "bogus"),
        lambda d: d["universe"].__setitem__("tickers", ["AAPL", "AAPL"]),
        lambda d: d["universe"].__setitem__("start", "2025-01-01"),
        lambda d: d["universe"].__setitem__("start", "not-a-date"),
    ],
)
def test_invalid_config_values_rejected(tmp_path, mutate):
    data = yaml.safe_load(open("config/params.yaml", encoding="utf-8"))
    mutate(data)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
