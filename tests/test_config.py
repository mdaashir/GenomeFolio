import pytest
import yaml

from src.config import Config, load_config


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert len(cfg.symbolize.cutoffs) == 4
    assert cfg.cluster.k_min >= 2
    assert cfg.window.train_days > 0
    assert cfg.window.forecast_horizon > 0


def test_k_cap_formula():
    cfg = load_config()
    # k <= min(k_max, floor(N / divisor))
    assert cfg.cluster.k_cap(20) == 6
    assert cfg.cluster.k_cap(10) == 3
    assert cfg.cluster.k_cap(8) == 2


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
        lambda d: d["window"].__setitem__("forecast_horizon", 0),
        lambda d: d["forecast"].__setitem__("model", "bogus"),
        lambda d: d["align"].__setitem__("match_score", 0.0),
        lambda d: d["cluster"].__setitem__("linkage", "bogus"),
        lambda d: d["universe"].__setitem__("tickers", ["AAPL", "AAPL"]),
        lambda d: d["universe"].__setitem__("start", "2025-01-01"),
    ],
)
def test_invalid_config_values_rejected(tmp_path, mutate):
    data = yaml.safe_load(open("config/params.yaml", encoding="utf-8"))
    mutate(data)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
