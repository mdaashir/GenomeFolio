import pytest
import yaml

from src.config import Config, load_config, load_sectors


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
