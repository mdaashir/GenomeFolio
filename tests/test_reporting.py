import json

import pandas as pd

from genomefolio.config import load_config
from genomefolio.reporting import config_hash, save_series, write_run_meta


def test_run_meta_written(tmp_path):
    cfg = load_config()
    path = write_run_meta(tmp_path / "meta.json", cfg, {"note": "x"})
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["config_hash"] == config_hash(cfg)
    assert meta["seed"] == cfg.seed
    assert "git_commit" in meta
    assert "uv_lock_hash" in meta
    assert "numpy" in meta["packages"]
    assert meta["note"] == "x"


def test_config_hash_is_stable_and_config_sensitive():
    cfg = load_config()
    assert config_hash(cfg) == config_hash(load_config())
    other = load_config()
    object.__setattr__(other, "seed", 999)  # frozen dataclass; bypass for the test
    assert config_hash(other) != config_hash(cfg)


def test_save_series_round_trip(tmp_path):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    series = {"a": pd.Series(range(5), index=idx, dtype=float), "b": pd.Series(1.0, index=idx)}
    path = save_series(series, tmp_path / "daily.csv")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    assert list(frame.columns) == ["a", "b"]
    assert len(frame) == 5
