"""Test configuration loading and CLI override."""
import os
import sys
import json
import tempfile

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)


def test_load_yaml_missing():
    from src.mto.config_util import load_yaml_config
    cfg = load_yaml_config("/nonexistent/path.yaml")
    assert cfg == {}


def test_merge_configs():
    from src.mto.config_util import merge_configs
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 100}
    merged = merge_configs(base, override)
    assert merged["a"] == 1
    assert merged["b"]["c"] == 99  # overridden
    assert merged["b"]["d"] == 3   # preserved
    assert merged["e"] == 100


def test_merge_configs_empty_override():
    from src.mto.config_util import merge_configs
    base = {"a": 1, "b": 2}
    merged = merge_configs(base, {})
    assert merged == base


def test_get_nested():
    from src.mto.config_util import get_nested
    cfg = {"model": {"feature_dim": 128, "mto": {"hidden_dim": 64}}}
    assert get_nested(cfg, "model.feature_dim") == 128
    assert get_nested(cfg, "model.mto.hidden_dim") == 64
    assert get_nested(cfg, "nonexistent.key", 42) == 42


def test_set_nested():
    from src.mto.config_util import set_nested, get_nested
    cfg = {}
    set_nested(cfg, "model.feature_dim", 128)
    assert cfg["model"]["feature_dim"] == 128
    set_nested(cfg, "train.lr", 1e-3)
    assert cfg["train"]["lr"] == 1e-3
