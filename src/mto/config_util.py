"""Configuration loading with CLI override support.

Config precedence: CLI args > YAML config > Python defaults.
"""
import os
import yaml
import argparse


def load_yaml_config(path):
    """Load a YAML configuration file. Returns empty dict if file missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base, override):
    """Merge override dict into base dict recursively. Override wins."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def apply_cli_overrides(config, args, cli_map):
    """Apply CLI args to config dict using a mapping of cli_attr -> config_path.

    Args:
        config: dict to update in-place
        args: argparse.Namespace
        cli_map: dict[str, str or list[str]] mapping arg_name to "section.key"
                 e.g. {"feature_dim": "model.feature_dim", "lr": "train.lr"}

    Only applies override if the CLI arg differs from its default.
    """
    for arg_name, config_path in cli_map.items():
        val = getattr(args, arg_name, None)
        if val is None:
            continue
        keys = config_path.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = val


def set_nested(config, path, value):
    """Set a nested key 'a.b.c' in config dict to value."""
    keys = path.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def get_nested(config, path, default=None):
    """Get a nested key 'a.b.c' from config dict."""
    keys = path.split(".")
    d = config
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d
