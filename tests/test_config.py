"""Tests for YAML configuration loading."""

from pathlib import Path

import yaml

from backend.config import AppConfig, load_config


def test_load_config() -> None:
    """Verify load_config parses all sections from config.yaml."""
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    cfg = load_config()

    assert isinstance(cfg, AppConfig)
    assert cfg.models.model_dump() == raw["models"]
    assert cfg.audio.model_dump() == raw["audio"]
    assert cfg.thresholds.model_dump() == raw["thresholds"]
    assert {key: str(val) for key, val in cfg.paths.model_dump().items()} == raw["paths"]

