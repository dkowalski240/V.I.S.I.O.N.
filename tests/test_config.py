from pathlib import Path

import pytest

from assistant.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_yaml_matches_defaults():
    assert load_config(ROOT / "config.yaml") == Config()


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("faces:\n  matching:\n    t_mach: 0.7\n")
    with pytest.raises(ValueError, match="t_mach"):
        load_config(p)


def test_thresholds_must_be_ordered(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("faces:\n  matching:\n    t_sure: 0.5\n")
    with pytest.raises(ValueError, match="t_sure"):
        load_config(p)


def test_partial_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("direction:\n  mode: angle\n  hfov_deg: 115\ncamera:\n  source: http://phone:8080/video\n")
    cfg = load_config(p)
    assert cfg.direction.mode == "angle" and cfg.direction.zones == (10, 11, 12, 1, 2)
    assert cfg.camera.source == "http://phone:8080/video"
