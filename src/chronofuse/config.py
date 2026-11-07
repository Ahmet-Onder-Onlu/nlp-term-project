"""Experiment configuration.

Configs are YAML files. An experiment file may declare ``inherit: <path>`` to
extend a base config; nested mappings are merged recursively, so an experiment
only states what it changes.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


class Config(dict):
    """A dict with attribute access and dotted ``get``: ``cfg.get("bm25.k1")``."""

    def __getattr__(self, item: str) -> Any:
        try:
            val = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("inherit", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        raw = _deep_merge(dict(load_config(parent_path)), raw)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
