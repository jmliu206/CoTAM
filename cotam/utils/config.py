from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Union

import yaml


class ConfigError(ValueError):
    """Raised when a config file or override is malformed."""


def load_config(
    path: Optional[Union[str, Path]],
    overrides: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Load a YAML config and apply dotted ``key=value`` overrides."""

    data: dict[str, Any] = {}
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config must be a mapping: {path}")
        data = loaded

    data = copy.deepcopy(data)
    for item in overrides or []:
        apply_override(data, item)
    return data


def apply_override(config: MutableMapping[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"Override must be key=value: {override}")
    key, raw_value = override.split("=", 1)
    if not key:
        raise ConfigError(f"Override key is empty: {override}")
    value = yaml.safe_load(raw_value)
    cursor: MutableMapping[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, MutableMapping):
            raise ConfigError(f"Cannot override through non-mapping key: {part}")
        cursor = next_value
    cursor[parts[-1]] = value


def deep_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    value = deep_get(config, dotted_key)
    if value is None:
        raise ConfigError(f"Missing required config key: {dotted_key}")
    return value
