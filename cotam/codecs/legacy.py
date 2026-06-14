from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def train_root() -> Path:
    return Path(__file__).resolve().parents[3]


def add_legacy_paths(kind: str = "examples") -> None:
    root = train_root()
    paths = [root]
    if kind == "base":
        paths.insert(0, root / "base_codec")
    else:
        paths.insert(0, root / "examples")
    for path in paths:
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def import_legacy(module_name: str, kind: str = "examples") -> ModuleType:
    add_legacy_paths(kind=kind)
    return importlib.import_module(module_name)
