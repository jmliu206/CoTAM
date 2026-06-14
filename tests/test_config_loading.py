from __future__ import annotations

from pathlib import Path

from cotam.utils.config import deep_get, load_config


def test_config_loading_and_override() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "cotam_elic_clip_adapter.yaml"
    config = load_config(config_path, ["loss.total_weight=2.5", "mask.provider=random"])
    assert deep_get(config, "loss.total_weight") == 2.5
    assert deep_get(config, "mask.provider") == "random"
    assert deep_get(config, "adapter.in_features") == 320


def test_clean_configs_do_not_contain_sensitive_literals() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        "HUGGING" + "_FACE_HUB_TOKEN",
        "WANDB" + "_API_KEY",
        "h" + "f_",
        "/data" + "/data",
        "/data" + "/code",
    ]
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".sh", ".txt"}:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"{token} found in {path}"
