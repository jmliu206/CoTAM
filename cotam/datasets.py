from __future__ import annotations

import glob
import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HashTextTokenizer:
    """Small deterministic tokenizer for tests and fake-data smoke runs."""

    def __init__(self, context_length: int = 77) -> None:
        self.context_length = context_length

    def tokenize(self, text: str) -> torch.Tensor:
        result = torch.zeros(self.context_length, dtype=torch.long)
        tokens = [1]
        tokens.extend((sum(bytearray(word.encode("utf-8"))) % 49400) + 2 for word in text.split())
        tokens.append(2)
        tokens = tokens[: self.context_length]
        result[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return result


def load_text_tokenizer(context_length: int = 77, prefer_legacy: bool = True) -> Any:
    if prefer_legacy:
        try:
            from simple_tokenizer import SimpleTokenizer  # type: ignore

            legacy = SimpleTokenizer()

            class LegacyAdapter:
                def tokenize(self, text: str) -> torch.Tensor:
                    sot = legacy.encoder["<|startoftext|>"]
                    eot = legacy.encoder["<|endoftext|>"]
                    tokens = [sot] + legacy.encode(text) + [eot]
                    tokens = tokens[:context_length]
                    result = torch.zeros(context_length, dtype=torch.long)
                    result[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                    return result

            return LegacyAdapter()
        except Exception:
            pass
    return HashTextTokenizer(context_length=context_length)


def build_image_transform(image_size: int = 384) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def _read_json(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_hook=OrderedDict)


class CocoCaptionDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        tokenizer: Any,
        split: str = "val2017",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.tokenizer = tokenizer
        annotations = _read_json(self.root / "annotations" / f"captions_{split}.json")
        self.img_dir = self.root / split

        img_id_to_filename = {item["id"]: item["file_name"] for item in annotations["images"]}
        img_id_to_captions: dict[int, list[str]] = {}
        for item in annotations["annotations"]:
            img_id_to_captions.setdefault(item["image_id"], []).append(item["caption"])
        self.items = [
            (img_id_to_filename[img_id], captions)
            for img_id, captions in img_id_to_captions.items()
            if img_id in img_id_to_filename
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, captions = self.items[index]
        image = Image.open(self.img_dir / filename).convert("RGB")
        tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        text = random.choice(captions)
        return tensor, self.tokenizer.tokenize(text)


class CC3MDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        tokenizer: Any,
        split: str = "train",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.img_dir = self.root / split
        self.transform = transform
        self.tokenizer = tokenizer
        pattern_root = str(self.img_dir)
        self.images = sorted(glob.glob(os.path.join(pattern_root, "*.jpg")))
        self.images.extend(sorted(glob.glob(os.path.join(pattern_root, "*", "*.jpg"))))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.images[index])
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        text_path = path.with_suffix(".txt")
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else "None"
        return tensor, self.tokenizer.tokenize(text)


class FakeCaptionDataset(Dataset):
    def __init__(self, length: int = 8, image_size: int = 384, context_length: int = 77) -> None:
        self.length = length
        self.image_size = image_size
        self.context_length = context_length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        tokens = torch.zeros(self.context_length, dtype=torch.long)
        tokens[0] = 1
        tokens[1] = 10 + index
        tokens[2] = 2
        return image, tokens


def collate_image_text(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in batch], dim=0)
    texts = torch.stack([item[1] for item in batch], dim=0)
    return images, texts
