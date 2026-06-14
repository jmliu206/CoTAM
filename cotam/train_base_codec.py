from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader

from cotam.codecs import RateDistortionLoss, build_base_codec
from cotam.datasets import FakeCaptionDataset
from cotam.utils.config import deep_get, load_config
from cotam.utils.seed import set_seed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean base codec training entrypoint.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--use-fake-data", action="store_true")
    parser.add_argument("--use-tiny-codec", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> dict:
    args = parse_args(argv)
    config = load_config(args.config, args.override)
    set_seed(deep_get(config, "experiment.seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codec_cfg = dict(config.get("codec", {}))
    if args.use_tiny_codec:
        codec_cfg["use_tiny"] = True
        codec_cfg["name"] = "tiny"
    model = build_base_codec(codec_cfg).to(device)
    criterion = RateDistortionLoss(lmbda=float(deep_get(config, "codec.lmbda", 1e-2)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(deep_get(config, "codec.learning_rate", 1e-4)))

    if not args.use_fake_data:
        raise NotImplementedError("Real base-codec data loading will be wired after adapter clean path.")
    dataset = FakeCaptionDataset(length=4, image_size=int(deep_get(config, "data.image_size", 256)))
    loader = DataLoader(dataset, batch_size=int(deep_get(config, "data.batch_size", 1)))

    last_loss = 0.0
    for step, (images, _) in enumerate(loader, start=1):
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        loss = criterion(out, images)["loss"]
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step >= args.max_steps:
            break
    return {"loss": last_loss}


if __name__ == "__main__":
    main()
