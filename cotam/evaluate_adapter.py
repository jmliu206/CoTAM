from __future__ import annotations

from typing import Optional

from cotam.train_adapter import main as train_main


def main(argv: Optional[list[str]] = None):
    argv = list(argv or [])
    if "--eval-only" not in argv:
        argv.append("--eval-only")
    return train_main(argv)


if __name__ == "__main__":
    main()
