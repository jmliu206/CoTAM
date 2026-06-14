"""CLIP wrappers used by CoTAM losses."""

from .wrapper import FakeClipWrapper, build_clip_wrapper

__all__ = ["FakeClipWrapper", "build_clip_wrapper"]
