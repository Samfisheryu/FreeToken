"""Backward-compatible DeepSeek-V4 entry point for the common NoWAG reader."""

from __future__ import annotations

from pathlib import Path

import torch

from freetoken.moe.nowag import load_nowag_expert_sources as _load_common


def load_nowag_expert_sources(
    output_path: str | Path,
    model_config,
    *,
    dtype: torch.dtype = torch.bfloat16,
):
    return _load_common(
        output_path,
        model_config,
        dtype=dtype,
        model_type="deepseek_v4",
    )


__all__ = ["load_nowag_expert_sources"]
