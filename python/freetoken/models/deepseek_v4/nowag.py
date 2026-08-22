"""Read the expert-only NoWAG output produced for DeepSeek-V4."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


_FORMAT = "deepseek_v4_nowag_expert_sidecar_v1"
_CODEBOOK_KEY = "global_all.codebook"
_PROJECTION_BANK = {"w1": "gate", "w3": "up", "w2": "down"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _words(width: int, group_size: int, assignment_bits: int) -> int:
    groups = math.ceil(width / group_size)
    return math.ceil(groups * assignment_bits / 32)


def _tensor_key(layer: int, expert: int, projection: str, suffix: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"


def load_nowag_expert_sources(
    output_path: str | Path,
    model_config,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, list[torch.Tensor]], torch.Tensor]:
    """Load and pin the nine per-expert banks plus the one shared codebook."""
    if dtype != torch.bfloat16:
        raise ValueError("DeepSeek-V4 NoWAG serving currently requires bfloat16")

    root = Path(output_path).resolve()
    manifest_path = root / "manifest.json" if root.is_dir() else root
    root = manifest_path.parent
    manifest = _read_json(manifest_path)
    if manifest.get("format") != _FORMAT:
        raise ValueError(f"{manifest_path}: unsupported NoWAG output format")
    if manifest.get("scope") != "expert_only" or manifest.get("codebook_sharing") != "global_all":
        raise ValueError("DeepSeek-V4 runtime requires expert-only global_all quantization")
    if int(manifest.get("d", 0)) != 6 or int(manifest.get("assignment_bits", 0)) != 12:
        raise ValueError("DeepSeek-V4 runtime currently supports only NoWAG D6/B12")
    if manifest.get("assignments_packed") is not True:
        raise ValueError("DeepSeek-V4 runtime requires packed assignments")
    layers = int(model_config.num_moe_layers)
    experts = int(model_config.num_experts)
    hidden = int(model_config.hidden_size)
    intermediate = int(model_config.moe_intermediate_size)
    if int(manifest.get("matrix_count", -1)) != layers * experts * 3:
        raise ValueError("NoWAG output does not cover every routed expert matrix")

    layer_entries = manifest.get("layers")
    if not isinstance(layer_entries, list) or len(layer_entries) != layers:
        raise ValueError("NoWAG output has the wrong number of expert layers")
    files_by_layer: dict[int, Path] = {}
    for entry in layer_entries:
        if not isinstance(entry, dict):
            raise TypeError("NoWAG layer entry must be a JSON object")
        layer = int(entry["layer"])
        file = entry.get("file")
        if not isinstance(file, str):
            raise TypeError("NoWAG layer file must be a string")
        files_by_layer[layer] = root / file
    if set(files_by_layer) != set(range(layers)):
        raise ValueError("NoWAG layer indices are incomplete")

    gate_words = _words(hidden, 6, 12)
    down_words = _words(intermediate, 6, 12)
    specs = {
        "gate_assignments": ((experts, intermediate, gate_words), torch.int32),
        "gate_input_norm": ((experts, hidden), dtype),
        "gate_output_norm": ((experts, intermediate), dtype),
        "up_assignments": ((experts, intermediate, gate_words), torch.int32),
        "up_input_norm": ((experts, hidden), dtype),
        "up_output_norm": ((experts, intermediate), dtype),
        "down_assignments": ((experts, hidden, down_words), torch.int32),
        "down_input_norm": ((experts, intermediate), dtype),
        "down_output_norm": ((experts, hidden), dtype),
    }

    from freetoken.moe.host_banks import PinPipeline, alloc_layer_banks

    host_banks = alloc_layer_banks(specs, layers)
    sources = {
        name: [bank.tensor for bank in per_layer]
        for name, per_layer in host_banks.items()
    }
    with PinPipeline() as pins:
        for layer in range(layers):
            path = files_by_layer[layer]
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for expert in range(experts):
                    for projection, bank in _PROJECTION_BANK.items():
                        keys = {
                            "assignments": _tensor_key(
                                layer, expert, projection, "assignments"
                            ),
                            "input_norm": _tensor_key(
                                layer, expert, projection, "normalizer.norms.0"
                            ),
                            "output_norm": _tensor_key(
                                layer, expert, projection, "normalizer.norms.1"
                            ),
                        }
                        missing = [name for name in keys.values() if name not in available]
                        if missing:
                            raise KeyError(f"{path}: missing {missing[0]}")
                        for kind, key in keys.items():
                            target = sources[f"{bank}_{kind}"][layer][expert]
                            loaded = handle.get_tensor(key)
                            if tuple(loaded.shape) != tuple(target.shape):
                                raise ValueError(
                                    f"{key}: expected {tuple(target.shape)}, got {tuple(loaded.shape)}"
                                )
                            if kind == "assignments" and loaded.dtype != torch.int32:
                                raise TypeError(f"{key}: assignments must use int32")
                            if kind != "assignments" and loaded.dtype != dtype:
                                raise TypeError(
                                    f"{key}: normalizer must use {dtype}, got {loaded.dtype}"
                                )
                            target.copy_(loaded)
            pins(layer, {name: per[layer] for name, per in host_banks.items()})

    codebook_entry = manifest.get("codebook")
    if not isinstance(codebook_entry, dict):
        raise ValueError("NoWAG output is missing the global codebook")
    codebook_file = codebook_entry.get("file")
    codebook_name = codebook_entry.get("tensor")
    if not isinstance(codebook_file, str) or not isinstance(codebook_name, str):
        raise TypeError("NoWAG codebook file and tensor must be strings")
    if codebook_entry.get("dtype") not in (None, "bfloat16"):
        raise TypeError("DeepSeek-V4 NoWAG serving requires a bfloat16 codebook")
    with safe_open(root / codebook_file, framework="pt", device="cpu") as handle:
        if codebook_name != _CODEBOOK_KEY or codebook_name not in handle.keys():
            raise KeyError("NoWAG output is missing global_all.codebook")
        codebook = handle.get_tensor(codebook_name)
    if codebook.dtype != dtype:
        raise TypeError(f"NoWAG codebook must use {dtype}, got {codebook.dtype}")
    codebook = codebook.contiguous()
    if tuple(codebook.shape) != (4096, 6):
        raise ValueError(f"NoWAG codebook must be [4096, 6], got {tuple(codebook.shape)}")
    return sources, codebook


__all__ = ["load_nowag_expert_sources"]
