"""Load expert-only NoWAG weights and describe their model-specific math."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


FORMAT = "nowag_expert_sidecar_v1"
LEGACY_DSV4_FORMAT = "deepseek_v4_nowag_expert_sidecar_v1"
CODEBOOK_KEY = "global_all.codebook"
RUNTIME_ASSIGNMENT_LAYOUT = "word_major"
NO_ACTIVATION_ROUNDING = "none"
DYNAMIC_E4M3_GROUP128_UE8M0 = "dynamic_e4m3_per_token_group128_ue8m0"
GATE_UP_EPILOGUE_NORM = "gate_up_epilogue"
DOWN_PROLOGUE_NORM = "down_prologue"

# The files keep the projection names used by the first DSV4 quantizer.  Their
# meaning is model-independent: w1 is gate, w3 is up, and w2 is down.
_PROJECTION_BANK = {"w1": "gate", "w3": "up", "w2": "down"}


@dataclass(frozen=True)
class NowagModelRule:
    model_type: str
    gate_up_input_rounding: str
    down_input_rounding: str
    down_norm_placement: str
    requires_swiglu_limit: bool


_MODEL_RULES = {
    "deepseek_v4": NowagModelRule(
        "deepseek_v4",
        gate_up_input_rounding=DYNAMIC_E4M3_GROUP128_UE8M0,
        down_input_rounding=DYNAMIC_E4M3_GROUP128_UE8M0,
        down_norm_placement=DOWN_PROLOGUE_NORM,
        requires_swiglu_limit=True,
    ),
    "qwen3_5_moe": NowagModelRule(
        "qwen3_5_moe",
        gate_up_input_rounding=NO_ACTIVATION_ROUNDING,
        down_input_rounding=NO_ACTIVATION_ROUNDING,
        down_norm_placement=GATE_UP_EPILOGUE_NORM,
        requires_swiglu_limit=False,
    ),
}


def get_nowag_model_rule(model_config_or_type) -> NowagModelRule:
    """Return the supported NoWAG rule for a parsed model config or model type."""
    if isinstance(model_config_or_type, str):
        model_type = model_config_or_type
        model_config = None
    else:
        model_config = model_config_or_type
        model_type = getattr(model_config, "model_type", None)
        if getattr(model_config, "dsv4_args", None) is not None:
            model_type = "deepseek_v4"

    rule = _MODEL_RULES.get(model_type)
    if rule is None:
        supported = ", ".join(sorted(_MODEL_RULES))
        raise ValueError(
            f"NoWAG expert serving does not support model_type {model_type!r}; "
            f"supported model types: {supported}"
        )
    if model_config is not None:
        if not getattr(model_config, "is_moe", True):
            raise ValueError("--nowag-expert-path requires a model with routed experts")
        hidden_act = getattr(model_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"NoWAG expert serving requires SwiGLU/silu experts, got {hidden_act!r}"
            )
    return rule


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


def _validate_manifest_model(
    manifest: dict[str, Any],
    manifest_path: Path,
    rule: NowagModelRule,
    *,
    layers: int,
    experts: int,
    hidden: int,
    intermediate: int,
) -> None:
    output_format = manifest.get("format")
    if output_format == LEGACY_DSV4_FORMAT:
        if rule.model_type != "deepseek_v4":
            raise ValueError(
                f"{manifest_path}: legacy DeepSeek-V4 NoWAG weights cannot serve "
                f"{rule.model_type}"
            )
        return
    if output_format != FORMAT:
        raise ValueError(f"{manifest_path}: unsupported NoWAG output format")
    if manifest.get("model_type") != rule.model_type:
        raise ValueError(
            f"{manifest_path}: NoWAG weights are for model_type "
            f"{manifest.get('model_type')!r}, expected {rule.model_type!r}"
        )
    expected_dims = {
        "num_moe_layers": layers,
        "num_experts": experts,
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
    }
    for name, expected in expected_dims.items():
        if int(manifest.get(name, -1)) != expected:
            raise ValueError(
                f"{manifest_path}: {name} must be {expected}, got {manifest.get(name)!r}"
            )


def load_nowag_expert_sources(
    output_path: str | Path,
    model_config,
    *,
    dtype: torch.dtype = torch.bfloat16,
    model_type: str | None = None,
) -> tuple[dict[str, list[torch.Tensor]], torch.Tensor]:
    """Load and pin the nine per-expert banks plus one model-wide codebook.

    Assignment banks are always returned as contiguous ``[E, W, N]`` tensors.
    Existing sidecars store each expert as ``[N, W]`` and are transposed while
    copying into the final pinned bank.  A sidecar that declares
    ``assignment_layout="word_major"`` is copied directly, so a quantizer can
    write the runtime layout and avoid the transpose altogether.
    """
    if dtype != torch.bfloat16:
        raise ValueError("NoWAG expert serving currently requires bfloat16")

    rule = get_nowag_model_rule(model_type or model_config)
    root = Path(output_path).resolve()
    manifest_path = root / "manifest.json" if root.is_dir() else root
    root = manifest_path.parent
    manifest = _read_json(manifest_path)

    layers = int(model_config.num_moe_layers)
    experts = int(model_config.num_experts)
    hidden = int(model_config.hidden_size)
    intermediate = int(model_config.moe_intermediate_size)
    _validate_manifest_model(
        manifest,
        manifest_path,
        rule,
        layers=layers,
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    if manifest.get("scope") != "expert_only" or manifest.get("codebook_sharing") != "global_all":
        raise ValueError("NoWAG runtime requires expert-only quantization with one global codebook")
    group_size = int(manifest.get("d", 0))
    assignment_bits = int(manifest.get("assignment_bits", 0))
    if group_size != 6 or assignment_bits != 12:
        raise ValueError("NoWAG runtime currently supports only D6/B12")
    if manifest.get("assignments_packed") is not True:
        raise ValueError("NoWAG runtime requires packed assignments")
    source_assignment_layout = manifest.get("assignment_layout", "row_major")
    if source_assignment_layout not in ("row_major", RUNTIME_ASSIGNMENT_LAYOUT):
        raise ValueError(
            "NoWAG assignment_layout must be 'row_major' or 'word_major'"
        )
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

    gate_words = _words(hidden, group_size, assignment_bits)
    down_words = _words(intermediate, group_size, assignment_bits)
    specs = {
        "gate_assignments": ((experts, gate_words, intermediate), torch.int32),
        "gate_input_norm": ((experts, hidden), dtype),
        "gate_output_norm": ((experts, intermediate), dtype),
        "up_assignments": ((experts, gate_words, intermediate), torch.int32),
        "up_input_norm": ((experts, hidden), dtype),
        "up_output_norm": ((experts, intermediate), dtype),
        "down_assignments": ((experts, down_words, hidden), torch.int32),
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
                            expected_shape = tuple(target.shape)
                            if kind == "assignments" and source_assignment_layout == "row_major":
                                expected_shape = (target.shape[1], target.shape[0])
                            if tuple(loaded.shape) != expected_shape:
                                raise ValueError(
                                    f"{key}: expected {expected_shape}, "
                                    f"got {tuple(loaded.shape)}"
                                )
                            if kind == "assignments" and loaded.dtype != torch.int32:
                                raise TypeError(f"{key}: assignments must use int32")
                            if kind != "assignments" and loaded.dtype != dtype:
                                raise TypeError(
                                    f"{key}: normalizer must use {dtype}, got {loaded.dtype}"
                                )
                            if kind == "assignments" and source_assignment_layout == "row_major":
                                target.copy_(loaded.transpose(0, 1))
                            else:
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
        raise TypeError("NoWAG expert serving requires a bfloat16 codebook")
    with safe_open(root / codebook_file, framework="pt", device="cpu") as handle:
        if codebook_name != CODEBOOK_KEY or codebook_name not in handle.keys():
            raise KeyError(f"NoWAG output is missing {CODEBOOK_KEY}")
        codebook = handle.get_tensor(codebook_name)
    if codebook.dtype != dtype:
        raise TypeError(f"NoWAG codebook must use {dtype}, got {codebook.dtype}")
    codebook = codebook.contiguous()
    expected_codebook_shape = (1 << assignment_bits, group_size)
    if tuple(codebook.shape) != expected_codebook_shape:
        raise ValueError(
            f"NoWAG codebook must be {list(expected_codebook_shape)}, "
            f"got {tuple(codebook.shape)}"
        )
    return sources, codebook


__all__ = [
    "CODEBOOK_KEY",
    "DOWN_PROLOGUE_NORM",
    "DYNAMIC_E4M3_GROUP128_UE8M0",
    "FORMAT",
    "GATE_UP_EPILOGUE_NORM",
    "LEGACY_DSV4_FORMAT",
    "NO_ACTIVATION_ROUNDING",
    "RUNTIME_ASSIGNMENT_LAYOUT",
    "NowagModelRule",
    "get_nowag_model_rule",
    "load_nowag_expert_sources",
]
