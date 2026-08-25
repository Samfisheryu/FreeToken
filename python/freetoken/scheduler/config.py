from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    batching_policy: Literal["legacy", "mixed", "layered", "joint"] = "legacy"
    prefill_layer_group_size: int = 2
    prefill_wave_max_chunks: int = 4
    prefill_execution: Literal["serial", "concurrent"] = "serial"
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def __post_init__(self) -> None:
        if self.prefill_layer_group_size < 1:
            raise ValueError("prefill_layer_group_size must be >= 1")
        if self.prefill_wave_max_chunks < 1:
            raise ValueError("prefill_wave_max_chunks must be >= 1")
        if self.prefill_execution not in ("serial", "concurrent"):
            raise ValueError(
                "prefill_execution must be either 'serial' or 'concurrent'"
            )

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
