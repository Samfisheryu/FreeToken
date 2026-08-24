from __future__ import annotations

import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine import EngineConfig
from freetoken.server.args import ServerArgs


def _engine_config(distributed_port: int = 2333) -> EngineConfig:
    return EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        distributed_port=distributed_port,
    )


def test_offline_engine_configs_can_use_different_distributed_ports():
    first = _engine_config(24101)
    second = _engine_config(24102)

    assert first.distributed_addr == "tcp://127.0.0.1:24101"
    assert second.distributed_addr == "tcp://127.0.0.1:24102"
    assert _engine_config().distributed_addr == "tcp://127.0.0.1:2333"


def test_llm_passes_distributed_port_to_scheduler_config(monkeypatch):
    from freetoken.llm import LLM
    from freetoken.scheduler import Scheduler

    configs = []
    monkeypatch.setattr(Scheduler, "__init__", lambda self, config: configs.append(config))

    LLM("unused", distributed_port=24201)
    LLM("unused", distributed_port=24202)

    assert [config.distributed_addr for config in configs] == [
        "tcp://127.0.0.1:24201",
        "tcp://127.0.0.1:24202",
    ]


def test_server_args_still_use_server_port_plus_one():
    config = ServerArgs(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        server_port=1919,
        distributed_port=24999,
    )

    assert config.distributed_addr == "tcp://127.0.0.1:1920"
