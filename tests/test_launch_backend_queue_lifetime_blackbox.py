"""Black-box regression test for queues owned by a live backend handle."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import multiprocessing
import weakref
from typing import Any


class _FakeQueue:
    """Weak-referenceable stand-in; no IPC is performed by this test."""


def _queue_references(value: Any) -> list[weakref.ReferenceType[_FakeQueue]]:
    if isinstance(value, _FakeQueue):
        return [weakref.ref(value)]
    if isinstance(value, dict):
        references: list[weakref.ReferenceType[_FakeQueue]] = []
        for key, item in value.items():
            references.extend(_queue_references(key))
            references.extend(_queue_references(item))
        return references
    if isinstance(value, (list, tuple, set, frozenset)):
        references = []
        for item in value:
            references.extend(_queue_references(item))
        return references
    return []


def test_live_backend_handle_keeps_every_spawned_process_queue_alive(monkeypatch) -> None:
    import freetoken.server.api_server as api_server
    import freetoken.server.args as server_args_parser
    import freetoken.server.launch as launch

    @dataclass
    class FakeServerArgs:
        gpu: tuple[int, ...]
        shell_mode: bool
        tp_info: object
        num_tokenizer: int
        model_path: str
        zmq_address: str = "inproc://blackbox-zmq"
        create_address: str = "inproc://blackbox-create"

        def __getattr__(self, name: str) -> str:
            # Address spellings are an input detail outside this lifetime contract.
            if "zmq" in name or "create" in name or name.endswith("_address"):
                return f"inproc://blackbox-{name}"
            raise AttributeError(name)

    server_args = FakeServerArgs(
        gpu=(),
        shell_mode=False,
        tp_info=launch.DistributedInfo(0, 1),
        num_tokenizer=0,
        model_path="blackbox-model",
    )

    fake_processes: list[Any] = []

    class FakeProcess:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.constructor_args = args
            self.constructor_kwargs = kwargs
            self.started = False
            fake_processes.append(self)

        def start(self) -> None:
            self.started = True

    def fake_queue(*args: Any, **kwargs: Any) -> _FakeQueue:
        del args, kwargs
        return _FakeQueue()

    def fake_run_api_server(*args: Any, **kwargs: Any) -> None:
        callbacks = [value for value in (*args, *kwargs.values()) if callable(value)]
        assert len(callbacks) == 1, "API runner must receive one start_backend callback"

        handle = callbacks[0]()
        assert type(handle).__name__ == "BackendHandle"

        spawned_processes = [process for process in fake_processes if process.started]
        assert spawned_processes, "start_backend did not start a Process"

        queue_references: list[weakref.ReferenceType[_FakeQueue]] = []
        for process in spawned_processes:
            queue_references.extend(_queue_references(process.constructor_args))
            queue_references.extend(_queue_references(process.constructor_kwargs))

            # multiprocessing.Process discards its target/args/kwargs after start;
            # do the same so the fake itself cannot keep queues alive accidentally.
            process.constructor_args = ()
            process.constructor_kwargs = {}

        assert queue_references, "no Queue was passed to a spawned Process"
        gc.collect()

        assert handle is not None  # Keep the public handle observably live here.
        assert all(reference() is not None for reference in queue_references), (
            "a Queue passed to a spawned Process was collected while its "
            "BackendHandle remained live"
        )

    monkeypatch.setattr(
        server_args_parser,
        "parse_args",
        lambda *args, **kwargs: (server_args, False),
    )
    monkeypatch.setattr(api_server, "run_api_server", fake_run_api_server)
    monkeypatch.setattr(multiprocessing, "Queue", fake_queue)
    monkeypatch.setattr(multiprocessing, "Process", FakeProcess)
    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)

    launch.launch_server(argv=[])
