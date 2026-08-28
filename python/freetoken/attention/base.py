from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from freetoken.core import Batch


class AttnType(str, Enum):
    """Attention-type taxonomy, one value per KV-pool family. A model's attention
    groups map onto these (KVCacheGroupSpec.attn_type); each backend declares the
    set it can serve (BackendInfo.supported_types)."""

    FULL = "full"  # uniform causal MHA/GQA -> MHAKVCache
    SWA = "swa"  # sliding-window hybrid (window/sinks) -> HybridSWAKVCache
    MLA = "mla"  # plain latent-KV MLA -> MLAKVCache
    DSA = "dsa"  # latent-KV MLA + DSA sparse indexer -> DSAKVCache
    DSV4 = "dsv4"  # DSV4 window+compressed sparse -> DSV4PagedKVCache
    LINEAR = "linear"  # GDN/mamba state layers -> LinearStatePool
    # GQA block-sparse (MiniMax-M3): paged GQA K/V + a per-sparse-layer index-key
    # slab; the indexer picks top-k 128-token blocks per query -> BSAKVCache
    BSA = "bsa"

    @property
    def backend_driven(self) -> bool:
        # LINEAR layers reach their kernels directly (fla ops + batch.fla_metadata),
        # not through an attention backend, so they never constrain backend choice.
        return self is not AttnType.LINEAR


@dataclass
class AttentionSpec:
    sliding_window: int | None = None
    sm_scale: float | None = None
    sinks: torch.Tensor | None = None


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @property
    def supports_layer_range_graphs(self) -> bool:
        """Whether decode metadata can be staged for partial-model graph replay."""
        return False

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...

    def prepare_for_layer_range_capture(self, batch: Batch) -> None:
        del batch
        raise RuntimeError("attention backend does not support layer-range graphs")

    def prepare_for_layer_range_replay(self, batch: Batch) -> None:
        del batch
        raise RuntimeError("attention backend does not support layer-range graphs")

    def prepare_metadata_view(
        self,
        source: Batch,
        target: Batch,
    ) -> bool:
        """Attach metadata to a request/row view of an already prepared batch.

        Decode views rebuild by default. Prefill views may defer their rebuild
        until they are actually executed without the source mixed batch.
        """
        del source
        if target.is_decode_only:
            self.prepare_metadata(target)
            return True
        return False

    def capture_stable_decode_state(self, batch: Batch) -> object | None:
        """Return opaque reusable decode metadata, or None if replay must rebuild."""
        del batch
        return None

    def restore_stable_decode_state(self, batch: Batch, state: object) -> bool:
        """Restore a state returned above; false invalidates scheduler reuse."""
        del batch, state
        return False

    def reset_capture(self) -> None:
        """Drop CUDA-graph capture scratch so ``init_capture_graph`` can re-run after a
        runtime cache rebuild. The default clears the common capture state (guarded by
        ``hasattr`` so backends that hold only a subset, e.g. dsv4, are safe). Backends
        with extra per-bs graph state (FlashInfer ``graph_wrappers``) override this."""
        if hasattr(self, "capture"):
            self.capture = None
        if hasattr(self, "capture_bs"):
            self.capture_bs = []
        if hasattr(self, "max_graph_bs"):
            self.max_graph_bs = 0


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.uses_extend_path else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch, attn_spec=attn_spec)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.uses_extend_path else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)

    @property
    def supports_layer_range_graphs(self) -> bool:
        return self.decode_backend.supports_layer_range_graphs

    def prepare_for_layer_range_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_layer_range_capture(batch)

    def prepare_for_layer_range_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_layer_range_replay(batch)

    def prepare_metadata_view(self, source: Batch, target: Batch) -> bool:
        backend = (
            self.prefill_backend
            if target.uses_extend_path
            else self.decode_backend
        )
        if backend is self.prefill_backend:
            return backend.prepare_metadata_view(source, target)
        if self.prefill_backend is self.decode_backend:
            return backend.prepare_metadata_view(source, target)
        if target.is_decode_only:
            backend.prepare_metadata(target)
            return True
        return backend.prepare_metadata_view(source, target)

    def capture_stable_decode_state(self, batch: Batch) -> object | None:
        return self.decode_backend.capture_stable_decode_state(batch)

    def restore_stable_decode_state(self, batch: Batch, state: object) -> bool:
        return self.decode_backend.restore_stable_decode_state(batch, state)

    def reset_capture(self) -> None:
        # Only the decode backend is ever captured (see init_capture_graph above).
        self.decode_backend.reset_capture()
