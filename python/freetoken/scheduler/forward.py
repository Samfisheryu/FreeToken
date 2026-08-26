from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import torch

from freetoken.core import Batch

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


Indice2D: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "tuple[ForwardInput, ForwardOutput]"


__all__ = ["ForwardData", "ForwardInput", "Indice2D"]
