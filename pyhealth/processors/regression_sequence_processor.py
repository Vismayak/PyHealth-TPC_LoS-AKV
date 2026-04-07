from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("regression_sequence")
class RegressionSequenceProcessor(FeatureProcessor):
    """Label processor for variable-length regression sequences.

    Input: a list/sequence of floats (length T)
    Output: torch.FloatTensor of shape (T,)
    """

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        return

    def process(self, value: Sequence[float]) -> torch.Tensor:
        return torch.tensor(list(value), dtype=torch.float32)

    def size(self) -> int:
        # Variable-length sequence; size is not meaningful.
        return 1

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (True,)

