from __future__ import annotations

from typing import Any, Dict, Iterable

import torch

from . import register_processor
from .tensor_processor import TensorProcessor


@register_processor("regression_sequence")
class RegressionSequenceProcessor(TensorProcessor):
    """
Label processor for variable-length remaining LoS regression sequences.
Subclasses TensorProcessor with dtype=torch.float32 and spatial_dims=(True,).
Converts the raw list of per-hour remaining LoS values produced by
RemainingLengthOfStayTPC_MIMIC4 into a 1D float tensor.

Input:
    list[float] of length T — remaining LoS in days at each prediction hour.
    T varies per stay (max 332 for a 336-hour stay starting at hour 5).

Returns:
    torch.FloatTensor of shape (T,)

Examples:
    >>> processor = RegressionSequenceProcessor()
    >>> processor.fit([], "y")
    >>> out = processor.process([2.0, 1.5, 1.0, 0.5])
    >>> out.shape    # torch.Size([4])
    >>> out.dtype    # torch.float32
"""

    def __init__(self) -> None:
        """Initialise with float32 dtype, spatial T dimension, and fixed n_dim=1."""
        super().__init__(dtype=torch.float32, spatial_dims=(True,))
        self._n_dim = 1

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """No-op fit method since regression sequence labels require no fitting."""
        return

    def size(self) -> int:
        """Regression sequence labels have fixed size of 1."""
        return 1

