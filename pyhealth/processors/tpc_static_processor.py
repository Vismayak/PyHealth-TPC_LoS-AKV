from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("tpc_static")
class TPCStaticProcessor(FeatureProcessor):
    """Static feature encoder for TPC (paper Table 6 style).

    Input: a dict of raw values, e.g.:
      {
        "gender": "M",
        "race": "...",
        "admission_location": "...",
        "insurance": "...",
        "first_careunit": "...",
        "hour_of_admission": 13,
        "admission_height": 170.0,
        "admission_weight": 80.0,
        "gcs_eye": 4.0,
        "gcs_motor": 6.0,
        "gcs_verbal": 5.0,
        "anchor_age": 65
      }

    Output: a 1D float tensor containing:
      [one-hot categoricals..., scaled numerics...]
    """

    CATEGORICAL_KEYS: Tuple[str, ...] = (
        "gender",
        "race",
        "admission_location",
        "insurance",
        "first_careunit",
    )
    NUMERIC_KEYS: Tuple[str, ...] = (
        "hour_of_admission",
        "admission_height",
        "admission_weight",
        "gcs_eye",
        "gcs_motor",
        "gcs_verbal",
        "anchor_age",
    )

    def __init__(self, clip_min: float = -4.0, clip_max: float = 4.0):
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self._cat_vocab: Dict[str, List[str]] = {k: [] for k in self.CATEGORICAL_KEYS}
        self._cat_index: Dict[str, Dict[str, int]] = {k: {} for k in self.CATEGORICAL_KEYS}

        self._p5: Dict[str, float] = {}
        self._p95: Dict[str, float] = {}

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        cat_values: Dict[str, set[str]] = {k: set() for k in self.CATEGORICAL_KEYS}
        num_values: Dict[str, List[float]] = {k: [] for k in self.NUMERIC_KEYS}

        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            s = sample[field]
            if not isinstance(s, dict):
                continue

            for k in self.CATEGORICAL_KEYS:
                v = s.get(k, None)
                if v is None:
                    continue
                cat_values[k].add(str(v))

            for k in self.NUMERIC_KEYS:
                v = s.get(k, None)
                if v is None:
                    continue
                try:
                    num_values[k].append(float(v))
                except Exception:
                    continue

        for k in self.CATEGORICAL_KEYS:
            vocab = sorted(cat_values[k])
            # reserve index 0 for unknown/missing
            self._cat_vocab[k] = ["<UNK>"] + vocab
            self._cat_index[k] = {tok: i for i, tok in enumerate(self._cat_vocab[k])}

        for k in self.NUMERIC_KEYS:
            arr = np.asarray(num_values[k], dtype=float)
            if arr.size == 0:
                self._p5[k] = 0.0
                self._p95[k] = 1.0
            else:
                self._p5[k] = float(np.nanpercentile(arr, 5))
                self._p95[k] = float(np.nanpercentile(arr, 95))

    def _scale(self, key: str, x: float) -> float:
        p5 = self._p5.get(key, 0.0)
        p95 = self._p95.get(key, 1.0)
        if p95 == p5:
            return 0.0
        scaled = 2.0 * (x - p5) / (p95 - p5) - 1.0
        return float(np.clip(scaled, self.clip_min, self.clip_max))

    def process(self, value: Dict[str, Any]) -> torch.Tensor:
        parts: List[float] = []

        # Categorical one-hots.
        for k in self.CATEGORICAL_KEYS:
            vocab = self._cat_vocab.get(k, ["<UNK>"])
            idx_map = self._cat_index.get(k, {"<UNK>": 0})
            one_hot = np.zeros(len(vocab), dtype=float)
            raw = value.get(k, None)
            tok = "<UNK>" if raw is None else str(raw)
            one_hot[idx_map.get(tok, 0)] = 1.0
            parts.extend(one_hot.tolist())

        # Numeric robust scaling.
        for k in self.NUMERIC_KEYS:
            raw = value.get(k, None)
            if raw is None:
                parts.append(0.0)
                continue
            try:
                parts.append(self._scale(k, float(raw)))
            except Exception:
                parts.append(0.0)

        return torch.tensor(parts, dtype=torch.float32)

    def size(self) -> int:
        cat_size = sum(len(self._cat_vocab.get(k, ["<UNK>"])) for k in self.CATEGORICAL_KEYS)
        return cat_size + len(self.NUMERIC_KEYS)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

