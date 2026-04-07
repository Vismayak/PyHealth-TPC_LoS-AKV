from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("tpc_timeseries")
class TPCTimeseriesProcessor(FeatureProcessor):
    """TPC-specific time-series processor.

    Input value format (dict) is produced by `RemainingLengthOfStayTPC_MIMIC4`:
      {
        "prefill_start": datetime,
        "icu_start": datetime,
        "pred_start": datetime,
        "pred_end": datetime,
        "feature_itemids": list[str],
        "long_df": {"timestamp": [...], "itemid": [...], "value": [...], "source": [...]}
      }

    Output:
      torch.FloatTensor of shape (T, F, 2) where:
        - [:, :, 0] = forward-filled (hourly) values
        - [:, :, 1] = decay indicators (0.75 ** hours_since_last_observed)
    """

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        decay_base: float = 0.75,
        clip_min: float = -4.0,
        clip_max: float = 4.0,
    ):
        self.sampling_rate = sampling_rate
        self.decay_base = float(decay_base)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        # Feature-dependent robust scaling parameters, keyed by itemid.
        self._p5: Dict[str, float] = {}
        self._p95: Dict[str, float] = {}
        self._feature_itemids: List[str] = []

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        values_by_item: Dict[str, List[float]] = {}
        feature_itemids: List[str] | None = None

        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            payload = sample[field]
            if not isinstance(payload, dict):
                continue
            if feature_itemids is None:
                feature_itemids = [str(x) for x in payload.get("feature_itemids", [])]
            long_df = payload.get("long_df") or {}
            itemids = long_df.get("itemid", [])
            vals = long_df.get("value", [])
            for itemid, v in zip(itemids, vals):
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                key = str(itemid)
                values_by_item.setdefault(key, []).append(fv)

        self._feature_itemids = feature_itemids or sorted(values_by_item.keys())
        for itemid in self._feature_itemids:
            arr = np.asarray(values_by_item.get(itemid, []), dtype=float)
            if arr.size == 0:
                self._p5[itemid] = 0.0
                self._p95[itemid] = 1.0
                continue
            self._p5[itemid] = float(np.nanpercentile(arr, 5))
            self._p95[itemid] = float(np.nanpercentile(arr, 95))

    def _scale(self, itemid: str, x: float) -> float:
        p5 = self._p5.get(itemid, 0.0)
        p95 = self._p95.get(itemid, 1.0)
        if p95 == p5:
            return 0.0
        scaled = 2.0 * (x - p5) / (p95 - p5) - 1.0
        return float(np.clip(scaled, self.clip_min, self.clip_max))

    def process(self, value: Dict[str, Any]) -> torch.Tensor:
        prefill_start: datetime = value["prefill_start"]
        pred_start: datetime = value["pred_start"]
        pred_end: datetime = value["pred_end"]
        feature_itemids: Sequence[str] = value["feature_itemids"]
        long_df = value["long_df"]

        # Compute hourly grid.
        step_hours = int(self.sampling_rate.total_seconds() // 3600)
        if step_hours != 1:
            raise ValueError("TPCTimeseriesProcessor currently supports 1-hour sampling only.")

        total_steps = int((pred_end - prefill_start).total_seconds() // 3600)
        if total_steps <= 0:
            raise ValueError("Invalid time window for TPC time series.")

        # Determine the prediction window slice indices (inclusive start, exclusive end).
        start_idx = int((pred_start - prefill_start).total_seconds() // 3600)
        pred_steps = int((pred_end - pred_start).total_seconds() // 3600)
        if pred_steps <= 0:
            raise ValueError("Invalid prediction window for TPC time series.")

        F = len(feature_itemids)
        sampled = np.full((total_steps, F), np.nan, dtype=float)
        observed = np.zeros((total_steps, F), dtype=bool)

        # Map itemid -> column index.
        col_index = {str(itemid): j for j, itemid in enumerate(feature_itemids)}

        # Place observations into hourly bins (keep last observation in each hour).
        ts_list = long_df.get("timestamp", [])
        item_list = long_df.get("itemid", [])
        val_list = long_df.get("value", [])
        for ts, itemid, v in zip(ts_list, item_list, val_list):
            if ts is None or itemid is None or v is None:
                continue
            itemid = str(itemid)
            if itemid not in col_index:
                continue
            try:
                t: datetime = ts  # already datetime from task
                idx = int((t - prefill_start).total_seconds() // 3600)
                if idx < 0 or idx >= total_steps:
                    continue
                fv = self._scale(itemid, float(v))
            except Exception:
                continue
            j = col_index[itemid]
            sampled[idx, j] = fv
            observed[idx, j] = True

        # Forward fill + decay.
        values_ff = np.zeros((total_steps, F), dtype=float)
        decay = np.zeros((total_steps, F), dtype=float)
        for j in range(F):
            last_value = 0.0
            last_seen = None  # type: int | None
            for t in range(total_steps):
                if observed[t, j] and not np.isnan(sampled[t, j]):
                    last_value = float(sampled[t, j])
                    last_seen = t
                    values_ff[t, j] = last_value
                    decay[t, j] = 1.0
                else:
                    values_ff[t, j] = last_value
                    if last_seen is None:
                        decay[t, j] = 0.0
                    else:
                        dt = t - last_seen
                        decay[t, j] = float(self.decay_base ** dt)

        # Slice to prediction window.
        values_ff = values_ff[start_idx : start_idx + pred_steps]
        decay = decay[start_idx : start_idx + pred_steps]

        out = np.stack([values_ff, decay], axis=-1)  # (T, F, 2)
        return torch.tensor(out, dtype=torch.float32)

    def size(self) -> int:
        return len(self._feature_itemids)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        # (time, feature, channel)
        return (True, False, False)

