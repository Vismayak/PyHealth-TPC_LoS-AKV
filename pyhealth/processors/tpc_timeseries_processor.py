from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("tpc_timeseries")
class TPCTimeseriesProcessor(FeatureProcessor):
    """
    Feature processor for TPC time-series inputs.

    Consumes the raw payload produced by RemainingLengthOfStayTPC_MIMIC4 task and outputs
    a (T, F, 2) tensor of hourly-resampled, forward-filled values with decay indicators.
    Scaling uses per-feature 5th/95th percentiles computed during fit().

    Input format (dict) produced by RemainingLengthOfStayTPC_MIMIC4:
        {
            "prefill_start": datetime,   # start of forward-fill seeding window (icu_start - 24h)
            "icu_start":     datetime,   # ICU admission time
            "pred_start":    datetime,   # first prediction hour (icu_start + 5h)
            "pred_end":      datetime,   # last prediction hour (min(icu_end, icu_start + 336h))
            "feature_itemids": list[str],  # ordered list of itemids defining column layout
            "long_df": {
                "timestamp": list[datetime],
                "itemid":    list[str],
                "value":     list[float],
                "source":    list[str],   # "chartevents" or "labevents"
            }
        }

    Args:
        sampling_rate: Resampling interval. Must be 1 hour (paper setting). Default: timedelta(hours=1).
        decay_base:    Base for decay indicator: decay = decay_base ** hours_since_last_obs. Default: 0.75.
        clip_min:      Lower clip bound after scaling. Default: -4.0.
        clip_max:      Upper clip bound after scaling. Default:  4.0.

    Returns:
        torch.FloatTensor of shape (T, F, 2) where T = prediction hours, F = number of features:
            [:, :, 0] = forward-filled scaled values. Initialised to 0.0 before first observation.
            [:, :, 1] = decay indicators:
                        1.0  = fresh observation at this timestep
                        0.75^j = j hours since last observation
                        0.0  = feature never observed up to this point

    Examples:
        >>> from datetime import datetime, timedelta
        >>> processor = TPCTimeseriesProcessor()
        >>> prefill = datetime(2020, 1, 1, 0)
        >>> payload = {
        ...     "prefill_start": prefill,
        ...     "icu_start":     prefill,
        ...     "pred_start":    prefill + timedelta(hours=5),
        ...     "pred_end":      prefill + timedelta(hours=10),
        ...     "feature_itemids": ["A", "B"],
        ...     "long_df": {
        ...         "timestamp": [prefill],
        ...         "itemid":    ["A"],
        ...         "value":     [80.0],
        ...         "source":    ["chartevents"],
        ...     }
        ... }
        >>> processor.fit([{"ts": payload}], "ts")
        >>> out = processor.process(payload)
        >>> out.shape          # (5, 2, 2) — T=5, F=2, channels=2
        >>> out[0, 0, 1]       # 0.75**5 — feature A last seen at hour 0, pred starts at hour 5
        >>> out[0, 1, 1]       # 0.0 — feature B never observed
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
        """Compute per-feature 5th/95th percentiles from all samples for robust scaling."""
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
        """Scale a single value using the per-feature 5th/95th percentiles."""
        p5 = self._p5.get(itemid, 0.0)
        p95 = self._p95.get(itemid, 1.0)
        if p95 == p5:
            return 0.0
        scaled = 2.0 * (x - p5) / (p95 - p5) - 1.0
        return float(np.clip(scaled, self.clip_min, self.clip_max))

    def process(self, value: Dict[str, Any]) -> torch.Tensor:
        """Resample observations to hourly grid, forward-fill, compute decay indicators, and return (T, F, 2) tensor."""
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
        """Return the number of time-series features, equal to len(feature_itemids)."""

        return len(self._feature_itemids)

    def is_token(self) -> bool:
        """Time-series values are continuous, not discrete tokens."""
        return False

    def schema(self) -> tuple[str, ...]:
        """Output is a tuple of (value, decay) tensors."""
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 3D tensor (time, feature, channel)."""
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        """Time dimension is spatial, feature and channel dimensions are not."""
        return (True, False, False)

