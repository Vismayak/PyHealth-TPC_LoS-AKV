from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class TPCBlock(nn.Module):
    """One TPC layer (temporal + pointwise + dense skip connections).

    This block follows the key structural ideas in Rocheteau et al. (CHIL 2021):
    - **Temporal**: per-feature causal convolution (no parameter sharing across features)
    - **Pointwise**: per-time-step mixing across features (shared across time)
    - **Dense skips**: propagate original + previous pointwise outputs

    Shapes:
      input  x: (B, T, R, C_in)
      output h: (B, T, R + Z, C_out) where C_out = Y + 1
    """

    def __init__(
        self,
        *,
        in_features: int,
        in_channels: int,
        temporal_channels: int,
        pointwise_channels: int,
        kernel_size: int,
        dilation: int,
        main_dropout: float,
        temporal_dropout: float,
        use_batchnorm: bool = True,
        static_dim: int = 0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.in_channels = int(in_channels)
        self.temporal_channels = int(temporal_channels)
        self.pointwise_channels = int(pointwise_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.use_batchnorm = bool(use_batchnorm)
        self.static_dim = int(static_dim)

        # Temporal branch: grouped Conv1d => separate weights per feature.
        self.temporal_conv = nn.Conv1d(
            in_channels=self.in_features * self.in_channels,
            out_channels=self.in_features * self.temporal_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.in_features,
            bias=True,
        )
        self.bn_temporal = nn.BatchNorm1d(self.in_features * self.temporal_channels)
        self.dropout_temporal = nn.Dropout(temporal_dropout)

        # Pointwise branch: Linear applied to each time step.
        # Input to pointwise uses r = [x_value_skip, temporal_out] => channels (Y + 1).
        point_in_dim = self.in_features * (self.temporal_channels + 1) + self.static_dim
        self.pointwise = nn.Linear(point_in_dim, self.pointwise_channels)
        self.bn_pointwise = nn.BatchNorm1d(self.pointwise_channels)
        self.dropout_main = nn.Dropout(main_dropout)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, static: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, R, C_in)
            static: (B, S) or None
        Returns:
            (B, T, R + Z, Y + 1)
        """
        B, T, R, C = x.shape
        if R != self.in_features or C != self.in_channels:
            raise ValueError(f"TPCBlock got x shape {x.shape}, expected (B,T,{self.in_features},{self.in_channels})")

        # === Temporal branch ===
        # reshape to (B, R*C, T) for grouped conv; causal left padding.
        x_tc = x.permute(0, 2, 3, 1).reshape(B, R * C, T)
        pad = (self.kernel_size - 1) * self.dilation
        x_tc = F.pad(x_tc, (pad, 0), mode="constant", value=0.0)
        t_out = self.temporal_conv(x_tc)  # (B, R*Y, T)
        if self.use_batchnorm:
            t_out = self.bn_temporal(t_out)
        t_out = self.dropout_temporal(t_out)
        t_out = t_out.reshape(B, R, self.temporal_channels, T).permute(0, 3, 1, 2)  # (B,T,R,Y)

        # Skip: use the (current) value channel as 1 extra channel.
        x_value = x[..., 0:1]  # (B,T,R,1)
        r = torch.cat([x_value, t_out], dim=-1)  # (B,T,R,Y+1)

        # === Pointwise branch ===
        r_flat = r.reshape(B, T, R * (self.temporal_channels + 1))
        if static is not None:
            static_rep = static.unsqueeze(1).expand(B, T, static.shape[-1])
            p_in = torch.cat([r_flat, static_rep], dim=-1)
        else:
            p_in = r_flat

        pw = self.pointwise(p_in)  # (B,T,Z)
        if self.use_batchnorm:
            pw_bn = self.bn_pointwise(pw.reshape(B * T, -1)).reshape(B, T, -1)
        else:
            pw_bn = pw
        pw_bn = self.dropout_main(pw_bn)

        # Broadcast pointwise outputs to (B,T,Z,Y+1) so they can be treated as new "features".
        pw_feat = pw_bn.unsqueeze(-1).expand(B, T, self.pointwise_channels, self.temporal_channels + 1)

        h = torch.cat([r, pw_feat], dim=2)  # (B,T,R+Z,Y+1)
        return self.relu(h)


class TPC(BaseModel):
    """Temporal Pointwise Convolution (TPC) for remaining LoS (sequence regression).

    Expected inputs (from this implementation's task + processors):
      - ts: (B, T, F, 2) where channels are (value, decay)
      - static: (B, S)
      - y: (B, T) remaining LoS in days (padded with 0)

    Outputs:
      - y_prob: (B, T) predicted remaining LoS in days
      - loss: masked MSLE over non-padding timesteps
    """

    def __init__(
        self,
        dataset: SampleDataset,
        *,
        temporal_channels: int = 11,
        pointwise_channels: int = 5,
        num_layers: int = 8,
        kernel_size: int = 5,
        main_dropout: float = 0.0,
        temporal_dropout: float = 0.05,
        use_batchnorm: bool = True,
        final_hidden: int = 36,
        decay_clip_min_days: float = 1.0 / 48.0,
        decay_clip_max_days: float = 100.0,
    ):
        super().__init__(dataset=dataset)
        assert "ts" in self.feature_keys and "static" in self.feature_keys, (
            "TPC expects dataset.input_schema to contain 'ts' and 'static'."
        )
        assert len(self.label_keys) == 1, "TPC currently supports a single label key."
        self.label_key = self.label_keys[0]

        self.mode = "regression"

        self.temporal_channels = int(temporal_channels)
        self.pointwise_channels = int(pointwise_channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.use_batchnorm = bool(use_batchnorm)
        self.final_hidden = int(final_hidden)

        self.min_days = float(decay_clip_min_days)
        self.max_days = float(decay_clip_max_days)

        # We infer feature/static dimensions from the dataset processors.
        ts_proc = dataset.input_processors["ts"]
        static_proc = dataset.input_processors["static"]
        self.F = ts_proc.size()
        self.S = static_proc.size()

        # Stack TPC blocks; feature dimension grows by Z each layer.
        blocks = []
        in_features = self.F
        in_channels = 2  # (value, decay)
        for layer_idx in range(self.num_layers):
            blocks.append(
                TPCBlock(
                    in_features=in_features,
                    in_channels=in_channels,
                    temporal_channels=self.temporal_channels,
                    pointwise_channels=self.pointwise_channels,
                    kernel_size=self.kernel_size,
                    dilation=layer_idx + 1,
                    main_dropout=main_dropout,
                    temporal_dropout=temporal_dropout,
                    use_batchnorm=self.use_batchnorm,
                    static_dim=self.S,
                )
            )
            # after first block, channels become (Y+1)
            in_channels = self.temporal_channels + 1
            in_features = in_features + self.pointwise_channels
        self.blocks = nn.ModuleList(blocks)

        # Final per-time-step head (2-layer pointwise MLP).
        final_in = in_features * (self.temporal_channels + 1) + self.S
        self.head_fc1 = nn.Linear(final_in, self.final_hidden)
        self.head_relu = nn.ReLU()
        self.head_fc2 = nn.Linear(self.final_hidden, 1)

        self.hardtanh = nn.Hardtanh(min_val=self.min_days, max_val=self.max_days)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        ts: torch.Tensor = kwargs["ts"].to(self.device)          # (B,T,F,2) padded
        static: torch.Tensor = kwargs["static"].to(self.device)  # (B,S)
        y_true: Optional[torch.Tensor] = None
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)      # (B,T) padded

        B, T, F, C = ts.shape
        if C != 2:
            raise ValueError(f"TPC expects ts channels=2, got {C}.")
        if F != self.F:
            raise ValueError(f"TPC expects F={self.F} features, got {F}.")

        h = ts
        for block in self.blocks:
            h = block(h, static=static)  # grows feature dimension, channels -> (Y+1)

        # Final predictions per hour.
        h_flat = h.reshape(B, T, -1)  # (B,T, features*channels)
        static_rep = static.unsqueeze(1).expand(B, T, static.shape[-1])
        head_in = torch.cat([h_flat, static_rep], dim=-1)

        logit = self.head_fc2(self.head_relu(self.head_fc1(head_in))).squeeze(-1)  # (B,T)

        # Predict log(LoS) then exponentiate + clip (paper Appendix A).
        y_pred = self.hardtanh(torch.exp(logit))

        results: Dict[str, torch.Tensor] = {
            "logit": logit,
            "y_prob": y_pred,
        }

        if y_true is not None:
            # mask: padded labels are 0 (pad_sequence default), true labels are clipped >= 1/48.
            mask = (y_true != 0).float()
            # MSLE = mean((log(y_pred) - log(y_true))^2) over valid timesteps.
            eps = 1e-8
            log_pred = torch.log(torch.clamp(y_pred, min=eps))
            log_true = torch.log(torch.clamp(y_true, min=eps))
            se = (log_pred - log_true) ** 2
            loss = (se * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def forward_from_embedding(self, **kwargs) -> Dict[str, torch.Tensor]:
        # This model already consumes dense tensors.
        return self.forward(**kwargs)

