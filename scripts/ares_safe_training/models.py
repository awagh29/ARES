"""Two-branch state-augmented SAITS for 14 physical LiDAR sector features.

The temporal branch derives from the project's MIT-attributed SAITS model.
See ../../LICENSES/SAITS-MIT.txt and saits_endpoint_model.py. The extra state
branch attends across current variables, including absolute NED position.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from saits_endpoint_model import SAITSEndpointModel


@dataclass
class ModelConfig:
    window_size: int = 50
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    temporal_layers: int = 1
    state_layers: int = 2
    dropout: float = 0.1

    def validate(self):
        for name in ("window_size", "d_model", "n_heads", "d_ff", "temporal_layers", "state_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.window_size < 2 or self.d_model % 2 or self.d_model % self.n_heads:
            raise ValueError("Invalid history/attention dimensions.")
        if min(self.temporal_layers, self.state_layers, self.d_ff) < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid model depth, width or dropout.")


class TemporalEncoder(SAITSEndpointModel):
    """Retain both SAITS stages and expose the current endpoint embedding."""

    def forward(self, values, observed_mask):
        if tuple(values.shape[1:]) != (self.n_steps, self.n_features) or values.shape != observed_mask.shape:
            raise ValueError("Temporal values/mask have incompatible shapes.")
        mask = observed_mask.to(values.dtype)
        filled = values * mask
        hidden = self.embedding_first(torch.cat((filled, mask), dim=-1))
        hidden = self.first_embedding_dropout(hidden + self.position_encoding.to(hidden.dtype))
        for layer in self.first_block:
            hidden, _ = layer(hidden, self.allowed_attention)
        first = self.reduce_first(hidden)
        completed = mask * filled + (1 - mask) * first
        hidden = self.embedding_second(torch.cat((completed, mask), dim=-1))
        hidden = hidden + self.position_encoding.to(hidden.dtype)
        attention = None
        for index, layer in enumerate(self.second_block):
            hidden, weights = layer(hidden, self.allowed_attention, need_weights=index == len(self.second_block) - 1)
            if weights is not None:
                attention = weights.mean(dim=1)
        second = self.reduce_second_gamma(torch.relu(self.reduce_second_beta(hidden)))
        mixing = torch.sigmoid(self.combination_gate(torch.cat((mask[:, -1], attention[:, -1]), dim=-1)))[:, :14]
        first_endpoint, second_endpoint = first[:, -1, :14], second[:, -1, :14]
        prediction = (1 - mixing) * second_endpoint + mixing * first_endpoint
        return {"embedding": hidden[:, -1], "prediction": prediction,
                "stage1": first_endpoint, "stage2": second_endpoint}


class StateEncoder(nn.Module):
    """Feature-token attention across 34 current variables, rather than time."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.value_weight = nn.Parameter(torch.randn(34, config.d_model) * 0.02)
        self.feature_identity = nn.Parameter(torch.randn(34, config.d_model) * 0.02)
        self.availability_embedding = nn.Embedding(2, config.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff,
                                          config.dropout, batch_first=True, norm_first=True,
                                          activation="gelu")
        self.attention = nn.TransformerEncoder(layer, config.state_layers,
                                               norm=nn.LayerNorm(config.d_model),
                                               enable_nested_tensor=False)

    def forward(self, values, observed_mask):
        if values.shape[-1] != 34 or values.shape != observed_mask.shape:
            raise ValueError("The state branch requires [B,34] values and masks.")
        available = observed_mask.bool()
        safe = values.masked_fill(~available, 0)
        tokens = (safe.unsqueeze(-1) * self.value_weight + self.feature_identity
                  + self.availability_embedding(available.long()))
        tokens = torch.cat((self.cls_token.expand(len(values), -1, -1), tokens), dim=1)
        return self.attention(tokens)[:, 0]


class AugmentedSAITS(nn.Module):
    """Historical measurements + current absolute state -> 14 joint outputs.

    history/current values must already be normalized and finite-filled, with
    separate masks. Current state is [absolute NED position, current 31 inputs].
    Physical output bounds are architectural, not post-safety-control clipping.
    """

    def __init__(self, config: ModelConfig | dict | None = None):
        super().__init__()
        self.config = ModelConfig(**config) if isinstance(config, dict) else config or ModelConfig()
        self.config.validate()
        cfg = self.config
        self.temporal = TemporalEncoder(cfg.window_size, 31, 14, d_model=cfg.d_model,
                                        n_heads=cfg.n_heads, d_ff=cfg.d_ff,
                                        n_layers=cfg.temporal_layers, dropout=cfg.dropout)
        self.state = StateEncoder(cfg)
        self.decoder = nn.Sequential(nn.Linear(2 * cfg.d_model + 14, cfg.d_ff), nn.GELU(),
                                     nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, 14))
        self.register_buffer("physical_scale", torch.tensor([7.] * 7 + [1.] * 7))

    def forward(self, history_values, history_mask, state_values, state_mask):
        history = self.temporal(history_values, history_mask)
        state = self.state(state_values, state_mask)
        joined = torch.cat((history["embedding"], state, history["prediction"]), dim=-1)
        normalized = torch.sigmoid(self.decoder(joined))
        return {"normalized": normalized, "physical": normalized * self.physical_scale,
                "temporal_prediction": history["prediction"],
                "stage1": history["stage1"], "stage2": history["stage2"]}

    def architecture(self):
        return {"schema_version": "ares-state-augmented-saits-v1", "config": asdict(self.config),
                "history_features": 31, "state_features": 34, "targets": 14,
                "state_order": "absolute-NED-position-then-current-31-features",
                "online_map_input": False, "output_units": ["m"] * 7 + ["fraction"] * 7}
