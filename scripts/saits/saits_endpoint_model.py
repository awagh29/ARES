"""SAITS architecture adapted to supervised LiDAR window endpoints.

Derived from Wenjie Du's MIT-licensed official SAITS implementation:
https://github.com/WenjieDu/SAITS/blob/main/modeling/saits.py
https://github.com/WenjieDu/SAITS/blob/main/modeling/layers.py
Copyright 2022 Wenjie Du. See ../LICENSES/SAITS-MIT.txt.

The two diagonally masked self-attention stages and attention-based mixture
are retained. Layers have independent parameters rather than optional upstream
ALBERT-style sharing. The caller supervises the first ``n_targets`` features at
the window's last row; this is not the upstream full-sequence training objective
and the module is not compatible with upstream checkpoint parameter names.

Only PyTorch and the Python standard library are required. There is no GRU.
"""

import math

import torch


class _DMSALayer(torch.nn.Module):
    """Pre-normalized attention and feed-forward residual sublayers."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_norm = torch.nn.LayerNorm(d_model)
        self.query = torch.nn.Linear(d_model, d_model, bias=False)
        self.key = torch.nn.Linear(d_model, d_model, bias=False)
        self.value = torch.nn.Linear(d_model, d_model, bias=False)
        self.output = torch.nn.Linear(d_model, d_model, bias=False)
        self.attention_residual_dropout = torch.nn.Dropout(dropout)
        self.ff_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.ff_first = torch.nn.Linear(d_model, d_ff)
        self.ff_second = torch.nn.Linear(d_ff, d_model)
        self.ff_residual_dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden, allowed_attention, need_weights=False):
        batch, steps, d_model = hidden.shape
        normalized = self.attention_norm(hidden)

        def project(layer):
            return layer(normalized).reshape(
                batch, steps, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query = project(self.query)
        key = project(self.key)
        value = project(self.value)
        weights = None
        if need_weights:
            # Explicitly disable autocast: calling .float() alone does not
            # prevent an enclosing autocast context from downcasting matmul.
            # The final stage needs these differentiable weights for its gate.
            with torch.autocast(device_type=query.device.type, enabled=False):
                scores = torch.matmul(
                    query.float(), key.float().transpose(-2, -1)
                ) * (1.0 / math.sqrt(self.head_dim))
                scores = scores.masked_fill(~allowed_attention, float("-inf"))
                weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights.to(dtype=value.dtype), value)
        else:
            # SDPA bool True means ALLOWED; the original SAITS eye mask instead
            # uses True/1 for BLOCKED entries. No weights are materialized here.
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed_attention,
                dropout_p=0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, steps, d_model
        )
        hidden = hidden + self.attention_residual_dropout(self.output(attended))
        feed_forward = self.ff_second(
            torch.nn.functional.relu(self.ff_first(self.ff_norm(hidden)))
        )
        hidden = hidden + self.ff_residual_dropout(feed_forward)
        return hidden, weights


class SAITSEndpointModel(torch.nn.Module):
    """Two-stage SAITS with an attention-weighted endpoint reconstruction.

    Inputs are finite floating-point ``values[B,T,F]`` and an equally shaped
    boolean or floating-point observation mask (one means observed). Missing
    inputs use zero placeholders. Finiteness and binary mask validity should be
    checked once during data preparation, avoiding per-step GPU synchronization.

    Returns ``prediction``, ``stage1``, and ``stage2``, each ``[B,n_targets]``.
    With ``return_attention=True``, also returns the final stage's head-averaged
    ``attention[B,T,T]``. Attention excludes the diagonal and is bidirectional
    within the supplied historical window, whose last row is the target time.
    Outputs are unbounded; target scaling and loss belong to the trainer.
    """

    def __init__(
        self,
        n_steps,
        n_features,
        n_targets,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        dimensions = {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_targets": n_targets,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "n_layers": n_layers,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if n_steps < 2:
            raise ValueError("n_steps must be at least two for diagonal masking")
        if n_targets > n_features:
            raise ValueError("n_targets cannot exceed n_features")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if d_model % 2:
            raise ValueError("d_model must be even for sinusoidal position encoding")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0,1)")
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_targets = n_targets
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers

        positions = torch.arange(n_steps, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        position_encoding = torch.zeros(n_steps, d_model, dtype=torch.float32)
        position_encoding[:, 0::2] = torch.sin(positions * frequencies)
        position_encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position_encoding", position_encoding.unsqueeze(0))
        self.register_buffer(
            "allowed_attention",
            ~torch.eye(n_steps, dtype=torch.bool),
            persistent=False,
        )

        self.embedding_first = torch.nn.Linear(2 * n_features, d_model)
        self.embedding_second = torch.nn.Linear(2 * n_features, d_model)
        self.first_embedding_dropout = torch.nn.Dropout(dropout)
        self.first_block = torch.nn.ModuleList([
            _DMSALayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.second_block = torch.nn.ModuleList([
            _DMSALayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.reduce_first = torch.nn.Linear(d_model, n_features)
        self.reduce_second_beta = torch.nn.Linear(d_model, n_features)
        self.reduce_second_gamma = torch.nn.Linear(n_features, n_features)
        self.combination_gate = torch.nn.Linear(n_features + n_steps, n_features)

    def forward(self, values, observed_mask, *, return_attention=False):
        if values.ndim != 3 or tuple(values.shape[1:]) != (
            self.n_steps, self.n_features
        ):
            raise ValueError(
                f"values must have shape [B,{self.n_steps},{self.n_features}]"
            )
        if observed_mask.shape != values.shape:
            raise ValueError("observed_mask must have the same shape as values")
        if not values.is_floating_point():
            raise TypeError("values must be floating point")
        if observed_mask.dtype != torch.bool and not observed_mask.is_floating_point():
            raise TypeError("observed_mask must be boolean or floating point")
        if observed_mask.device != values.device:
            raise ValueError("values and observed_mask must be on the same device")
        mask = observed_mask.to(dtype=values.dtype)
        # Missing finite placeholders cannot carry hidden clean labels through
        # the embedding even if a caller accidentally supplies a nonzero value.
        filled = values * mask
        hidden = self.embedding_first(torch.cat((filled, mask), dim=-1))
        hidden = self.first_embedding_dropout(
            hidden + self.position_encoding.to(dtype=hidden.dtype)
        )
        for layer in self.first_block:
            hidden, _ = layer(hidden, self.allowed_attention)
        first = self.reduce_first(hidden)
        completed_first = mask * filled + (1.0 - mask) * first

        hidden = self.embedding_second(torch.cat((completed_first, mask), dim=-1))
        # Upstream SAITS does not add another embedding dropout at stage two.
        hidden = hidden + self.position_encoding.to(dtype=hidden.dtype)
        final_attention = None
        for index, layer in enumerate(self.second_block):
            hidden, weights = layer(
                hidden,
                self.allowed_attention,
                need_weights=(index == self.n_layers - 1),
            )
            if weights is not None:
                final_attention = weights.mean(dim=1)
        second = self.reduce_second_gamma(
            torch.nn.functional.relu(self.reduce_second_beta(hidden))
        )
        # Only the endpoint is supervised. Computing the gate for that row is
        # exactly equivalent to slicing the full SAITS mixture afterward.
        mixing = torch.sigmoid(self.combination_gate(torch.cat(
            (mask[:, -1], final_attention[:, -1]), dim=-1)))[:, :self.n_targets]
        first_endpoint = first[:, -1, :self.n_targets]
        second_endpoint = second[:, -1, :self.n_targets]
        third = (1.0 - mixing) * second_endpoint + mixing * first_endpoint
        result = {
            "prediction": third,
            "stage1": first_endpoint,
            "stage2": second_endpoint,
        }
        if return_attention:
            result["attention"] = final_attention
        return result
