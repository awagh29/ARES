"""Inference-only definitions from the sparse single-episode training notebook.

The exporter verifies these definitions against the notebook AST before exporting.
No notebook cells, training code or saved outputs are executed at inference time.
"""


import copy
import torch
from torch import nn
from torch.nn import functional as F


def encode_sensor_state(raw_states, use_position):
    """[..., 9] raw (north, east, down, vn, ve, vd, roll, pitch, yaw) -> 12 or 9 features."""
    if raw_states.ndim < 1 or raw_states.shape[-1] != 9:
        raise ValueError(
            "Expected raw sensor states with last dimension 9."
        )

    raw_states = raw_states.float()
    angles = raw_states[..., 6:9]

    angle_features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    angle_features = angle_features.flatten(start_dim=-2)

    linear = raw_states[..., :6] if use_position else raw_states[..., 3:6]
    return torch.cat((linear, angle_features), dim=-1)


def pad_ring(tensor, pad_elevation, pad_azimuth):
    """Circular padding in azimuth (scan wraps at +/-179 deg), zero padding in elevation."""
    tensor = F.pad(tensor, (pad_azimuth, pad_azimuth, 0, 0), mode="circular")
    return F.pad(tensor, (0, 0, pad_elevation, pad_elevation), mode="constant", value=0.0)


class SparseConv2d(nn.Module):
    """Sparsity-invariant convolution: y = W*(x.m) / (sum_window m + eps) + b, m_out = maxpool(m)."""
    def __init__(self, in_channels, out_channels, kernel, stride=(1, 1)):
        super().__init__()
        self.kernel, self.stride = tuple(kernel), tuple(stride)
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel, self.stride,
                              padding=0, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("ones", torch.ones(1, 1, *self.kernel), persistent=False)
        self.pool = nn.MaxPool2d(self.kernel, self.stride)

    def forward(self, x, mask):
        pad_el, pad_az = self.kernel[0] // 2, self.kernel[1] // 2
        masked, padded_mask = pad_ring(x * mask, pad_el, pad_az), pad_ring(mask, pad_el, pad_az)
        y = self.conv(masked)
        support = F.conv2d(padded_mask, self.ones.to(padded_mask.dtype), stride=self.stride)
        y = y / support.clamp_min(1e-6) + self.bias.view(1, -1, 1, 1)
        mask_out = self.pool(padded_mask)
        return F.relu(y) * mask_out, mask_out


class RingConv2d(nn.Module):
    """Dense convolution with the same ring padding; the decoder uses it to fill holes."""
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.kernel = tuple(kernel)
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel)

    def forward(self, x):
        return self.conv(pad_ring(x, self.kernel[0] // 2, self.kernel[1] // 2))


class SparseRangeReconstructor(nn.Module):
    """Masked [batch, 15, 716] scan + encoded state -> dense [batch, 15, 716] ranges."""
    def __init__(self, model_config):
        super().__init__()
        self.model_config = copy.deepcopy(model_config)
        c = self.model_config
        self.max_range = float(c["max_range"])
        widths = list(c["widths"])
        kernel = (3, c["kernel"])

        ############################################################################
        # Sparse encoder: two sparse convolutions per level; levels after the first
        # halve azimuth with stride (1, 2).
        self.encoder = nn.ModuleList()
        in_channels = 2                              # nearness, hit indicator
        for level, width in enumerate(widths):
            stride = (1, 1) if level == 0 else (1, 2)
            self.encoder.append(nn.ModuleList([
                SparseConv2d(in_channels, width, kernel, stride),
                SparseConv2d(width, width, kernel),
            ]))
            in_channels = width

        ############################################################################
        # FiLM state conditioning of the bottleneck, initialized as identity.
        self.film = nn.Sequential(
            nn.Linear(c["input_features"], c["state_hidden"]), nn.ReLU(),
            nn.Linear(c["state_hidden"], 2 * widths[-1]),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        ############################################################################
        # Dense decoder: upsample x2, concatenate skip features and support masks.
        self.decoder = nn.ModuleList()
        for level in range(len(widths) - 1, 0, -1):
            in_channels = widths[level] + 1 + widths[level - 1] + 1
            self.decoder.append(nn.Sequential(
                RingConv2d(in_channels, widths[level - 1], kernel), nn.ReLU(),
                RingConv2d(widths[level - 1], widths[level - 1], kernel), nn.ReLU(),
            ))
        # Head sees decoder features, their support mask and the 3 raw input channels.
        self.head = nn.Conv2d(widths[0] + 1 + 3, 1, kernel_size=1)

    def forward(self, ranges, valid, available, state):
        """ranges [B,15,716] m; valid/available [B,15,716] bool; state [B, input_features]."""
        support = available.float().unsqueeze(1)
        hit = (valid & available).float().unsqueeze(1)
        nearness = (1.0 - ranges.float().unsqueeze(1) / self.max_range).clamp(0, 1) * hit
        inputs = torch.cat((nearness, hit, support), dim=1)

        x, mask = inputs[:, :2], support
        skips = []
        for first, second in self.encoder:
            x, mask = first(x, mask)
            x, mask = second(x, mask)
            skips.append((x, mask))

        gamma, beta = self.film(state)[..., None, None].chunk(2, dim=1)
        x = x * (1 + gamma) + beta

        for block, (skip_x, skip_mask) in zip(self.decoder, reversed(skips[:-1])):
            x = torch.cat((x, mask), dim=1).repeat_interleave(2, dim=-1)
            x = block(torch.cat((x, skip_x, skip_mask), dim=1))
            mask = skip_mask

        features = torch.cat((x, mask, inputs), dim=1)
        with torch.autocast(features.device.type, enabled=False):
            logits = self.head(features.float())
        return self.max_range * torch.sigmoid(logits.squeeze(1))
