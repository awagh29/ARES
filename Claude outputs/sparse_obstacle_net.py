"""
Baseline sparsity-invariant CNN for ARES LiDAR obstacle reconstruction.

Idea: don't densify the whole 15 x 716 sphere. Detect *which azimuth columns
hold an obstacle* and *how far it is*, then reduce to the 7 controller sectors.

Input  (per scan):  ranges (15,716) in metres, valid mask (15,716),
                    available mask (15,716)  [False = beam lost to outage]
                    state vector (S,)        [velocity, sin/cos attitude, rates, ...]
Output (per scan):  occ_logit (716,)  -> P(obstacle in this azimuth column)
                    range_m   (716,)  -> distance if occupied (0..7 m)
                    sector_min (7,)   -> soft-min range per controller sector
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

MAX_RANGE = 7.0
N_EL, N_AZ = 15, 716


# --------------------------------------------------------------------------
# 1. Sparsity-invariant convolution (Uhrig et al., 3DV 2017)
#    y = W * (x ⊙ m) / (Σ_window m + eps) + b,   m_out = maxpool(m)
#    Azimuth is padded circularly (the scan wraps at ±179°).
# --------------------------------------------------------------------------
class SparseConv2d(nn.Module):
    def __init__(self, c_in, c_out, k=(3, 3), stride=(1, 1)):
        super().__init__()
        self.k, self.stride = k, stride
        self.conv = nn.Conv2d(c_in, c_out, k, stride, padding=0, bias=False)
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.register_buffer("ones", torch.ones(1, 1, *k))
        self.pool = nn.MaxPool2d(k, stride)

    def _pad(self, t):
        ph, pw = self.k[0] // 2, self.k[1] // 2
        t = F.pad(t, (pw, pw, 0, 0), mode="circular")        # azimuth wraps
        return F.pad(t, (0, 0, ph, ph), mode="constant", value=0.0)  # elevation doesn't

    def forward(self, x, m):
        xm, mp = self._pad(x * m), self._pad(m)
        y = self.conv(xm)
        norm = F.conv2d(mp, self.ones, stride=self.stride)
        y = y / norm.clamp_min(1e-6) + self.bias.view(1, -1, 1, 1)
        m_out = self.pool(mp)
        return F.relu(y) * m_out, m_out          # zero features where still unsupported


class SparseConv1d(nn.Module):
    """Same operator on the azimuth ring after elevation is collapsed."""
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(c_in, c_out, k, bias=False)
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.register_buffer("ones", torch.ones(1, 1, k))

    def forward(self, x, m):
        p = self.k // 2
        xm = F.pad(x * m, (p, p), mode="circular")
        mp = F.pad(m, (p, p), mode="circular")
        y = self.conv(xm) / F.conv1d(mp, self.ones).clamp_min(1e-6)
        y = y + self.bias.view(1, -1, 1)
        m_out = F.max_pool1d(mp, self.k, 1)
        return F.relu(y) * m_out, m_out


# --------------------------------------------------------------------------
# 2. Network
# --------------------------------------------------------------------------
class SparseObstacleNet(nn.Module):
    def __init__(self, state_dim=12, width=32, sector_edges_deg=None):
        super().__init__()
        # Stage A: 2-D sparse encoder over (elevation, azimuth), azimuth stride 2
        self.enc2d = nn.ModuleList([
            SparseConv2d(1, width, (3, 11)),
            SparseConv2d(width, width, (3, 7), stride=(1, 2)),   # 716 -> 358
            SparseConv2d(width, width, (3, 5)),
        ])
        # Stage B: 1-D sparse ring encoder over azimuth (obstacles are vertical columns)
        self.enc1d = nn.ModuleList([
            SparseConv1d(width + 1, 2 * width, 7),   # +1: no-return evidence channel
            SparseConv1d(2 * width, 2 * width, 7),
            SparseConv1d(2 * width, 2 * width, 7),
        ])
        # Dynamics conditioning (FiLM): lets attitude/velocity *shift and scale*
        # features, e.g. to propagate a known obstacle through an outage.
        self.film = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 4 * width))
        # Dense 1-D decoder fills the outage holes, then upsamples back to 716
        self.dec = nn.Sequential(
            nn.Conv1d(2 * width + 1, 2 * width, 9, padding=4, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(2 * width, 2 * width, 9, padding=4, padding_mode="circular"), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(2 * width, width, 5, padding=2, padding_mode="circular"), nn.ReLU(),
        )
        self.head = nn.Conv1d(width, 3, 1)    # occ logit, range, log-scale

        # Sector membership (body bearing = −sensor azimuth, per the barrier doc)
        if sector_edges_deg is None:
            sector_edges_deg = torch.linspace(-60, 60, 8)
        az = torch.linspace(-179, 179, N_AZ)
        bearing = -az
        member = torch.stack([(bearing >= lo) & (bearing < hi)
                              for lo, hi in zip(sector_edges_deg[:-1], sector_edges_deg[1:])])
        self.register_buffer("sector_member", member.float())   # (7, 716)

    def forward(self, ranges, valid, available, state, tau=0.1):
        """
        ranges, valid, available: (B, 15, 716);  state: (B, S)
        valid     = beam returned a hit
        available = beam was measured at all (False during an outage)
        """
        B = ranges.shape[0]
        r = (ranges / MAX_RANGE).unsqueeze(1)
        hit = (valid & available).float().unsqueeze(1)          # sparse support
        free = ((~valid) & available).float().unsqueeze(1)       # measured free space

        # Stage A: features only from real hits
        x, m = 1.0 - r, hit                                      # "nearness" is 0 on free beams
        for layer in self.enc2d:
            x, m = layer(x, m)

        # Collapse elevation with mask-weighted mean -> azimuth ring
        x = (x * m).sum(2) / m.sum(2).clamp_min(1e-6)            # (B, C, 358)
        m = m.amax(2)                                            # (B, 1, 358)

        # Free-space evidence on the ring: fraction of measured-and-free beams
        free_ring = F.avg_pool1d(free.mean(2), 2)                # (B, 1, 358)
        meas_ring = F.avg_pool1d(available.float().unsqueeze(1).mean(2), 2)
        x = torch.cat([x, free_ring], 1)
        m = torch.maximum(m, (meas_ring > 0).float())            # free beams count as support

        # Stage B
        gamma, beta = self.film(state).view(B, 2, -1, 1).unbind(1)
        for layer in self.enc1d:
            x, m = layer(x, m)
        x = x * (1 + gamma) + beta

        # Decoder sees the support mask so it knows what it is inventing
        out = self.head(self.dec(torch.cat([x, m], 1)))          # (B, 3, 716)
        occ_logit = out[:, 0]
        range_m = MAX_RANGE * torch.sigmoid(out[:, 1])
        log_scale = out[:, 2]

        # Sector soft-min: free columns count as 7 m
        p = torch.sigmoid(occ_logit)
        eff = p * range_m + (1 - p) * MAX_RANGE                  # (B, 716)
        w = torch.softmax(-eff.unsqueeze(1) / tau + torch.log(self.sector_member + 1e-9), -1)
        sector_min = (w * eff.unsqueeze(1)).sum(-1)              # (B, 7)
        return dict(occ_logit=occ_logit, range_m=range_m,
                    log_scale=log_scale, sector_min=sector_min)


# --------------------------------------------------------------------------
# 3. Targets and loss (computed from the CLEAN scan, inputs are perturbed)
# --------------------------------------------------------------------------
def column_targets(clean_ranges, clean_valid):
    """occupied column = any hit across elevation; range = min hit range."""
    occ = clean_valid.any(1).float()                              # (B, 716)
    r = torch.where(clean_valid, clean_ranges, torch.full_like(clean_ranges, MAX_RANGE))
    return occ, r.amin(1)


def sector_targets(col_range, sector_member):
    big = torch.where(sector_member.bool().unsqueeze(0),
                      col_range.unsqueeze(1), torch.full_like(col_range.unsqueeze(1), MAX_RANGE))
    return big.amin(-1)                                           # (B, 7)


def loss_fn(out, occ_t, range_t, sector_t, over_weight=4.0, pos_weight=5.0):
    # (a) where are the obstacles?
    l_occ = F.binary_cross_entropy_with_logits(
        out["occ_logit"], occ_t, pos_weight=torch.tensor(pos_weight, device=occ_t.device))
    # (b) how far, only on occupied columns; asymmetric: overestimating distance is worse
    err = out["range_m"] - range_t
    asym = torch.where(err > 0, over_weight * err.abs(), err.abs())
    scale = out["log_scale"].exp()
    l_rng = ((asym / scale + out["log_scale"]) * occ_t).sum() / occ_t.sum().clamp_min(1)
    # (c) what the controller actually consumes
    e_s = out["sector_min"] - sector_t
    l_sec = torch.where(e_s > 0, over_weight * e_s.abs(), e_s.abs()).mean()
    return l_occ + l_rng + l_sec, dict(occ=l_occ.item(), rng=l_rng.item(), sec=l_sec.item())


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    net = SparseObstacleNet(state_dim=12)
    ranges = torch.full((B, N_EL, N_AZ), MAX_RANGE)
    valid = torch.zeros(B, N_EL, N_AZ, dtype=torch.bool)
    valid[:, :, 300:360] = True; ranges[:, :, 300:360] = 3.0     # fake pillar
    available = torch.ones_like(valid); available[:, :, 320:400] = False  # fake outage
    state = torch.randn(B, 12)

    out = net(ranges, valid, available, state)
    occ_t, rng_t = column_targets(ranges, valid)
    sec_t = sector_targets(rng_t, net.sector_member)
    loss, parts = loss_fn(out, occ_t, rng_t, sec_t)
    loss.backward()
    print({k: tuple(v.shape) for k, v in out.items()})
    print("loss parts:", parts)
    print("params:", sum(p.numel() for p in net.parameters()))
