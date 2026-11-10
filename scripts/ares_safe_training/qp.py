"""Batched, differentiable strictly convex inequality QPs in native PyTorch.

An infeasible-start Mehrotra primal-dual interior-point method solves the
actual QP. Backpropagation solves the active-constraint KKT adjoint rather than
unrolling iterations or substituting a penalty/projection heuristic. FP64 is
intentional; the solver disables autocast. The selected active set gives the
usual almost-everywhere derivative; at degenerate boundaries a minimum-norm
KKT adjoint supplies a generalized derivative.
"""

from __future__ import annotations

import torch
from torch import nn


def _mv(matrix, vector):
    return torch.bmm(matrix, vector.unsqueeze(-1)).squeeze(-1)


def _step_length(value, direction, fraction=1.0):
    ratio = torch.where(direction < 0, -value / direction.clamp_max(-1e-300),
                        torch.full_like(value, float("inf")))
    return torch.minimum(torch.ones_like(value[:, 0]), fraction * ratio.amin(dim=1))


def _newton(Q, G, slack, dual, rd, rp, centering):
    ratio = (dual / slack).clamp(1e-25, 1e25)
    rhs_dual = (-centering + dual * rp) / slack
    matrix = Q + torch.bmm(G.transpose(1, 2), ratio.unsqueeze(-1) * G)
    rhs = -rd - _mv(G.transpose(1, 2), rhs_dual)
    # Regularize a Newton linear solve, not the problem objective/constraints.
    # Residual validation below still uses the original QP coefficients.
    eye = torch.eye(Q.shape[-1], dtype=Q.dtype, device=Q.device).unsqueeze(0)
    regularized = matrix + 1e-12 * eye
    dx, info = torch.linalg.solve_ex(regularized, rhs.unsqueeze(-1), check_errors=False)
    correction_rhs = rhs.unsqueeze(-1) - torch.bmm(matrix, dx)
    correction, correction_info = torch.linalg.solve_ex(regularized, correction_rhs, check_errors=False)
    dx = dx + correction
    info = torch.maximum(info, correction_info)
    dx = dx.squeeze(-1)
    ds = -rp - _mv(G, dx)
    dl = (-centering - dual * ds) / slack
    good = (info == 0) & torch.isfinite(dx).all(dim=1)
    good &= torch.isfinite(ds).all(dim=1) & torch.isfinite(dl).all(dim=1)
    return dx, ds, dl, good


def _pdipm(Q, p, G, h, max_iter, tolerance):
    batch, constraints, variables = G.shape
    initial_x, initial_info = torch.linalg.solve_ex(Q, -p.unsqueeze(-1), check_errors=False)
    x = initial_x.squeeze(-1)
    gap = h - _mv(G, x)
    shift = (1 - gap.amin(dim=1)).clamp_min(0)
    slack = gap + shift[:, None]
    dual = torch.ones_like(slack)
    best_x, best_s, best_l = x.clone(), slack.clone(), dual.clone()
    best_score = torch.full((batch,), float("inf"), dtype=Q.dtype, device=Q.device)
    failed = (initial_info != 0) | ~torch.isfinite(x).all(1)
    iterations = torch.zeros(batch, dtype=torch.int64, device=Q.device)
    for iteration in range(max_iter):
        rd = _mv(Q, x) + p + _mv(G.transpose(1, 2), dual)
        rp = _mv(G, x) + slack - h
        mu = (slack * dual).mean(dim=1)
        score = torch.maximum(rd.abs().amax(dim=1), rp.abs().amax(dim=1))
        score = torch.maximum(score, mu)
        better = (score < best_score) & torch.isfinite(score) & ~failed
        best_x = torch.where(better[:, None], x, best_x)
        best_s = torch.where(better[:, None], slack, best_s)
        best_l = torch.where(better[:, None], dual, best_l)
        best_score = torch.where(better, score, best_score)
        done = best_score <= tolerance
        active = ~done & ~failed
        iterations = torch.where(active, torch.full_like(iterations, iteration + 1), iterations)
        # One synchronization per five iterations, never a per-example CPU QP.
        if iteration % 5 == 0 and not bool(active.any().item()):
            break
        affine_x, affine_s, affine_l, affine_ok = _newton(
            Q, G, slack, dual, rd, rp, slack * dual)
        primal_step = _step_length(slack, affine_s)
        dual_step = _step_length(dual, affine_l)
        affine_mu = ((slack + primal_step[:, None] * affine_s)
                     * (dual + dual_step[:, None] * affine_l)).mean(dim=1)
        sigma = (affine_mu / mu.clamp_min(1e-300)).clamp(0, 1).pow(3)
        center = slack * dual + affine_s * affine_l - sigma[:, None] * mu[:, None]
        dx, ds, dl, corrector_ok = _newton(Q, G, slack, dual, rd, rp, center)
        good = affine_ok & corrector_ok
        failed |= active & ~good
        update = active & good
        ap = _step_length(slack, ds, 0.995)
        ad = _step_length(dual, dl, 0.995)
        x = torch.where(update[:, None], x + ap[:, None] * dx, x)
        slack = torch.where(update[:, None], (slack + ap[:, None] * ds).clamp_min(1e-25), slack)
        dual = torch.where(update[:, None], (dual + ad[:, None] * dl).clamp_min(1e-25), dual)
    # Include the final iterate, including short max_iter test configurations.
    rd = _mv(Q, x) + p + _mv(G.transpose(1, 2), dual)
    rp = _mv(G, x) + slack - h
    score = torch.maximum(torch.maximum(rd.abs().amax(1), rp.abs().amax(1)),
                          (slack * dual).mean(1))
    better = torch.isfinite(score) & (score < best_score) & ~failed
    best_x = torch.where(better[:, None], x, best_x)
    best_s = torch.where(better[:, None], slack, best_s)
    best_l = torch.where(better[:, None], dual, best_l)
    best_score = torch.where(better, score, best_score)
    converged = best_score <= tolerance
    return best_x, best_s, best_l, converged, iterations


class _ImplicitQP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, p, G, h, max_iter, tolerance, active_tolerance):
        x, slack, dual, converged, iterations = _pdipm(Q, p, G, h, max_iter, tolerance)
        ctx.save_for_backward(Q, G, h, x, slack, dual, converged)
        ctx.active_tolerance = active_tolerance
        ctx.mark_non_differentiable(slack, dual, converged, iterations)
        return x, slack, dual, converged, iterations

    @staticmethod
    def backward(ctx, grad_x, _grad_slack, _grad_dual, _grad_converged, _grad_iterations):
        Q, G, h, x, slack, dual, converged = ctx.saved_tensors
        batch, constraints, variables = G.shape
        active = (h - _mv(G, x) <= ctx.active_tolerance) & (dual > 1e-7)
        active &= converged[:, None]
        A = G * active.unsqueeze(-1)
        size = variables + constraints
        kkt = Q.new_zeros(batch, size, size)
        kkt[:, :variables, :variables] = Q
        kkt[:, :variables, variables:] = A.transpose(1, 2)
        kkt[:, variables:, :variables] = A
        kkt[:, variables:, variables:] = -torch.diag_embed((~active).to(Q.dtype))
        rhs = torch.cat((grad_x * converged[:, None], Q.new_zeros(batch, constraints)), dim=1)
        # Pseudoinverse handles redundant active inequalities (e.g. cone apex).
        # This remains a batched CUDA operation and is only in the backward pass.
        adjoint = _mv(torch.linalg.pinv(kkt, rtol=1e-10), rhs)
        v, w = adjoint[:, :variables], adjoint[:, variables:]
        w = w * active
        active_dual = dual * active
        dQ = -0.5 * (v.unsqueeze(2) * x.unsqueeze(1) + x.unsqueeze(2) * v.unsqueeze(1))
        dp = -v
        dG = -(active_dual.unsqueeze(2) * v.unsqueeze(1)
               + w.unsqueeze(2) * x.unsqueeze(1))
        dh = w
        return dQ, dp, dG, dh, None, None, None


class DifferentiableQP(nn.Module):
    """Minimize .5*x'Q*x+p'x subject to G*x<=h, with Q strictly SPD.

    Coefficients may be batched or shared. All numerical work and returned
    solutions are FP64; casts retain upstream gradients. ``feasible`` reports
    actual primal feasibility and ``converged`` additionally requires KKT
    residual/complementarity convergence. No infeasible result is post-clipped.
    """

    def __init__(self, max_iter=70, tolerance=1e-8, active_tolerance=1e-6):
        super().__init__()
        if max_iter < 1 or tolerance <= 0 or active_tolerance <= 0:
            raise ValueError("Invalid QP iteration/tolerance settings")
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.active_tolerance = float(active_tolerance)

    def forward(self, Q, p, G, h):
        if p.ndim not in (1, 2) or G.ndim not in (2, 3):
            raise ValueError("Expected p[n] or [B,n], G[m,n] or [B,m,n]")
        batch = max(p.shape[0] if p.ndim == 2 else 1,
                    G.shape[0] if G.ndim == 3 else 1,
                    h.shape[0] if h.ndim == 2 else 1,
                    Q.shape[0] if Q.ndim == 3 else 1)
        device = p.device
        values = []
        for value, rank in ((Q, 3), (p, 2), (G, 3), (h, 2)):
            if value.device != device:
                raise ValueError("All QP coefficients must be on the same device")
            if value.ndim == rank - 1:
                value = value.unsqueeze(0)
            if value.ndim != rank or value.shape[0] not in (1, batch):
                raise ValueError("Incompatible QP batch dimensions")
            values.append(value.to(dtype=torch.float64).expand(batch, *value.shape[1:]))
        Q, p, G, h = values
        n, m = p.shape[1], h.shape[1]
        if Q.shape[1:] != (n, n) or G.shape[1:] != (m, n) or m < 1:
            raise ValueError("Incompatible QP matrix dimensions")
        finite = torch.stack([torch.isfinite(value).all() for value in values]).all()
        symmetric = torch.isclose(Q, Q.transpose(1, 2), atol=1e-10, rtol=1e-10).all()
        positive_definite = (torch.linalg.cholesky_ex(Q, check_errors=False).info == 0).all()
        # Consolidate diagnostics into one device-to-host transfer. In
        # particular, torch.allclose / torch.linalg.solve would each introduce
        # an additional implicit CUDA synchronization in this scan-rate path.
        valid_finite, valid_symmetric, valid_spd = torch.stack(
            (finite, symmetric, positive_definite)).detach().cpu().tolist()
        if not valid_finite:
            raise ValueError("QP coefficients must be finite")
        if not valid_symmetric:
            raise ValueError("QP Hessian must be symmetric")
        if not valid_spd:
            raise ValueError("QP Hessian must be strictly positive definite")
        with torch.autocast(device_type=device.type, enabled=False):
            x, slack, dual, converged, iterations = _ImplicitQP.apply(
                Q, p, G, h, self.max_iter, self.tolerance, self.active_tolerance)
            inequality = _mv(G, x) - h
            violation = inequality.clamp_min(0).amax(dim=1)
            finite = torch.isfinite(x).all(dim=1)
            feasible = finite & (violation <= self.tolerance)
            dual_residual = (_mv(Q, x) + p + _mv(G.transpose(1, 2), dual)).abs().amax(1)
            return {"solution": x, "feasible": feasible, "converged": converged,
                    "max_violation": violation, "inequality_violation": inequality.clamp_min(0),
                    "dual_residual": dual_residual, "complementarity": (slack * dual).mean(1),
                    "slack": slack, "dual": dual, "iterations": iterations}


def solve_qp(Q, p, G, h, **settings):
    return DifferentiableQP(**settings)(Q, p, G, h)
