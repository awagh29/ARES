"""Analytic physical LiDAR and collision truth at newly rolled states.

All map access stays in this simulator object. An online policy/barrier must
receive reconstructed observations and sensed motion, never these box buffers.
Distances are piecewise differentiable; hit masks and return fractions change
discretely at visibility boundaries. Computation explicitly stays float32.
"""
from __future__ import annotations

import numpy as np
import torch


class TorchLidarScene:
    def __init__(self, scene, device="cpu", *, batch_chunk=16, ray_chunk=512):
        self.device = torch.device(device)
        self.scene_metadata = dict(scene)
        self.batch_chunk, self.ray_chunk = int(batch_chunk), int(ray_chunk)
        if self.batch_chunk <= 0 or self.ray_chunk <= 0:
            raise ValueError("Ray and batch chunk sizes must be positive.")
        minimum = np.asarray(scene["box_min_ned_m"], dtype=np.float32)
        maximum = np.asarray(scene["box_max_ned_m"], dtype=np.float32)
        if minimum.ndim != 2 or minimum.shape[1] != 3 or minimum.shape != maximum.shape or not len(minimum):
            raise ValueError("Scene requires nonempty matching [boxes,3] bounds.")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or not (maximum > minimum).all():
            raise ValueError("Physical box bounds must be finite and have positive extent.")
        self.box_min = torch.as_tensor(minimum, device=self.device)
        self.box_max = torch.as_tensor(maximum, device=self.device)
        self.device = self.box_min.device
        self.mount = torch.as_tensor(scene["mounting_translation_body_m"], dtype=torch.float32, device=self.device)
        mount_rotation = np.asarray(scene["sensor_to_body_rotation"], dtype=np.float64)
        if mount_rotation.shape != (3, 3) or not np.allclose(mount_rotation.T @ mount_rotation, np.eye(3), atol=1e-8):
            raise ValueError("Sensor mounting rotation must be orthogonal.")
        self.max_range = float(scene["maximum_range_m"])
        if not self.max_range > 0 or self.mount.shape != (3,):
            raise ValueError("Invalid physical sensor range or mounting translation.")
        az = np.linspace(*scene["azimuth_limits_deg"], int(scene["azimuth_count"]))
        el = np.linspace(*scene["elevation_limits_deg"], int(scene["elevation_count"]))
        edges = np.asarray(scene["sector_edges_body_deg"], dtype=np.float64)
        if edges.shape != (8,) or not np.all(np.diff(edges) > 0):
            raise ValueError("Expected seven ordered body-forward sectors.")
        azimuth, elevation = np.meshgrid(np.deg2rad(az), np.deg2rad(el))
        sensor_rays = np.stack((np.cos(elevation)*np.cos(azimuth),
                               np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), axis=-1)
        sensor_rays = sensor_rays.reshape(-1, 3, order="F")
        body_rays = sensor_rays @ mount_rotation.T
        body_bearing = np.rad2deg(np.arctan2(body_rays[:, 1], body_rays[:, 0]))
        forward = (body_bearing >= edges[0]) & (body_bearing <= edges[-1])
        body_rays, body_bearing = body_rays[forward], body_bearing[forward]
        sectors = np.minimum(np.searchsorted(edges, body_bearing, side="right") - 1, 6)
        counts = np.bincount(sectors, minlength=7)
        if not (counts > 0).all():
            raise ValueError("At least one physical sector contains no rays.")
        self.rays_body = torch.as_tensor(body_rays.copy(), dtype=torch.float32, device=self.device)
        self.sector_index = torch.as_tensor(sectors.copy(), dtype=torch.long, device=self.device)
        self.sector_beam_counts = torch.as_tensor(counts, dtype=torch.float32, device=self.device)
        self._chunks = []
        for start in range(0, len(sectors), self.ray_chunk):
            local = sectors[start:start+self.ray_chunk]
            indices = tuple(torch.as_tensor(np.flatnonzero(local == sector), dtype=torch.long,
                                            device=self.device) for sector in range(7))
            self._chunks.append((start, indices))

    @staticmethod
    def _rotation(rpy):
        roll, pitch, yaw = rpy.unbind(-1)
        sr, cr = roll.sin(), roll.cos()
        sp, cp = pitch.sin(), pitch.cos()
        sy, cy = yaw.sin(), yaw.cos()
        return torch.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                            sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                            -sp, cp*sr, cp*cr), dim=-1).reshape(-1, 3, 3)

    def _validate_state(self, state):
        if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 13:
            raise ValueError("State must be a [B,13] tensor [p,v,rpy,omega,F].")
        if state.device != self.device or not torch.isfinite(state).all():
            raise ValueError("State must be finite and on the scene device.")
        return state.float()

    def _ray_chunk(self, origin, rotation, rays):
        direction = torch.einsum("bij,rj->bri", rotation, rays)
        parallel = direction.abs() < 1e-8
        divisor = torch.where(parallel, torch.ones_like(direction), direction)
        first = (self.box_min[None, None] - origin[:, None, None]) / divisor[:, :, None]
        second = (self.box_max[None, None] - origin[:, None, None]) / divisor[:, :, None]
        near = torch.where(parallel[:, :, None], -torch.inf, torch.minimum(first, second)).amax(-1)
        far = torch.where(parallel[:, :, None], torch.inf, torch.maximum(first, second)).amin(-1)
        outside_parallel = parallel[:, :, None] & ((origin[:, None, None] < self.box_min[None, None])
                                                   | (origin[:, None, None] > self.box_max[None, None]))
        intersects = (far >= torch.maximum(near, torch.zeros_like(near))) & ~outside_parallel.any(-1)
        distance = torch.where(near >= 0, near, far)
        distance = torch.where(intersects & (distance > 0), distance, torch.inf).amin(-1)
        valid = torch.isfinite(distance) & (distance > 0) & (distance <= self.max_range + 1e-6*self.max_range)
        ranges = torch.where(valid, distance.clamp(max=self.max_range), self.max_range)
        return ranges, valid

    def features(self, state):
        """Return [B,14] clean physical features at CURRENT rolled poses.

        The last state element F, velocity and omega do not enter optical
        geometry. Position/RPY gradients flow through valid hit distances.
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            state = self._validate_state(state)
            outputs = []
            for start in range(0, len(state), self.batch_chunk):
                batch = state[start:start+self.batch_chunk]
                rotation = self._rotation(batch[:, 6:9])
                origin = batch[:, :3] + torch.einsum("bij,j->bi", rotation, self.mount)
                nearest = torch.full((len(batch), 7), self.max_range, device=self.device)
                hits = torch.zeros_like(nearest)
                for ray_start, sector_indices in self._chunks:
                    rays = self.rays_body[ray_start:ray_start+self.ray_chunk]
                    ranges, valid = self._ray_chunk(origin, rotation, rays)
                    minima, counts = [], []
                    for selected in sector_indices:
                        if selected.numel():
                            minima.append(ranges.index_select(-1, selected).amin(-1))
                            counts.append(valid.index_select(-1, selected).sum(-1))
                        else:
                            minima.append(torch.full((len(batch),), self.max_range, device=self.device))
                            counts.append(torch.zeros(len(batch), device=self.device))
                    nearest = torch.minimum(nearest, torch.stack(minima, dim=-1))
                    hits = hits + torch.stack(counts, dim=-1)
                outputs.append(torch.cat((nearest, hits/self.sector_beam_counts), dim=-1))
            return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 14), device=self.device)

    def clearance(self, positions, vehicle_radius=.1):
        """Signed minimum sphere-to-box clearance, simulator truth ONLY.

        Accepts [...,3] NED positions and returns [...] metres. Negative values
        denote penetration. This function must not become a policy input or an
        online QP constraint based on hidden map coordinates.
        """
        if not isinstance(positions, torch.Tensor) or positions.ndim < 1 or positions.shape[-1] != 3 or positions.device != self.device:
            raise ValueError("Positions must end in 3 NED coordinates on the scene device.")
        if not np.isfinite(vehicle_radius) or vehicle_radius < 0 or not torch.isfinite(positions).all():
            raise ValueError("Positions and nonnegative vehicle radius must be finite.")
        with torch.autocast(device_type=self.device.type, enabled=False):
            p = positions.float()
            center, half = (self.box_min+self.box_max)/2, (self.box_max-self.box_min)/2
            q = (p[..., None, :] - center).abs() - half
            outside = torch.linalg.vector_norm(q.clamp(min=0), dim=-1)
            inside = q.amax(-1).clamp(max=0)
            return (outside+inside).amin(-1) - float(vehicle_radius)
