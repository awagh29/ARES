"""Trajectory-level augmented SAITS and differentiable LiDAR control training.

This package is independent of the earlier single-episode trainer and its
deployment artifacts. Simulator geometry belongs exclusively to supervision
and evaluation; the online reconstructor and barrier filter do not receive it.
"""
