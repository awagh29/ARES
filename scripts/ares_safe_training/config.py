"""Versioned, explicit settings for the two training stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from .models import ModelConfig


@dataclass
class PipelineConfig:
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    calibration_fraction_of_train: float = 0.20
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain_epochs: int = 120
    pretrain_batch_size: int = 128
    pretrain_learning_rate: float = 1e-3
    finetune_epochs: int = 30
    rollout_batch_size: int = 8
    rollout_steps: int = 200
    evaluation_steps: int = 600
    gradient_horizon: int = 10
    control_dt: float = 0.1
    plant_dt: float = 0.001
    finetune_learning_rate: float = 1e-4
    barrier_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 20
    clip_grad_norm: float = 1.0
    observed_reconstruction_weight: float = 0.1
    auxiliary_reconstruction_weight: float = 0.1
    tracking_weight: float = 0.1
    control_correction_weight: float = 0.05
    barrier_violation_weight: float = 1.0
    geometry_violation_weight: float = 5.0
    tracking_slack_weight: float = 0.01
    reconstruction_weight: float = 1.0
    uncertainty_quantile: float = 1.0
    uncertainty_margin_m: float = 0.10
    minimum_range_bound_m: float = 0.05
    reference_speed_mps: float = 2.5
    tracking_tube_radius_m: float = 2.5
    maximum_yaw_rate_radps: float = 0.5
    nominal_position_gain: float = 2.0
    nominal_velocity_gain: float = 2.5
    nominal_repulsion_gain: float = 0.2
    compile_dynamics: bool = True
    checkpoint_every: int = 1
    log_every: int = 1

    def validate(self):
        self.model.validate()
        for name, value in vars(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        if not 0 < self.train_fraction < 1 or not 0 < self.val_fraction < 1 - self.train_fraction:
            raise ValueError("Invalid three-way trajectory split.")
        if not 0 < self.calibration_fraction_of_train < .5:
            raise ValueError("Reserve a nonempty subset inside training for empirical error calibration.")
        for name in ("pretrain_epochs", "pretrain_batch_size", "finetune_epochs", "rollout_batch_size",
                     "rollout_steps", "evaluation_steps", "gradient_horizon", "patience", "checkpoint_every", "log_every"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.gradient_horizon > self.rollout_steps:
            raise ValueError("gradient_horizon cannot exceed rollout_steps.")
        if abs(self.control_dt - .1) > 1e-12 or not 0 < self.plant_dt <= .001:
            raise ValueError("The validated sensor/controller interval is .1 s; plant step must be <=.001 s.")
        if not 0 < self.uncertainty_quantile <= 1:
            raise ValueError("Invalid empirical uncertainty quantile.")
        for name in ("pretrain_learning_rate", "finetune_learning_rate", "barrier_learning_rate",
                     "clip_grad_norm", "reference_speed_mps", "tracking_tube_radius_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("weight_decay", "observed_reconstruction_weight", "auxiliary_reconstruction_weight",
                     "tracking_weight", "control_correction_weight", "barrier_violation_weight",
                     "geometry_violation_weight", "tracking_slack_weight", "reconstruction_weight",
                     "uncertainty_margin_m", "minimum_range_bound_m", "nominal_position_gain",
                     "nominal_velocity_gain", "nominal_repulsion_gain"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if not 0 < self.maximum_yaw_rate_radps <= .5:
            raise ValueError("Yaw-rate policy must stay within its calibrated .5 rad/s limit.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["model"] = ModelConfig(**value.get("model", {}))
        result = cls(**value)
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_default_config(path):
    Path(path).write_text(json.dumps(PipelineConfig().to_dict(), indent=2) + "\n", encoding="utf-8")
