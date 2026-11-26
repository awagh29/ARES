"""Controllers ("systems under test") the Gazebo runner can fly.

* ``map-temporal-cnn`` - Atharva's map-temporal sparsity-invariant CNN with the
  original VFH/PID stack (default; guidance runs in MATLAB via the bridge).
* ``map-correction``   - Atharva's barrier/map-correction controller (Python port).
"""

from .base import Command, Controller, VehicleState

CONTROLLERS = ("map-temporal-cnn", "map-correction")


def make_controller(name: str, **kwargs) -> Controller:
    if name == "map-temporal-cnn":
        from .map_temporal import MapTemporalController
        return MapTemporalController(**kwargs)
    if name == "map-correction":
        from .map_correction import MapCorrectionAdapter
        return MapCorrectionAdapter(**kwargs)
    raise ValueError(f"Unknown controller {name!r}; choose one of {CONTROLLERS}")


__all__ = ["CONTROLLERS", "Command", "Controller", "VehicleState", "make_controller"]
