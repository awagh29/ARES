"""Load the scenario matrix (``config/single_drone_scenarios.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .wind import WindProfile

DEFAULT_MATRIX = Path(__file__).resolve().parents[1] / "config" / "single_drone_scenarios.yaml"


@dataclass(frozen=True)
class Scenario:
    id: str
    lidar: str
    wind: WindProfile
    note: str = ""


@dataclass(frozen=True)
class Start:
    id: str
    north: float
    east: float
    note: str = ""

    def state(self, altitude_down: float) -> list[float]:
        return [float(self.north), float(self.east), float(altitude_down), 0.0, 0.0, 0.0]


@dataclass(frozen=True)
class Matrix:
    campaign: str
    mission: dict
    seeds: list
    scenarios: list
    controller: dict
    starts: list

    def select_starts(self, ids: list[str] | None) -> list[Start]:
        if not ids:
            return list(self.starts)
        chosen = [start for start in self.starts if start.id in ids]
        missing = set(ids) - {start.id for start in chosen}
        if missing:
            raise SystemExit(f"Unknown start(s) {sorted(missing)}; known: {[s.id for s in self.starts]}")
        return chosen

    def select(self, ids: list[str] | None) -> list[Scenario]:
        if not ids:
            return list(self.scenarios)
        chosen = []
        for wanted in ids:
            matches = [scenario for scenario in self.scenarios if scenario.id == wanted or scenario.id.startswith(wanted)]
            if not matches:
                raise SystemExit(f"Unknown scenario {wanted!r}. Use --list to see the ids.")
            chosen.extend(match for match in matches if match not in chosen)
        return chosen


def load_matrix(path: Path = DEFAULT_MATRIX) -> Matrix:
    data = yaml.safe_load(Path(path).read_text())
    profiles = {name: replace(WindProfile(), name=name, **values) for name, values in data["wind_profiles"].items()}
    scenarios = []
    for item in data["scenarios"]:
        if item["wind"] not in profiles:
            raise ValueError(f"Scenario {item['id']} uses unknown wind profile {item['wind']}")
        scenarios.append(Scenario(id=item["id"], lidar=item["lidar"], wind=profiles[item["wind"]],
                                  note=item.get("note", "")))
    controller = {"name": "map-temporal-cnn", "model_path": "", "export_python": ""}
    controller.update(data.get("controller") or {})
    mission = dict(data.get("mission", {}))
    starts = [Start(id=str(item["id"]), north=float(item["north"]), east=float(item["east"]),
                    note=item.get("note", "")) for item in data.get("starts") or []]
    if not starts:  # no list: fly from mission.StartState only
        state = mission.get("StartState", [0, 0, -7, 0, 0, 0])
        starts = [Start(id="default", north=float(state[0]), east=float(state[1]))]
    return Matrix(campaign=data.get("campaign", "campaign"), mission=mission,
                  seeds=[int(seed) for seed in data.get("seeds", [513])], scenarios=scenarios,
                  controller=controller, starts=starts)
