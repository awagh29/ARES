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
class Matrix:
    campaign: str
    mission: dict
    seeds: list
    scenarios: list

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
    return Matrix(campaign=data.get("campaign", "campaign"), mission=dict(data.get("mission", {})),
                  seeds=[int(seed) for seed in data.get("seeds", [513])], scenarios=scenarios)
