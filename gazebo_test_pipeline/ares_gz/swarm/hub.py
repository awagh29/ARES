"""The swarm hub: the only node every drone talks to (star topology).

Every hub tick (10 Hz) it:
1. reads the uplinks that arrived (state, LiDAR health, measured scan ring,
   victim sightings, task status);
2. updates each drone's interpretable trust (``trust.py``) and fuses only
   trusted, measured scans into the shared map (``mapping.py``);
3. applies the safe operating modes:
     NOMINAL      full speed, eligible for any task
     DEGRADED     LiDAR flagged: speed cap, keeps its task, gets no new one,
                  its scans are not fused, neighbours take more of the
                  separation duty
     QUARANTINED  contained: hover for ``hold_s``, then limp to its
                  rendezvous slot at low speed on the CNN reconstruction;
                  its task is reassigned and neighbours keep a wider berth
     LOST         no uplink for ``lost_after_s``: task reassigned, wider berth
4. allocates tasks (frontier exploration, victim confirmation, rendezvous
   slots) to NOMINAL drones (``exploration.py``);
5. sends each drone its mode, target, speed cap and the separation contracts
   that make the pairwise distance invariant hold (``cbf.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene import SceneMap
from .cbf import SeparationContract
from .exploration import Task, allocate, assign_slots, frontier_targets, safe_free_mask
from .mapping import Ring, SharedMap
from .trust import DEGRADED, NOMINAL, QUARANTINED, TrustModel

LOST = "LOST"


@dataclass
class Uplink:
    drone: str
    time_s: float
    seq: int
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    lidar_measured: bool
    reconstruction_used: bool = False
    ring: Ring | None = None
    scan_sane: bool = True
    victims: list = field(default_factory=list)
    task_id: str | None = None
    target_rejected: str | None = None
    terminal: bool = False


@dataclass
class Downlink:
    drone: str
    time_s: float
    seq: int
    mode: str
    task_id: str | None
    task_kind: str | None
    target_ned: np.ndarray | None
    speed_cap: float
    contracts: list
    stamps: list
    neighbour_speed: list
    gamma: float


@dataclass
class HubConfig:
    altitude_down: float = -7.0
    separation_m: float = 3.0  # the invariant: centre distance never below this
    buffer_m: float = 0.5  # CBF works on separation + buffer to absorb tracking lag
    gamma: float = 1.0
    cruise_mps: float = 1.5
    degraded_cap_mps: float = 0.8
    limp_cap_mps: float = 0.6
    quarantine_extra_m: float = 1.0
    explore_time_s: float = 90.0
    coverage_goal: float = 0.9
    reach_radius_m: float = 1.5
    task_timeout_s: float = 35.0
    lost_after_s: float = 1.0
    hold_s: float = 3.0
    allocation_period_s: float = 1.0
    frontier_clearance_m: float = 2.0
    target_separation_m: float = 6.0
    victim_merge_m: float = 2.5


@dataclass
class DroneRecord:
    id: str
    slot: np.ndarray
    trust: TrustModel = field(default_factory=TrustModel)
    mode: str = NOMINAL
    last: Uplink | None = None
    last_heard_s: float = -np.inf
    task: Task | None = None
    quarantined_s: float | None = None
    arrived: bool = False
    seq: int = 0


class Hub:
    def __init__(self, drones: list[str], slots: dict, scene: SceneMap, config: HubConfig | None = None):
        self.config = config or HubConfig()
        self.scene = scene
        self.shared = SharedMap(drones, self.config.altitude_down)
        self.drones = {drone: DroneRecord(drone, np.asarray(slots[drone], float)) for drone in drones}
        self.tasks: list[Task] = []
        self.task_counter = 0
        self.phase = "EXPLORE"
        self.events: list[dict] = []
        self.victims: list[dict] = []
        self.trust_log: list[dict] = []
        self.coverage_log: list[tuple] = []
        self.blocked_targets: list = []
        self.next_allocation_s = 0.0
        self.time_s = 0.0

    # ---- helpers -----------------------------------------------------------
    def event(self, drone: str | None, text: str, **extra) -> None:
        self.events.append({"time_s": round(self.time_s, 2), "drone": drone, "event": text, **extra})

    def new_task(self, kind: str, target_ne, gain: float, note: str = "") -> Task:
        self.task_counter += 1
        target = np.array([float(target_ne[0]), float(target_ne[1]), self.config.altitude_down])
        task = Task(f"{kind[0].upper()}{self.task_counter}", kind, target, gain, created_s=self.time_s, note=note)
        self.tasks.append(task)
        return task

    def release(self, record: DroneRecord, why: str) -> None:
        task = record.task
        if task is not None and task.status == "assigned" and task.kind in ("explore", "confirm"):
            task.status, task.assigned_to = "open", None
            task.history.append((round(self.time_s, 2), f"released by {record.id}: {why}"))
            self.event(record.id, f"task {task.id} ({task.kind}) released for reassignment: {why}")
        record.task = None

    # ---- uplink processing ------------------------------------------------
    def receive(self, message: Uplink) -> None:
        record = self.drones[message.drone]
        if record.last is not None and message.time_s <= record.last.time_s:
            return  # out of order or duplicate
        record.last = message
        record.last_heard_s = message.time_s
        contradiction, checked = (None, 0)
        if message.ring is not None:
            contradiction, checked = self.shared.peer_contradiction(message.drone, message.ring)
        assessment = record.trust.update(message.lidar_measured, message.scan_sane, contradiction, checked)
        self.trust_log.append({"time_s": round(message.time_s, 2), "drone": message.drone,
                               "trust": assessment.trust, "state": assessment.state,
                               **{f"penalty_{k}": round(v, 3) for k, v in assessment.penalties.items()},
                               "reasons": "; ".join(assessment.reasons)})
        if message.ring is not None and assessment.state == NOMINAL:
            self.shared.fuse(message.drone, message.ring)
        self.shared.mark_searched(message.position_ned)  # the camera works with or without LiDAR
        for sighting in message.victims:
            self.report_victim(record, sighting, message)
        if message.target_rejected and record.task is not None and record.task.id == message.target_rejected:
            task = record.task
            task.status = "failed"
            task.history.append((round(self.time_s, 2), "target rejected by the drone's planner"))
            self.blocked_targets.append(task.target_ned)
            self.event(record.id, f"task {task.id} target rejected by onboard planner; blacklisted")
            record.task = None

    def report_victim(self, record: DroneRecord, sighting: dict, message: Uplink) -> None:
        position = np.asarray(sighting["position_ne"], float)
        trusted = message.lidar_measured and record.mode == NOMINAL
        report = {"drone": record.id, "time_s": round(message.time_s, 2), "lidar_ok": bool(message.lidar_measured),
                  "trust": record.trust.last.trust if record.trust.last else None, "mode": record.mode}
        for victim in self.victims:
            if np.linalg.norm(victim["position_ne"] - position) < self.config.victim_merge_m:
                victim["reports"].append(report)
                if trusted and victim["status"] != "CONFIRMED":
                    victim["status"], victim["confirmed_by"], victim["confirmed_s"] = "CONFIRMED", record.id, report["time_s"]
                    self.event(record.id, f"victim {victim['id']} confirmed by {record.id}")
                    for task in self.tasks:
                        if task.kind == "confirm" and task.note == victim["id"] and task.status in ("open", "assigned"):
                            task.status = "done"
                            task.history.append((round(self.time_s, 2), f"confirmed by {record.id}"))
                return
        victim = {"id": f"V{len(self.victims) + 1}", "position_ne": position, "truth_id": sighting.get("truth_id"),
                  "first_reported_by": record.id, "first_reported_s": report["time_s"],
                  "reporter_lidar_failed": not message.lidar_measured,
                  "status": "CONFIRMED" if trusted else "UNCONFIRMED",
                  "confirmed_by": record.id if trusted else None,
                  "confirmed_s": report["time_s"] if trusted else None, "reports": [report]}
        self.victims.append(victim)
        lidar = "LiDAR OK" if message.lidar_measured else "LiDAR FAILED (reconstruction)"
        self.event(record.id, f"victim {victim['id']} reported at ({position[0]:.1f}, {position[1]:.1f}) "
                              f"by {record.id} [{lidar}, mode {record.mode}]"
                              + ("" if trusted else " -> needs confirmation by a trusted drone"))
        if not trusted:
            self.new_task("confirm", self.confirmation_point(position), gain=1000.0, note=victim["id"])

    def confirmation_point(self, victim_ne: np.ndarray) -> np.ndarray:
        safe = safe_free_mask(self.shared, self.config.frontier_clearance_m)
        cells = np.argwhere(safe)
        if cells.size:
            centers = np.column_stack((self.shared.grid.north[cells[:, 0]], self.shared.grid.east[cells[:, 1]]))
            distance = np.linalg.norm(centers - victim_ne, axis=1)
            if distance.min() <= 3.0:
                return centers[int(np.argmin(distance))]
        return victim_ne

    # ---- hub tick ------------------------------------------------------------
    def tick(self, time_s: float) -> list[Downlink]:
        self.time_s = time_s
        cfg = self.config
        for record in self.drones.values():
            self.update_mode(record)
            self.update_task(record)
        coverage = self.shared.search_coverage(self.scene)
        if not self.coverage_log or time_s - self.coverage_log[-1][0] >= 1.0 - 1e-9:
            self.coverage_log.append((round(time_s, 2), round(coverage, 4), round(self.shared.coverage(self.scene), 4)))
        open_confirm = any(t.kind == "confirm" and t.status in ("open", "assigned") for t in self.tasks)
        if self.phase == "EXPLORE" and (time_s >= cfg.explore_time_s or (coverage >= cfg.coverage_goal and not open_confirm)):
            self.start_rendezvous(coverage)
        if time_s >= self.next_allocation_s - 1e-9:
            self.next_allocation_s = time_s + cfg.allocation_period_s
            self.allocate()
        return [self.downlink(record) for record in self.drones.values()]

    def update_mode(self, record: DroneRecord) -> None:
        cfg = self.config
        previous = record.mode
        if self.time_s - record.last_heard_s > cfg.lost_after_s and record.last is not None:
            mode = LOST
        elif record.last is None:
            mode = NOMINAL
        else:
            mode = record.trust.state
        if mode == previous:
            return
        record.mode = mode
        reasons = record.trust.last.reasons if record.trust.last else []
        trust = record.trust.last.trust if record.trust.last else None
        self.event(record.id, f"mode {previous} -> {mode}", trust=trust, reasons="; ".join(reasons))
        if mode == QUARANTINED:
            record.quarantined_s = self.time_s
            self.release(record, "quarantined (LiDAR untrusted)")
        elif mode == LOST:
            self.release(record, "lost contact")
        elif mode == NOMINAL and previous in (QUARANTINED, LOST):
            record.quarantined_s = None
            record.task = None  # rejoins and gets a fresh task at the next allocation

    def update_task(self, record: DroneRecord) -> None:
        task = record.task
        if record.last is None:
            return
        position = record.last.position_ned
        if record.mode == QUARANTINED:
            # A contained drone limps to its slot; it counts as arrived once it rests there.
            at_slot = np.linalg.norm(record.slot[:2] - position[:2]) <= 0.5
            if not record.arrived and at_slot and float(np.linalg.norm(record.last.velocity_ned)) <= 0.3:
                record.arrived = True
                self.event(record.id, "contained drone is holding safely at its rendezvous slot")
            return
        if task is None:
            return
        distance = float(np.linalg.norm(task.target_ned[:2] - position[:2]))
        if task.kind in ("explore", "confirm"):
            if task.status != "assigned":  # e.g. a confirm task completed by another drone's sighting
                record.task = None
            elif task.kind == "explore" and distance <= self.config.reach_radius_m:
                task.status = "done"
                task.history.append((round(self.time_s, 2), f"reached by {record.id}"))
                self.blocked_targets.append(task.target_ned)  # visited: do not send anyone back here
                record.task = None
            elif task.assigned_s is not None and self.time_s - task.assigned_s > self.config.task_timeout_s:
                task.status = "failed"
                task.history.append((round(self.time_s, 2), "timed out"))
                self.blocked_targets.append(task.target_ned)
                self.event(record.id, f"task {task.id} timed out; blacklisted")
                record.task = None
        elif task.kind == "rendezvous" and not record.arrived:
            speed = float(np.linalg.norm(record.last.velocity_ned))
            if (distance <= 0.5 and speed <= 0.3) or record.last.terminal:
                record.arrived = True
                task.status = "done"
                self.event(record.id, f"arrived at rendezvous slot ({task.target_ned[0]:.0f}, {task.target_ned[1]:.0f})")

    def start_rendezvous(self, coverage: float) -> None:
        self.phase = "RENDEZVOUS"
        self.event(None, f"search finished (camera coverage {coverage:.0%}); all drones to rendezvous slots")
        for record in self.drones.values():
            self.release(record, "rendezvous")
        positions = {d: r.last.position_ned for d, r in self.drones.items() if r.last is not None}
        for drone, slot in assign_slots(positions, [r.slot for r in self.drones.values()]).items():
            self.drones[drone].slot = slot

    def allocate(self) -> None:
        cfg = self.config
        if self.phase == "EXPLORE":
            for task in self.tasks:  # frontiers move as the map grows: rebuild the open ones
                if task.kind == "explore" and task.status == "open":
                    task.status = "superseded"
            self.tasks = [task for task in self.tasks if task.status != "superseded"]
            for target, size in frontier_targets(self.shared, cfg.frontier_clearance_m, blocked=self.blocked_targets):
                self.new_task("explore", target, gain=float(size))
        if self.phase == "EXPLORE":
            self.preempt_for_confirmation()
        idle = {}
        for drone, record in self.drones.items():
            if record.last is None or record.mode != NOMINAL:
                continue
            if self.phase == "RENDEZVOUS":
                if record.task is None or record.task.kind != "rendezvous":
                    record.task = Task(f"R-{drone}", "rendezvous", np.array([*record.slot[:2], cfg.altitude_down]),
                                       assigned_to=drone, status="assigned", created_s=self.time_s)
                continue
            if record.task is None:
                idle[drone] = record.last.position_ned
        if idle:
            for drone, task in allocate(idle, self.tasks, cfg.target_separation_m, self.time_s).items():
                self.drones[drone].task = task
                self.event(drone, f"task {task.id} ({task.kind}) -> ({task.target_ned[0]:.1f}, {task.target_ned[1]:.1f})")

    def preempt_for_confirmation(self) -> None:
        """An unconfirmed victim outranks exploration: free the nearest trusted explorer if nobody is idle."""
        for task in self.tasks:
            if task.kind != "confirm" or task.status != "open":
                continue
            nominal = [r for r in self.drones.values() if r.mode == NOMINAL and r.last is not None]
            if any(r.task is None for r in nominal):
                continue
            explorers = [r for r in nominal if r.task is not None and r.task.kind == "explore"]
            if not explorers:
                continue
            nearest = min(explorers, key=lambda r: np.linalg.norm(r.last.position_ned[:2] - task.target_ned[:2]))
            self.release(nearest, f"pre-empted to confirm victim {task.note}")

    # ---- downlink ------------------------------------------------------------
    def target_for(self, record: DroneRecord) -> tuple[np.ndarray | None, float, str | None, str | None]:
        cfg = self.config
        position = record.last.position_ned if record.last is not None else None
        if record.mode == QUARANTINED:
            if record.quarantined_s is not None and self.time_s - record.quarantined_s < cfg.hold_s:
                return None, 0.0, None, "hold"  # hover in place
            return np.array([*record.slot[:2], cfg.altitude_down]), cfg.limp_cap_mps, "limp-home", "limp-home"
        if record.mode == LOST:
            return None, 0.0, None, "hold"
        cap = cfg.degraded_cap_mps if record.mode == DEGRADED else cfg.cruise_mps
        if record.task is not None:
            return record.task.target_ned, cap, record.task.id, record.task.kind
        if position is None:
            return None, 0.0, None, "hold"
        return None, cap, None, "hold"

    def downlink(self, record: DroneRecord) -> Downlink:
        cfg = self.config
        target, cap, task_id, kind = self.target_for(record)
        contracts, stamps, speeds = [], [], []
        for other in self.drones.values():
            if other.id == record.id or other.last is None:
                continue
            unreliable_self = record.mode != NOMINAL
            unreliable_other = other.mode != NOMINAL
            share = 0.5 if unreliable_self == unreliable_other else (0.2 if unreliable_self else 0.8)
            extra = cfg.quarantine_extra_m if {record.mode, other.mode} & {QUARANTINED, LOST} else 0.0
            contracts.append(SeparationContract(other.id, other.last.position_ned.copy(),
                                                cfg.separation_m + cfg.buffer_m + extra, share))
            stamps.append(other.last.time_s)
            speeds.append(self.speed_limit(other))
        record.seq += 1
        return Downlink(record.id, self.time_s, record.seq, record.mode, task_id, kind, target, cap,
                        contracts, stamps, speeds, cfg.gamma)

    def speed_limit(self, record: DroneRecord) -> float:
        cfg = self.config
        return {NOMINAL: cfg.cruise_mps, DEGRADED: cfg.degraded_cap_mps}.get(record.mode, max(cfg.limp_cap_mps, 0.3)) + 0.3

    # ---- report ----------------------------------------------------------------
    def summary(self) -> dict:
        victims = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in victim.items()}
                   for victim in self.victims]
        return {
            "phase": self.phase,
            "coverage": self.coverage_log[-1][1] if self.coverage_log else 0.0,
            "map_coverage": self.coverage_log[-1][2] if self.coverage_log else 0.0,
            "coverage_log": self.coverage_log,
            "victims": victims,
            "modes": {d: r.mode for d, r in self.drones.items()},
            "arrived": {d: r.arrived for d, r in self.drones.items()},
            "trust_final": {d: (r.trust.last.trust if r.trust.last else None) for d, r in self.drones.items()},
            "failure_episodes": {d: r.trust.episodes for d, r in self.drones.items()},
            "fused_scans": dict(self.shared.fused_scans),
            "tasks": [{"id": t.id, "kind": t.kind, "target": t.target_ned.tolist(), "status": t.status,
                       "assigned_to": t.assigned_to, "note": t.note, "history": t.history} for t in self.tasks
                      if t.status != "superseded"],
        }
