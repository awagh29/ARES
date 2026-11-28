"""Swarm hub unit tests (no Gazebo or MATLAB)."""

from __future__ import annotations

import numpy as np

from ares_gz.scene import scene_map
from ares_gz.swarm.agent import DroneAgent
from ares_gz.swarm.cbf import SeparationContract, filter_velocity, joint_constraint_holds, project_box_halfspaces
from ares_gz.swarm.comms import CommsProfile, Link
from ares_gz.swarm.exploration import Task, allocate, assign_slots
from ares_gz.swarm.hub import Hub, HubConfig, Uplink
from ares_gz.swarm.mapping import SharedMap, ring_from_scan
from ares_gz.swarm.trust import DEGRADED, NOMINAL, QUARANTINED, TrustModel
from ares_gz.navigation import virtual_lidar


def test_qp_projection_matches_closed_form_halfspace():
    nominal = np.array([1.0, 0.0, 0.0])
    row = np.array([[-1.0, 0.0, 0.0]])  # -u_x >= -0.25  ->  u_x <= 0.25
    safe = project_box_halfspaces(nominal, row, np.array([-0.25]), np.array([2.0, 2.0, 2.0]))
    assert np.allclose(safe, [0.25, 0.0, 0.0], atol=1e-6)


def test_split_constraints_imply_the_pair_invariant():
    rng = np.random.default_rng(0)
    gamma, radius = 1.0, 3.5
    for _ in range(200):
        p_i, p_j = rng.uniform(-6, 6, 3), rng.uniform(-6, 6, 3)
        if np.linalg.norm(p_i - p_j) < radius:
            continue
        share = rng.choice([0.2, 0.5, 0.8])
        u_i, _ = filter_velocity(rng.uniform(-1.5, 1.5, 3), p_i, [SeparationContract("j", p_j, radius, share)], gamma, 1.5)
        u_j, _ = filter_velocity(rng.uniform(-1.5, 1.5, 3), p_j, [SeparationContract("i", p_i, radius, 1 - share)], gamma, 1.5)
        assert joint_constraint_holds(p_i, u_i, p_j, u_j, radius, gamma, tolerance=1e-5)


def test_cbf_keeps_head_on_drones_apart_in_closed_loop():
    """Two drones flying straight at each other never get closer than the CBF radius."""
    dt, radius = 0.01, 3.5
    p = {"a": np.array([0.0, 0.0, -7.0]), "b": np.array([10.0, 0.2, -7.0])}
    goal = {"a": np.array([10.0, 0.0, -7.0]), "b": np.array([0.0, 0.0, -7.0])}
    closest = np.inf
    for _ in range(2000):
        u = {}
        for me, other in (("a", "b"), ("b", "a")):
            nominal = 1.5 * (goal[me] - p[me]) / max(np.linalg.norm(goal[me] - p[me]), 1e-9)
            u[me], _ = filter_velocity(nominal, p[me], [SeparationContract(other, p[other], radius, 0.5)], 1.0, 1.5)
        for k in p:
            p[k] = p[k] + u[k] * dt
        closest = min(closest, np.linalg.norm(p["a"] - p["b"]))
    assert closest >= radius - 1e-3


def test_trust_flags_short_outage_and_quarantines_blackout():
    model = TrustModel()
    for _ in range(30):
        assert model.update(True).state == NOMINAL
    for _ in range(15):  # 1.5 s outage
        state = model.update(False).state
    assert state == DEGRADED
    for _ in range(80):  # recover with good scans
        state = model.update(True).state
    assert state == NOMINAL
    blackout = TrustModel()
    states = [blackout.update(False).state for _ in range(40)]
    assert states[0] == DEGRADED and states[-1] == QUARANTINED
    assert any("no measured LiDAR" in reason for reason in blackout.last.reasons)


def test_shared_map_fuses_scan_and_peer_contradiction_is_low_for_truth():
    scene = scene_map()
    shared = SharedMap(["UAV1", "UAV2"], -7.0)
    position = np.array([10.0, 20.0, -7.0])
    cloud, _, _ = virtual_lidar(scene, position, [0.0, 0.0, 0.0])
    ring = ring_from_scan(cloud, position, [0.0, 0.0, 0.0], scene)
    shared.fuse("UAV1", ring)
    free, occupied, _ = shared.states()
    assert free.sum() > 200
    other = np.array([12.0, 18.0, -7.0])
    cloud2, _, _ = virtual_lidar(scene, other, [0.5, 0.0, 0.0])
    contradiction, checked = shared.peer_contradiction("UAV2", ring_from_scan(cloud2, other, [0.5, 0.0, 0.0], scene))
    assert checked > 30 and contradiction < 0.1
    assert 0.0 < shared.coverage(scene) < 1.0


def test_allocation_spreads_drones_and_slots_minimise_distance():
    tasks = [Task("E1", "explore", np.array([5.0, 5.0, -7.0]), 5.0),
             Task("E2", "explore", np.array([6.0, 6.0, -7.0]), 5.0),
             Task("E3", "explore", np.array([30.0, 5.0, -7.0]), 5.0)]
    result = allocate({"UAV1": np.array([4.0, 4.0, -7.0]), "UAV2": np.array([28.0, 4.0, -7.0])}, tasks, 6.0, 0.0)
    assert result["UAV1"].id == "E1" and result["UAV2"].id == "E3"
    slots = assign_slots({"a": np.array([0.0, 0.0]), "b": np.array([10.0, 0.0])}, [[10.0, 1.0], [0.0, 1.0]])
    assert np.allclose(slots["a"], [0.0, 1.0]) and np.allclose(slots["b"], [10.0, 1.0])


def test_hub_quarantines_blackout_drone_reassigns_and_flags_victim_report():
    scene = scene_map()
    slots = {"UAV1": [30, 35], "UAV2": [30, 29], "UAV3": [24, 35]}
    hub = Hub(list(slots), slots, scene, HubConfig(explore_time_s=1e9))
    starts = {"UAV1": np.array([0.0, 15.0, -7.0]), "UAV2": np.array([4.0, 19.0, -7.0]), "UAV3": np.array([0.0, 23.0, -7.0])}
    for k in range(60):
        t = round(k * 0.1, 2)
        for drone, p in starts.items():
            measured = drone != "UAV2" or t < 1.0
            ring = None
            if measured:
                cloud, _, _ = virtual_lidar(scene, p, [0.0, 0.0, 0.0])
                ring = ring_from_scan(cloud, p, [0.0, 0.0, 0.0], scene)
            victims = [{"truth_id": "v1", "position_ne": [6.0, 19.0]}] if (drone == "UAV2" and k == 30) else []
            hub.receive(Uplink(drone, t, k, p, np.zeros(3), measured, not measured, ring, True, victims))
        downlinks = {d.drone: d for d in hub.tick(t)}
    assert hub.drones["UAV2"].mode == QUARANTINED
    assert downlinks["UAV2"].task_kind in ("limp-home", "hold")
    assert hub.victims and hub.victims[0]["status"] == "UNCONFIRMED" and hub.victims[0]["reporter_lidar_failed"]
    confirm = [t for t in hub.tasks if t.kind == "confirm"]
    assert confirm and confirm[0].assigned_to in ("UAV1", "UAV3")
    # Neighbours take the larger share of separation duty around the contained drone.
    shares = {c.neighbour: c.share for c in downlinks["UAV1"].contracts}
    assert shares["UAV2"] == 0.8
    assert any("QUARANTINED" in e["event"] for e in hub.events)


def test_agent_holds_on_comms_loss_and_link_drops_are_seeded():
    agent = DroneAgent("UAV1")
    assert np.allclose(agent.filter(0.0, np.zeros(3), [1.0, 0, 0]), 0.0)  # no downlink yet -> hold
    link_a = Link(CommsProfile("x", 0.05, 0.02, 0.3), seed=4)
    link_b = Link(CommsProfile("x", 0.05, 0.02, 0.3), seed=4)
    for k in range(100):
        link_a.send(k * 0.1, k)
        link_b.send(k * 0.1, k)
    assert link_a.receive(20.0) == link_b.receive(20.0)
    assert 0 < link_a.dropped < 60


def test_world_has_three_namespaced_drones_and_bridge_topics():
    from ares_gz.gazebo import bridge_args
    from ares_gz.world import world_sdf

    scene = scene_map()
    sdf = world_sdf(scene, None, drones=[("uav1", [0, 15, -7], (0, 0, 0)), ("uav2", [4, 19, -7], (0, 0, 0)),
                                         ("uav3", [0, 23, -7], (0, 0, 0))], markers=[("victim_v1", 5, 30)])
    for name in ("uav1", "uav2", "uav3"):
        assert f'<model name="{name}">' in sdf and f"<topic>/{name}/lidar</topic>" in sdf
        assert f"<odom_topic>/{name}/odometry</odom_topic>" in sdf
    assert '<model name="victim_v1">' in sdf
    args = bridge_args(["uav1", "uav2", "uav3"])
    assert sum("/cmd_vel@" in a for a in args) == 3 and sum("ControlWorld" in a for a in args) == 1
    single = world_sdf(scene, [0, 0, -7])
    assert single.count("<model name=\"ares_quad\">") == 1
