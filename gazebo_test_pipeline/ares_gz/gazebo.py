"""Lock-step Gazebo session over ROS 2 (``ros_gz_bridge``).

Gazebo starts paused. Each control tick the runner reads the latest state and
lidar scan, sends its command, then asks Gazebo to advance exactly
``steps_per_tick`` physics steps through the bridged ``ControlWorld`` service
and waits until the matching odometry arrives. Runs are therefore repeatable
and independent of how fast the computer is.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time as wallclock
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .world import MODEL_NAME, WORLD_NAME

ODOMETRY_TOPIC = f"/{MODEL_NAME}/odometry"
POINTS_TOPIC = f"/{MODEL_NAME}/lidar/points"
CMD_TOPIC = f"/{MODEL_NAME}/cmd_vel"
ENABLE_TOPIC = f"/{MODEL_NAME}/enable"
CONTROL_SERVICE = f"/world/{WORLD_NAME}/control"
WRENCH_TOPIC = f"/world/{WORLD_NAME}/wrench/persistent"
WRENCH_CLEAR_TOPIC = f"/world/{WORLD_NAME}/wrench/clear"
WRENCH_MIN_CHANGE_N = 0.02  # smaller drag changes are not sent (0.013 m/s^2 on the 1.5 kg quad)
WRENCH_MAX_ENTRIES = 50  # then clear the link's persistent list and re-apply the total
BASE_LINK = f"{MODEL_NAME}::base_link"

def model_topics(name: str) -> dict:
    return {"odometry": f"/{name}/odometry", "points": f"/{name}/lidar/points", "cmd": f"/{name}/cmd_vel",
            "enable": f"/{name}/enable", "link": f"{name}::base_link"}


def bridge_args(models=(MODEL_NAME,)) -> list[str]:
    args = [f"{CONTROL_SERVICE}@ros_gz_interfaces/srv/ControlWorld",
            f"{WRENCH_TOPIC}@ros_gz_interfaces/msg/EntityWrench]gz.msgs.EntityWrench",
            f"{WRENCH_CLEAR_TOPIC}@ros_gz_interfaces/msg/Entity]gz.msgs.Entity"]
    for name in models:
        topics = model_topics(name)
        args += [f"{topics['odometry']}@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                 f"{topics['points']}@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
                 f"{topics['cmd']}@geometry_msgs/msg/Twist]gz.msgs.Twist",
                 f"{topics['enable']}@std_msgs/msg/Bool]gz.msgs.Boolean"]
    return args


BRIDGE_ARGS = bridge_args()


class GazeboError(RuntimeError):
    pass


def process_domain_id() -> int:
    """One ROS 2 domain per process (rclpy can only join one). Workers set ARES_ROS_DOMAIN."""
    value = os.environ.get("ARES_ROS_DOMAIN")
    return int(value) if value else 40 + (os.getpid() % 100)


@dataclass
class OdometrySample:
    time_s: float
    position_enu: np.ndarray
    orientation_wxyz: tuple
    linear_velocity: np.ndarray  # as published (see Diagnostics: body or world frame)
    angular_velocity: np.ndarray


@dataclass
class ScanSample:
    time_s: float
    xyz: np.ndarray  # (height*width, 3) float32, Gazebo packing order
    height: int
    width: int


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _parse_points(message) -> np.ndarray:
    offsets = {field.name: field.offset for field in message.fields}
    if not {"x", "y", "z"} <= set(offsets):
        raise GazeboError(f"PointCloud2 without x/y/z fields: {list(offsets)}")
    count = message.width * message.height
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(count, message.point_step)
    xyz = np.empty((count, 3), dtype=np.float32)
    for column, name in enumerate("xyz"):
        start = offsets[name]
        xyz[:, column] = raw[:, start:start + 4].copy().view("<f4").reshape(-1)
    return xyz


class GazeboSession:
    """Owns the gz server, the bridge and one rclpy node for a single run.

    ``models`` names every vehicle in the world (one by default). The
    single-vehicle attributes (``odometry``, ``latest_scan``, ``send_velocity`` ...)
    refer to the first model; ``odometry_by``/``scan_by`` and ``model=`` address the others.
    """

    def __init__(self, world_file: Path, log_dir: Path, gui: bool = False, headless_rendering: bool = True,
                 steps_per_tick: int = 50, physics_step_s: float = 0.001, domain_id: int | None = None,
                 timeout_s: float = 30.0, render_warmup_s: float = 3.0, stall_retry_s: float = 5.0,
                 scan_wait_s: float = 5.0, models=(MODEL_NAME,)):
        self.world_file = Path(world_file)
        self.log_dir = Path(log_dir)
        self.gui = gui
        self.headless_rendering = headless_rendering
        self.steps_per_tick = steps_per_tick
        self.physics_step_s = physics_step_s
        self.timeout_s = timeout_s
        self.render_warmup_s = render_warmup_s
        self.stall_retry_s = stall_retry_s
        self.scan_wait_s = scan_wait_s
        self.step_retries = 0
        self.missed_scans = 0
        self.partition = f"ares_{uuid.uuid4().hex[:12]}"
        self.domain_id = domain_id if domain_id is not None else process_domain_id()
        self.processes: list[subprocess.Popen] = []
        self.models = tuple(models)
        self.odometry_by: dict[str, list[OdometrySample]] = {name: [] for name in self.models}
        self.scan_by: dict[str, ScanSample | None] = {name: None for name in self.models}
        self.scan_count_by = {name: 0 for name in self.models}
        self.wrench_by = {name: np.zeros(3) for name in self.models}
        self.wrench_entries = {name: 0 for name in self.models}
        self.wrench_clears = 0
        self.node = None

    # Single-vehicle view (first model).
    @property
    def odometry(self) -> list[OdometrySample]:
        return self.odometry_by[self.models[0]]

    @property
    def latest_scan(self) -> ScanSample | None:
        return self.scan_by[self.models[0]]

    @property
    def scan_count(self) -> int:
        return sum(self.scan_count_by.values())

    @property
    def wrench_total(self) -> np.ndarray:
        return self.wrench_by[self.models[0]]

    # ---- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "GazeboSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _env(self) -> dict:
        env = dict(os.environ)
        env["GZ_PARTITION"] = self.partition
        env["ROS_DOMAIN_ID"] = str(self.domain_id)
        env.setdefault("GZ_IP", "127.0.0.1")
        return env

    def _spawn(self, name: str, command: list[str]) -> subprocess.Popen:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"{name}.log", "w")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=self._env(),
                                   start_new_session=True)
        self.processes.append(process)
        return process

    def start(self) -> None:
        if shutil.which("gz") is None:
            raise GazeboError("`gz` not found. Install Gazebo: sudo apt install ros-lyrical-ros-gz")
        os.environ["ROS_DOMAIN_ID"] = str(self.domain_id)
        os.environ["GZ_PARTITION"] = self.partition
        server = ["gz", "sim", "-s", "-v", "3"]
        if self.headless_rendering:
            server.append("--headless-rendering")
        server.append(str(self.world_file))  # no -r: start paused, we step it
        self._spawn("gz_server", server)
        if self.gui:
            self._spawn("gz_gui", ["gz", "sim", "-g", "-v", "2"])
        self._spawn("bridge", ["ros2", "run", "ros_gz_bridge", "parameter_bridge", *bridge_args(self.models)])
        self._start_node()

    def _start_node(self) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from ros_gz_interfaces.msg import Entity, EntityWrench
        from ros_gz_interfaces.srv import ControlWorld
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import Bool

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(f"ares_gz_runner_{self.partition}")
        self._rclpy = rclpy
        self._Twist, self._Bool, self._EntityWrench, self._ControlWorld = Twist, Bool, EntityWrench, ControlWorld
        self.cmd_pubs, self.enable_pubs = {}, {}
        for name in self.models:
            topics = model_topics(name)
            self.node.create_subscription(Odometry, topics["odometry"],
                                          lambda message, name=name: self._on_odometry(message, name), 100)
            self.node.create_subscription(PointCloud2, topics["points"],
                                          lambda message, name=name: self._on_points(message, name), 5)
            self.cmd_pubs[name] = self.node.create_publisher(Twist, topics["cmd"], 10)
            self.enable_pubs[name] = self.node.create_publisher(Bool, topics["enable"], 10)
        self.cmd_pub = self.cmd_pubs[self.models[0]]
        self.enable_pub = self.enable_pubs[self.models[0]]
        self.wrench_pub = self.node.create_publisher(EntityWrench, WRENCH_TOPIC, 10)
        self.clear_pub = self.node.create_publisher(Entity, WRENCH_CLEAR_TOPIC, 10)
        self._Entity = Entity
        self.control = self.node.create_client(ControlWorld, CONTROL_SERVICE)
        deadline = wallclock.monotonic() + max(self.timeout_s, 60.0)
        while not self.control.wait_for_service(timeout_sec=0.5):
            self._check_processes()
            if wallclock.monotonic() > deadline:
                raise GazeboError(f"Gazebo control service did not appear. See logs in {self.log_dir}")
        # The bridge offers the ROS service before Gazebo has loaded the world; a step sent
        # then is lost. Wait until Gazebo itself advertises the control service.
        while not self._gazebo_service_ready():
            self._check_processes()
            if wallclock.monotonic() > deadline:
                raise GazeboError(f"Gazebo world did not finish loading. See logs in {self.log_dir}")
            wallclock.sleep(0.5)
        # The lidar renderer starts in a background thread; let it come up before sim time moves.
        warmup_end = wallclock.monotonic() + self.render_warmup_s
        while wallclock.monotonic() < warmup_end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        # Give the one-way bridges time to match publishers/subscribers before the first step.
        deadline = wallclock.monotonic() + 10.0
        publishers = [*self.cmd_pubs.values(), *self.enable_pubs.values(), self.wrench_pub, self.clear_pub]
        while wallclock.monotonic() < deadline and any(pub.get_subscription_count() == 0 for pub in publishers):
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _gazebo_service_ready(self) -> bool:
        try:
            listing = subprocess.run(["gz", "service", "-l"], capture_output=True, text=True, timeout=10,
                                     env=self._env()).stdout
        except (subprocess.TimeoutExpired, OSError):
            return False
        return CONTROL_SERVICE in listing.split()

    def close(self) -> None:
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = wallclock.monotonic() + 5.0
        for process in self.processes:
            try:
                process.wait(timeout=max(0.1, deadline - wallclock.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.processes.clear()

    def _check_processes(self) -> None:
        for process in self.processes:
            if process.poll() is not None:
                raise GazeboError(f"{' '.join(process.args[:3])} exited with code {process.returncode}. "
                                  f"See logs in {self.log_dir}")

    # ---- callbacks -----------------------------------------------------------
    def _on_odometry(self, message, name: str = MODEL_NAME) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        self.odometry_by[name].append(OdometrySample(
            time_s=_stamp(message.header),
            position_enu=np.array([pose.position.x, pose.position.y, pose.position.z]),
            orientation_wxyz=(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z),
            linear_velocity=np.array([twist.linear.x, twist.linear.y, twist.linear.z]),
            angular_velocity=np.array([twist.angular.x, twist.angular.y, twist.angular.z]),
        ))

    def _on_points(self, message, name: str = MODEL_NAME) -> None:
        self.scan_by[name] = ScanSample(_stamp(message.header), _parse_points(message), message.height, message.width)
        self.scan_count_by[name] += 1

    # ---- commands ------------------------------------------------------------
    def enable_controller(self, model: str | None = None) -> None:
        self.enable_pubs[model or self.models[0]].publish(self._Bool(data=True))

    def send_velocity(self, linear_xyz, yaw_rate: float, model: str | None = None) -> None:
        message = self._Twist()
        message.linear.x, message.linear.y, message.linear.z = (float(value) for value in linear_xyz)
        message.angular.z = float(yaw_rate)
        self.cmd_pubs[model or self.models[0]].publish(message)

    def set_force_enu(self, force_enu, model: str | None = None) -> None:
        """Set the total external force on the body (world frame).

        ApplyLinkWrench sums every persistent wrench on a link, so we publish
        only the change. The total is then correct whatever order the messages
        are processed in, and a late message delays the change by one step.

        Gazebo keeps every persistent wrench in a list and re-applies the whole
        list each physics step, so the list must stay short: changes below
        ``WRENCH_MIN_CHANGE_N`` are skipped, and after ``WRENCH_MAX_ENTRIES``
        entries the link's list is cleared and the total re-applied on the next
        tick (one 10 ms tick without the drag force, every >= 0.5 s).
        """
        model = model or self.models[0]
        force = np.asarray(force_enu, dtype=float)
        delta = force - self.wrench_by[model]
        if float(np.max(np.abs(delta))) < WRENCH_MIN_CHANGE_N:
            return
        if self.wrench_entries[model] >= WRENCH_MAX_ENTRIES:
            clear = self._Entity()
            clear.name = model_topics(model)["link"]
            clear.type = 3  # LINK
            self.clear_pub.publish(clear)
            self.wrench_by[model] = np.zeros(3)
            self.wrench_entries[model] = 0
            self.wrench_clears += 1
            return
        message = self._EntityWrench()
        message.entity.name = model_topics(model)["link"]
        message.entity.type = 3  # LINK
        message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = (float(value) for value in delta)
        self.wrench_pub.publish(message)
        self.wrench_by[model] = force
        self.wrench_entries[model] += 1

    def _request_step(self, deadline: float) -> None:
        request = self._ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = self.steps_per_tick
        future = self.control.call_async(request)
        while not future.done():
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            self._timeout(deadline, "ControlWorld reply")
        if not future.result().success:
            raise GazeboError("Gazebo rejected the step request.")

    def step(self, target_time_s: float, need_scan_at: float | None = None) -> OdometrySample:
        """Advance one control tick and wait for state (and scan) at ``target_time_s``.

        A step that produces no progress at all (no newer odometry) was lost and is
        sent again; that cannot double-step because nothing advanced. A scan that is
        late by more than ``scan_wait_s`` is counted in ``missed_scans`` and the run
        continues with the latest scan (its age is logged by the runner).
        """
        def newest(name: str) -> float:
            samples = self.odometry_by[name]
            return samples[-1].time_s if samples else -1.0

        before = min(newest(name) for name in self.models)
        deadline = wallclock.monotonic() + self.timeout_s
        self._request_step(deadline)
        sent = wallclock.monotonic()
        while not all(newest(name) >= target_time_s - 1e-6 for name in self.models):
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            progressed = any(newest(name) > before + 1e-9 for name in self.models)
            if not progressed and wallclock.monotonic() - sent > self.stall_retry_s:
                self.step_retries += 1
                self._request_step(deadline)
                sent = wallclock.monotonic()
            self._timeout(deadline, f"state at t={target_time_s:.3f} s")
        if need_scan_at is not None:
            scan_deadline = wallclock.monotonic() + self.scan_wait_s
            def scanned(name: str) -> bool:
                scan = self.scan_by[name]
                return scan is not None and scan.time_s >= need_scan_at - 1e-6

            while not all(scanned(name) for name in self.models):
                if wallclock.monotonic() > scan_deadline:
                    self.missed_scans += 1
                    break
                self._rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.odometry[-1]

    def _timeout(self, deadline: float, what: str) -> None:
        if wallclock.monotonic() > deadline:
            self._check_processes()
            raise GazeboError(f"Timed out waiting for {what}. See logs in {self.log_dir}")
