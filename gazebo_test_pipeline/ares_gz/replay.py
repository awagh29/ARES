"""Replay a finished run in Gazebo and record it as smooth MP4 video.

The live test steps Gazebo in lock-step (it pauses while MATLAB thinks), which
looks stuttery in the GUI. A replay re-renders the *recorded* flight
(``control.csv``): for every video frame the drone model is placed at its
logged pose, Gazebo advances exactly one frame (1/25 s), and the camera images
are collected. Nothing is re-simulated, and the resulting video plays at normal
speed. The drone is drawn 3x larger so it is visible in a wide shot: green while
the lidar works, orange while the CNN reconstruction is used.

Each frame gets a HUD (scenario, time, lidar/CNN state, wind, clearance), a
live mini-map, and a final PASS/FAIL banner; ffmpeg encodes the MP4.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time as wallclock
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import matrix_to_quaternion, ned_euler_to_gazebo_rpy
from .navigation.geometry import eul2rotm_zyx
from .frames import ENU_NED, FLU_FRD
from .scene import scene_map

FPS = 25
PHYSICS_STEP_S = 0.01
STEPS_PER_FRAME = 4  # 4 x 0.01 s = 1/25 s
WORLD = "ares_replay"
VIEWS = {
    # name: (pose x y z roll pitch yaw in ENU, width, height, horizontal fov)
    # From the south-east corner, high and steep, so the SW->NE route is not hidden behind buildings.
    "overview": ((50.0, -20.0, 44.0, 0.0, 0.70, math.radians(133)), 1280, 720, 1.2),
    "top": ((17.5, 15.0, 52.0, 0.0, math.pi / 2, math.pi / 2), 1080, 1080, 1.0),
}
CATEGORIES = {
    "S0": "0_control_healthy_lidar",
    "S1": "1_intermittent_lidar_failure",
    "S2": "2_complete_lidar_blackout",
    "S3": "3_wind_gusts",
    "S4": "4_wind_gusts_plus_intermittent_lidar",
    "S5": "5_wind_gusts_plus_complete_blackout",
}


def video_category(scenario_id: str) -> str:
    """The experiment folder a scenario's videos belong in (by its S0..S5 prefix)."""
    return CATEGORIES.get(str(scenario_id)[:2], "other")


def result_folder(success) -> str:
    return "PASSED" if success else "FAILED"


def video_label(summary: dict) -> str:
    start = summary.get("start_id")
    return f"{summary.get('scenario')}" + (f"_{start}" if start and start != "default" else "") + \
        f"_seed{summary.get('seed')}"


def organize_videos(videos_dir: Path, campaign_dir: Path | None = None) -> int:
    """Sort videos into <experiment>/<PASSED|FAILED>/, using each run's summary.json. Returns files moved."""
    videos_dir = Path(videos_dir)
    campaign_dir = Path(campaign_dir) if campaign_dir else videos_dir.parent
    outcomes = {}
    for path in campaign_dir.rglob("summary.json"):
        summary = json.loads(path.read_text())
        if "scenario" in summary:
            outcomes[video_label(summary)] = bool(summary.get("mission_success"))
    moved = 0
    for video in sorted(videos_dir.rglob("*.mp4")):
        label = next((key for key in sorted(outcomes, key=len, reverse=True)
                      if video.name.startswith(key + "_")), None)
        if label is None:
            continue  # unknown run: leave it where it is
        target = videos_dir / video_category(video.name) / result_folder(outcomes[label]) / video.name
        if target == video:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        video.replace(target)
        moved += 1
    return moved


OK_COLOR = "0.1 0.75 0.2 1"
CNN_COLOR = "1.0 0.5 0.0 1"
HIDDEN = (0.0, 0.0, -50.0)
DRONE_SCALE = 3.0


@dataclass
class Track:
    time: np.ndarray
    ned: np.ndarray  # (N,3)
    euler: np.ndarray  # (N,3) yaw pitch roll
    lidar: np.ndarray  # (N,) 1 healthy
    cnn: np.ndarray  # (N,) 1 reconstruction used
    wind: np.ndarray  # (N,) speed m/s
    clearance: np.ndarray  # (N,)


def load_track(run_dir: Path) -> Track:
    with open(run_dir / "control.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{run_dir}/control.csv is empty")

    def column(name, default=0.0):
        return np.array([float(row[name]) if row.get(name) not in (None, "") else default for row in rows])

    lidar = column("lidar_available", 1.0)
    return Track(time=column("time_s"), ned=np.column_stack((column("n"), column("e"), column("d"))),
                 euler=np.column_stack((column("yaw"), column("pitch"), column("roll"))),
                 lidar=lidar, cnn=1.0 - lidar, wind=np.hypot(column("wind_n"), column("wind_e")),
                 clearance=column("clearance_m"))


def _pose_at(track: Track, t: float) -> tuple[np.ndarray, np.ndarray, bool]:
    ned = np.array([np.interp(t, track.time, track.ned[:, k]) for k in range(3)])
    yaw = float(np.interp(t, track.time, np.unwrap(track.euler[:, 0])))
    pitch = float(np.interp(t, track.time, track.euler[:, 1]))
    roll = float(np.interp(t, track.time, track.euler[:, 2]))
    k = int(np.clip(np.searchsorted(track.time, t, side="right") - 1, 0, len(track.time) - 1))
    return ned, np.array([yaw, pitch, roll]), bool(track.cnn[k] > 0.5)


def _drone_model(name: str, color: str) -> str:
    s = DRONE_SCALE
    rotors = "".join(
        f"""<visual name="rotor_{k}"><pose>{x * s} {y * s} {0.04 * s} 0 0 0</pose>
          <geometry><cylinder><radius>{0.1 * s}</radius><length>{0.01 * s}</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>"""
        for k, (x, y) in enumerate(((0.13, -0.22), (-0.13, 0.2), (0.13, 0.22), (-0.13, -0.2))))
    return f"""
    <model name="{name}"><static>true</static><pose>{HIDDEN[0]} {HIDDEN[1]} {HIDDEN[2]} 0 0 0</pose>
      <link name="link">
        <visual name="body"><geometry><box><size>{0.47 * s} {0.47 * s} {0.11 * s}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse><emissive>{color}</emissive></material></visual>
        {rotors}
      </link>
    </model>"""


def replay_world(views: list[str], goal_ned) -> str:
    scene = scene_map()
    buildings = "".join(f"""
    <model name="building_{k}"><static>true</static>
      <pose>{c[0]:.3f} {c[1]:.3f} {scene.height_m / 2:.3f} 0 0 0</pose>
      <link name="link"><visual name="v"><geometry><box><size>{scene.width_m} {scene.width_m} {scene.height_m}</size></box></geometry>
        <material><ambient>0.6 0.6 0.62 1</ambient><diffuse>0.7 0.7 0.72 1</diffuse></material></visual></link>
    </model>""" for k, c in enumerate(scene.centers_enu))
    cameras = []
    for view in views:
        (x, y, z, roll, pitch, yaw), width, height, fov = VIEWS[view]
        cameras.append(f"""
    <model name="camera_{view}"><static>true</static><pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
      <link name="link"><sensor name="{view}" type="camera">
        <always_on>1</always_on><update_rate>{FPS}</update_rate><topic>/replay/{view}</topic>
        <camera><horizontal_fov>{fov}</horizontal_fov>
          <image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>
          <clip><near>0.5</near><far>400</far></clip>
        </camera>
      </sensor></link>
    </model>""")
    gx, gy, gz = goal_ned[1], goal_ned[0], -goal_ned[2]
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{WORLD}">
    <physics name="replay" type="dart"><max_step_size>{PHYSICS_STEP_S}</max_step_size><real_time_factor>0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.72 0.82 0.95 1</background><grid>false</grid></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 40 0 0 0</pose>
      <diffuse>0.95 0.95 0.95 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.3 -0.85</direction></light>
    <model name="ground"><static>true</static><link name="link"><visual name="v">
      <geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry>
      <material><ambient>0.35 0.45 0.33 1</ambient><diffuse>0.4 0.5 0.38 1</diffuse></material></visual></link></model>
    <model name="goal"><static>true</static><pose>{gx} {gy} {gz} 0 0 0</pose><link name="link">
      <visual name="v"><geometry><cylinder><radius>0.6</radius><length>0.05</length></cylinder></geometry>
      <material><ambient>0.9 0.1 0.1 1</ambient><diffuse>0.9 0.1 0.1 1</diffuse><emissive>0.6 0 0 1</emissive></material></visual></link></model>
{buildings}
{_drone_model("drone_lidar_ok", OK_COLOR)}
{_drone_model("drone_cnn", CNN_COLOR)}
{''.join(cameras)}
  </world>
</sdf>
"""


class ReplaySession:
    """Lock-step Gazebo for replays: place models, advance one frame, collect camera images."""

    def __init__(self, world_file: Path, views: list[str], log_dir: Path, gui: bool = False):
        self.views = views
        self.log_dir = log_dir
        self.partition = f"replay_{uuid.uuid4().hex[:10]}"
        self.processes: list[subprocess.Popen] = []
        from .gazebo import process_domain_id
        env = dict(os.environ, GZ_PARTITION=self.partition, GZ_IP="127.0.0.1",
                   ROS_DOMAIN_ID=str(process_domain_id()))
        self.env = env
        log_dir.mkdir(parents=True, exist_ok=True)
        server = ["gz", "sim", "-s", "--headless-rendering", str(world_file)]
        self._spawn("server", server)
        if gui:
            self._spawn("gui", ["gz", "sim", "-g"])
        bridge = [f"/world/{WORLD}/control@ros_gz_interfaces/srv/ControlWorld",
                  f"/world/{WORLD}/set_pose@ros_gz_interfaces/srv/SetEntityPose"]
        bridge += [f"/replay/{view}@sensor_msgs/msg/Image[gz.msgs.Image" for view in views]
        self._spawn("bridge", ["ros2", "run", "ros_gz_bridge", "parameter_bridge", *bridge])
        os.environ["ROS_DOMAIN_ID"] = env["ROS_DOMAIN_ID"]
        import rclpy
        from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node(f"ares_replay_{self.partition}")
        self.ControlWorld, self.SetEntityPose = ControlWorld, SetEntityPose
        self.images: dict[str, tuple[float, object]] = {}
        for view in views:
            self.node.create_subscription(Image, f"/replay/{view}", self._image_callback(view), 5)
        self.control = self.node.create_client(ControlWorld, f"/world/{WORLD}/control")
        self.set_pose = self.node.create_client(SetEntityPose, f"/world/{WORLD}/set_pose")
        deadline = wallclock.monotonic() + 60
        while not (self.control.wait_for_service(timeout_sec=0.5) and self.set_pose.wait_for_service(timeout_sec=0.5)):
            if wallclock.monotonic() > deadline:
                raise RuntimeError(f"Replay Gazebo did not start; see {log_dir}")
        while f"/world/{WORLD}/control" not in subprocess.run(["gz", "service", "-l"], capture_output=True, text=True,
                                                              env=env).stdout.split():
            if wallclock.monotonic() > deadline:
                raise RuntimeError(f"Replay world did not load; see {log_dir}")
            wallclock.sleep(0.5)
        end = wallclock.monotonic() + 3.0  # let the renderer start
        while wallclock.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.time = 0.0

    def _spawn(self, name: str, command: list[str]) -> None:
        log = open(self.log_dir / f"{name}.log", "w")
        self.processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=self.env,
                                               start_new_session=True))

    def _image_callback(self, view: str):
        def callback(message):
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.images[view] = (stamp, message)
        return callback

    def _call(self, client, request, timeout: float = 30.0):
        future = client.call_async(request)
        deadline = wallclock.monotonic() + timeout
        while not future.done():
            self.rclpy.spin_once(self.node, timeout_sec=0.02)
            if wallclock.monotonic() > deadline:
                raise RuntimeError("Replay Gazebo stopped answering")
        return future.result()

    def place(self, name: str, position_enu, quaternion_wxyz) -> None:
        request = self.SetEntityPose.Request()
        request.entity.name = name
        request.entity.type = 2  # MODEL
        pose = request.pose
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position_enu)
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (float(v) for v in quaternion_wxyz)
        self._call(self.set_pose, request)

    def advance(self, need_images: bool = True, image_wait_s: float = 10.0) -> dict:
        request = self.ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = STEPS_PER_FRAME
        self._call(self.control, request)
        self.time = round(self.time + STEPS_PER_FRAME * PHYSICS_STEP_S, 6)
        if need_images:
            deadline = wallclock.monotonic() + image_wait_s
            while not all(view in self.images and self.images[view][0] >= self.time - 1e-6 for view in self.views):
                self.rclpy.spin_once(self.node, timeout_sec=0.02)
                if wallclock.monotonic() > deadline:
                    break  # reuse the previous image for a late frame
        return {view: self.images.get(view, (None, None))[1] for view in self.views}

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        for process in reversed(self.processes):
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


def _quaternion_enu(euler_ned) -> tuple:
    return matrix_to_quaternion(ENU_NED @ eul2rotm_zyx(euler_ned) @ FLU_FRD)


def _to_pil(message):
    from PIL import Image
    array = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
    array = array[:, : message.width * 3].reshape(message.height, message.width, 3)
    return Image.fromarray(array.copy(), "RGB")


def _hud(image, t: float, track: Track, summary: dict, final: bool):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(18, height // 30))
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(14, height // 42))
    t = min(t, float(track.time[-1]))
    k = int(np.clip(np.searchsorted(track.time, t, side="right") - 1, 0, len(track.time) - 1))
    draw.rectangle((10, 10, 10 + int(width * 0.5), 26 + 2 * big.size + 2 * small.size + 30), fill=(0, 0, 0, 140))
    start = summary.get("start_ned")
    where = f"start ({start[0]:g}, {start[1]:g})  |  " if start else ""
    draw.text((20, 18), f"{summary.get('scenario')}  |  {where}seed {summary.get('seed')}  |  {summary.get('controller')}",
              font=small, fill=(255, 255, 255))
    draw.text((20, 26 + small.size), f"t = {t:5.1f} s", font=big, fill=(255, 255, 255))
    healthy = track.lidar[k] > 0.5
    y = 34 + small.size + big.size
    draw.text((20, y), "LIDAR OK" if healthy else "LIDAR OUTAGE - CNN reconstruction", font=big,
              fill=(40, 200, 70) if healthy else (255, 140, 0))
    clearance = track.clearance[k]
    draw.text((20, y + big.size + 8), f"wind {track.wind[k]:4.1f} m/s   clearance {clearance:5.2f} m",
              font=small, fill=(255, 90, 90) if clearance < 1.0 else (255, 255, 255))
    _minimap(draw, image.size, track, k)
    if final:
        passed = bool(summary.get("mission_success"))
        clearance_min = summary.get("minimum_clearance_m")
        text = ("PASS" if passed else "FAIL") + f"  -  {summary.get('termination_reason')}"
        detail = (f"min clearance {clearance_min:.2f} m (needs 1.0)   " if clearance_min is not None else "") + \
                 f"end {summary.get('arrival_time_s')} s"
        box_w = int(width * 0.7)
        x0 = (width - box_w) // 2
        y0 = height // 2 - big.size * 2
        draw.rectangle((x0, y0, x0 + box_w, y0 + big.size * 4), fill=(0, 0, 0, 190))
        draw.text((x0 + 20, y0 + 15), text, font=big, fill=(60, 220, 90) if passed else (255, 80, 80))
        draw.text((x0 + 20, y0 + 25 + big.size), detail, font=small, fill=(255, 255, 255))


def _minimap(draw, size, track: Track, k: int) -> None:
    width, height = size
    box = min(width, height) // 3
    x0, y0 = width - box - 15, height - box - 15
    draw.rectangle((x0, y0, x0 + box, y0 + box), fill=(255, 255, 255, 200), outline=(0, 0, 0))

    def to_px(north, east):
        return x0 + (east + 5.0) / 45.0 * box, y0 + box - (north + 5.0) / 45.0 * box

    scene = scene_map()
    for lo, hi in zip(scene.box_min, scene.box_max):
        a, b = to_px(hi[0], lo[1]), to_px(lo[0], hi[1])
        draw.rectangle((a[0], a[1], b[0], b[1]), fill=(120, 120, 125))
    step = max(1, len(track.time) // 600)
    previous = 0
    for j in range(step, k + 1, step):
        color = (30, 170, 60) if track.lidar[j] > 0.5 else (255, 130, 0)
        draw.line((*to_px(*track.ned[previous, :2]), *to_px(*track.ned[j, :2])), fill=color, width=3)
        previous = j
    px, py = to_px(*track.ned[k, :2])
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(20, 60, 220))


def render_replay(run_dir: Path, views: list[str] | None = None, speed: float = 1.0, keep_frames: bool = False,
                  copy_to: Path | None = None, verbose: bool = True, gui: bool = False) -> list[Path]:
    """Render ``replay_<view>.mp4`` for a finished run; returns the video paths."""
    run_dir = Path(run_dir)
    views = views or ["overview", "top"]
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    track = load_track(run_dir)
    work = Path(tempfile.mkdtemp(prefix="ares_replay_"))
    world = work / "replay.sdf"
    world.write_text(replay_world(views, manifest["options"]["GoalState"][0:3]))
    duration = float(track.time[-1])
    hold_frames = 2 * FPS  # result banner
    frames = int(math.ceil(duration / speed * FPS)) + 1
    if verbose:
        print(f"Replaying {run_dir.name}: {duration:.1f} s flight -> {(frames + hold_frames) / FPS:.1f} s video, views {views}")
    for view in views:
        (work / view).mkdir()
    session = ReplaySession(world, views, work / "logs", gui=gui)
    started = wallclock.monotonic()
    try:
        last_images = {}
        for index in range(frames + hold_frames):
            t = min(index / FPS * speed, duration)
            ned, euler, cnn = _pose_at(track, t)
            position = (ned[1], ned[0], -ned[2])
            quaternion = _quaternion_enu(euler)
            session.place("drone_cnn" if cnn else "drone_lidar_ok", position, quaternion)
            session.place("drone_lidar_ok" if cnn else "drone_cnn", HIDDEN, (1.0, 0.0, 0.0, 0.0))
            images = session.advance()
            for view in views:
                message = images.get(view) or last_images.get(view)
                if message is None:
                    continue
                last_images[view] = message
                image = _to_pil(message)
                _hud(image, t, track, summary, final=index >= frames)
                image.save(work / view / f"{index:06d}.png", compress_level=1)
            if verbose and index % (10 * FPS) == 0:
                print(f"  frame {index}/{frames + hold_frames}  ({wallclock.monotonic() - started:.0f} s)")
    finally:
        session.close()
    ffmpeg = shutil.which("ffmpeg")
    outputs = []
    for view in views:
        count = len(list((work / view).glob("*.png")))
        if count == 0:
            raise RuntimeError(f"No frames rendered for {view}; see {work / 'logs'}")
        if ffmpeg is None:
            target = run_dir / f"replay_{view}_frames"
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(work / view, target)
            print(f"  ffmpeg missing: saved {count} frames to {target}. Install it: sudo apt install -y ffmpeg")
            continue
        video = run_dir / f"replay_{view}.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(work / view / "%06d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video)], check=True)
        outputs.append(video)
        if copy_to is not None:
            success = summary.get("mission_success")
            folder = copy_to / video_category(summary.get("scenario", "")) / result_folder(success)
            folder.mkdir(parents=True, exist_ok=True)
            result = "PASS" if success else "FAIL"
            shutil.copy2(video, folder / f"{video_label(summary)}_{result}_{view}.mp4")
        if verbose:
            print(f"  {view}: {count} frames -> {video}")
    if not keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    return outputs


def watch_replay(run_dir: Path, speed: float = 1.0) -> None:
    """Open the Gazebo GUI and play the recorded flight at real speed (no recording)."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    track = load_track(run_dir)
    work = Path(tempfile.mkdtemp(prefix="ares_watch_"))
    world = work / "watch.sdf"
    world.write_text(replay_world([], manifest["options"]["GoalState"][0:3]))
    session = ReplaySession(world, [], work / "logs", gui=True)
    try:
        wallclock.sleep(3.0)  # let the GUI connect
        start = wallclock.monotonic()
        frames = int(math.ceil(float(track.time[-1]) / speed * FPS)) + 1
        for index in range(frames):
            t = min(index / FPS * speed, float(track.time[-1]))
            ned, euler, cnn = _pose_at(track, t)
            session.place("drone_cnn" if cnn else "drone_lidar_ok", (ned[1], ned[0], -ned[2]), _quaternion_enu(euler))
            session.place("drone_lidar_ok" if cnn else "drone_cnn", HIDDEN, (1.0, 0.0, 0.0, 0.0))
            session.advance(need_images=False)
            delay = start + (index + 1) / FPS - wallclock.monotonic()
            if delay > 0:
                wallclock.sleep(delay)  # pace to real time
        input("Replay finished. Press Enter to close Gazebo...")
    finally:
        session.close()
