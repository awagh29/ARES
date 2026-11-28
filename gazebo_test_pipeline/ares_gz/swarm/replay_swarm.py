"""Smooth MP4 replays of a three-drone run (same lock-step renderer as the single-drone replay).

Each drone has its own colour; while its LiDAR is down it is drawn in red.
The HUD lists every drone's LiDAR state, hub mode and task, the live minimum
separation, and the latest hub decision. The minimap shows trails, victims
(grey = not found, amber = reported, green = confirmed) and a 1.5 m circle
around each drone: the circles never overlap while the 3 m invariant holds.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
import time as wallclock
from pathlib import Path

import numpy as np

from ..replay import (FPS, HIDDEN, PHYSICS_STEP_S, VIEWS, WORLD, ReplaySession, _drone_model, _quaternion_enu,
                      _to_pil, result_folder, video_category)
from ..scene import scene_map

TEAM = ["0.15 0.35 0.95 1", "0.75 0.2 0.85 1", "0.1 0.75 0.3 1", "0.9 0.6 0.1 1"]
TEAM_RGB = [(40, 90, 240), (190, 50, 215), (25, 190, 75), (230, 150, 25)]
DOWN = "0.95 0.1 0.1 1"


def swarm_world(views: list[str], drones: list[dict], victims: list[dict]) -> str:
    scene = scene_map()
    buildings = "".join(f"""
    <model name="building_{k}"><static>true</static>
      <pose>{c[0]:.3f} {c[1]:.3f} {scene.height_m / 2:.3f} 0 0 0</pose>
      <link name="link"><visual name="v"><geometry><box><size>{scene.width_m} {scene.width_m} {scene.height_m}</size></box></geometry>
        <material><ambient>0.6 0.6 0.62 1</ambient><diffuse>0.7 0.7 0.72 1</diffuse></material></visual></link>
    </model>""" for k, c in enumerate(scene.centers_enu))
    pads = "".join(f"""
    <model name="slot_{d['id']}"><static>true</static><pose>{d['slot'][1]} {d['slot'][0]} 0.03 0 0 0</pose><link name="link">
      <visual name="v"><geometry><box><size>1.6 1.6 0.05</size></box></geometry>
      <material><ambient>{TEAM[k % 4]}</ambient><diffuse>{TEAM[k % 4]}</diffuse></material></visual></link></model>"""
                   for k, d in enumerate(drones))
    marks = "".join(f"""
    <model name="victim_{v['id']}"><static>true</static><pose>{v['east']} {v['north']} 0.15 0 0 0</pose><link name="link">
      <visual name="v"><geometry><cylinder><radius>0.5</radius><length>0.3</length></cylinder></geometry>
      <material><ambient>0.95 0.2 0.2 1</ambient><diffuse>0.95 0.2 0.2 1</diffuse><emissive>0.6 0 0 1</emissive></material></visual></link></model>"""
                    for v in victims)
    models = "".join(_drone_model(f"{d['id'].lower()}_ok", TEAM[k % 4]) + _drone_model(f"{d['id'].lower()}_down", DOWN)
                     for k, d in enumerate(drones))
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
{buildings}{pads}{marks}{models}
{''.join(cameras)}
  </world>
</sdf>
"""


class DroneTrack:
    def __init__(self, path: Path):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        f = lambda name: np.array([float(r[name]) if r.get(name) not in (None, "") else 0.0 for r in rows])  # noqa: E731
        self.time = f("time_s")
        self.ned = np.column_stack((f("n"), f("e"), f("d")))
        self.euler = np.column_stack((f("yaw"), f("pitch"), f("roll")))
        self.lidar = f("lidar_available")
        self.mode = [r.get("hub_mode") or r.get("agent_mode") or "" for r in rows]
        self.task = [r.get("task_kind") or "" for r in rows]

    def index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.time, t, side="right") - 1, 0, len(self.time) - 1))

    def pose(self, t: float):
        ned = np.array([np.interp(t, self.time, self.ned[:, k]) for k in range(3)])
        euler = np.array([np.interp(t, self.time, np.unwrap(self.euler[:, 0])),
                          np.interp(t, self.time, self.euler[:, 1]), np.interp(t, self.time, self.euler[:, 2])])
        return ned, euler


def _hud(image, t, tracks, drones, summary, events, victims_hub, final):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(18, height // 32))
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(13, height // 46))
    panel_h = 30 + big.size + (len(drones) + 2) * (small.size + 6)
    draw.rectangle((10, 10, 10 + int(width * 0.56), 10 + panel_h), fill=(0, 0, 0, 150))
    draw.text((20, 16), f"SWARM  {summary['scenario']}  |  seed {summary['seed']}  |  comms {summary.get('comms')}",
              font=small, fill=(255, 255, 255))
    draw.text((20, 20 + small.size), f"t = {t:5.1f} s", font=big, fill=(255, 255, 255))
    y = 26 + small.size + big.size
    positions = {}
    for k, drone in enumerate(drones):
        track = tracks[drone["id"]]
        i = track.index(t)
        positions[drone["id"]] = track.ned[i]
        ok = track.lidar[i] > 0.5
        text = f"{drone['id']}  {'LiDAR OK ' if ok else 'LiDAR DOWN'}  {track.mode[i]:<11}  {track.task[i]}"
        draw.text((20, y), text, font=small, fill=TEAM_RGB[k % 4] if ok else (255, 70, 70))
        y += small.size + 6
    ids = list(positions)
    separation = min((np.linalg.norm(positions[a] - positions[b]) for a in ids for b in ids if a < b), default=np.inf)
    need = summary.get("separation_required_m", 3.0)
    draw.text((20, y), f"min separation {separation:4.1f} m (rule {need:g} m)", font=small,
              fill=(255, 255, 255) if separation >= need else (255, 70, 70))
    recent = [e for e in events if e["time_s"] <= t + 1e-6]
    if recent:
        event = recent[-1]
        text = f"HUB t={event['time_s']:.1f}: " + (f"{event['drone']} " if event.get("drone") else "") + event["event"]
        draw.rectangle((10, height - 20 - small.size, width - 10, height - 8), fill=(0, 0, 0, 150))
        draw.text((18, height - 16 - small.size), text[:140], font=small, fill=(255, 230, 120))
    _minimap(draw, image.size, t, tracks, drones, victims_hub)
    if final:
        passed = bool(summary.get("mission_success"))
        box_w = int(width * 0.74)
        x0 = (width - box_w) // 2
        y0 = height // 2 - big.size * 2
        draw.rectangle((x0, y0, x0 + box_w, y0 + big.size * 4), fill=(0, 0, 0, 200))
        draw.text((x0 + 20, y0 + 12), ("PASS" if passed else "FAIL") + f"  -  {summary['termination_reason']}"[:80],
                  font=big, fill=(60, 220, 90) if passed else (255, 80, 80))
        detail = (f"min separation {summary['min_pairwise_distance_m']:.2f} m   coverage {summary['coverage']:.0%}   "
                  f"victims {summary['victims_confirmed']}/{summary['victims_total']} confirmed")
        draw.text((x0 + 20, y0 + 22 + big.size), detail, font=small, fill=(255, 255, 255))


def _minimap(draw, size, t, tracks, drones, victims_hub):
    width, height = size
    box = min(width, height) // 3
    x0, y0 = width - box - 15, height - box - 45

    def px(north, east):
        return x0 + (east + 5.0) / 45.0 * box, y0 + box - (north + 5.0) / 45.0 * box

    draw.rectangle((x0, y0, x0 + box, y0 + box), fill=(255, 255, 255, 210), outline=(0, 0, 0))
    scene = scene_map()
    for lo, hi in zip(scene.box_min, scene.box_max):
        a, b = px(hi[0], lo[1]), px(lo[0], hi[1])
        draw.rectangle((a[0], a[1], b[0], b[1]), fill=(120, 120, 125))
    for victim in victims_hub:
        status = victim.get("status_at", lambda _t: "none")(t)
        color = {"none": (150, 150, 150), "reported": (240, 170, 0), "confirmed": (20, 170, 60)}[status]
        cx, cy = px(victim["north"], victim["east"])
        draw.polygon([(cx, cy - 7), (cx + 6, cy + 5), (cx - 6, cy + 5)], fill=color, outline=(0, 0, 0))
    scale = box / 45.0
    for k, drone in enumerate(drones):
        track = tracks[drone["id"]]
        i = track.index(t)
        step = max(1, len(track.time) // 500)
        points = [px(*track.ned[j, :2]) for j in range(0, i + 1, step)]
        if len(points) > 1:
            draw.line(points, fill=TEAM_RGB[k % 4], width=2)
        cx, cy = px(*track.ned[i, :2])
        r = 1.5 * scale
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=TEAM_RGB[k % 4], width=2)
        color = TEAM_RGB[k % 4] if track.lidar[i] > 0.5 else (240, 40, 40)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)


def _victim_timelines(summary: dict, manifest: dict) -> list[dict]:
    out = []
    registry = {v.get("truth_id"): v for v in summary.get("hub", {}).get("victims", [])}
    for victim in manifest["victims"]:
        record = registry.get(victim["id"])
        reported = record["first_reported_s"] if record else None
        confirmed = record.get("confirmed_s") if record else None

        def status(t, reported=reported, confirmed=confirmed):
            if confirmed is not None and t >= confirmed:
                return "confirmed"
            if reported is not None and t >= reported:
                return "reported"
            return "none"
        out.append({**victim, "status_at": status})
    return out


def render_swarm_replay(run_dir: Path, views: list[str] | None = None, speed: float = 1.0, keep_frames: bool = False,
                        copy_to: Path | None = None, verbose: bool = True) -> list[Path]:
    run_dir = Path(run_dir)
    views = views or ["overview", "top"]
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = json.loads((run_dir / "hub_events.json").read_text()) if (run_dir / "hub_events.json").exists() else []
    drones = manifest["drones"]
    tracks = {d["id"]: DroneTrack(run_dir / "drones" / d["id"] / "control.csv") for d in drones}
    victims_hub = _victim_timelines(summary, manifest)
    work = Path(tempfile.mkdtemp(prefix="ares_swarm_replay_"))
    world = work / "replay.sdf"
    world.write_text(swarm_world(views, drones, manifest["victims"]))
    duration = min(float(track.time[-1]) for track in tracks.values())
    hold = 2 * FPS
    frames = int(math.ceil(duration / speed * FPS)) + 1
    if verbose:
        print(f"Replaying swarm {run_dir.name}: {duration:.1f} s -> {(frames + hold) / FPS:.1f} s video, views {views}")
    for view in views:
        (work / view).mkdir()
    session = ReplaySession(world, views, work / "logs")
    started = wallclock.monotonic()
    try:
        last = {}
        for index in range(frames + hold):
            t = min(index / FPS * speed, duration)
            for drone in drones:
                track = tracks[drone["id"]]
                ned, euler = track.pose(t)
                down = track.lidar[track.index(t)] < 0.5
                name = drone["id"].lower()
                session.place(f"{name}_down" if down else f"{name}_ok", (ned[1], ned[0], -ned[2]), _quaternion_enu(euler))
                session.place(f"{name}_ok" if down else f"{name}_down", HIDDEN, (1.0, 0.0, 0.0, 0.0))
            images = session.advance()
            for view in views:
                message = images.get(view) or last.get(view)
                if message is None:
                    continue
                last[view] = message
                image = _to_pil(message)
                _hud(image, t, tracks, drones, summary, events, victims_hub, final=index >= frames)
                image.save(work / view / f"{index:06d}.png", compress_level=1)
            if verbose and index % (10 * FPS) == 0:
                print(f"  frame {index}/{frames + hold}  ({wallclock.monotonic() - started:.0f} s)")
    finally:
        session.close()
    ffmpeg = shutil.which("ffmpeg")
    outputs = []
    for view in views:
        if not list((work / view).glob("*.png")):
            raise RuntimeError(f"No frames rendered for {view}; see {work / 'logs'}")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg missing: sudo apt install -y ffmpeg")
        video = run_dir / f"replay_{view}.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(work / view / "%06d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video)], check=True)
        outputs.append(video)
        if copy_to is not None:
            success = summary.get("mission_success")
            folder = Path(copy_to) / video_category(summary["scenario"]) / result_folder(success)
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, folder / f"{summary['scenario']}_seed{summary['seed']}_{'PASS' if success else 'FAIL'}_{view}.mp4")
        if verbose:
            print(f"  {view} -> {video}")
    if not keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    return outputs
