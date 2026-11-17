"""Gazebo world for the three copies of the single-UAV scene."""

from __future__ import annotations

import math

from .mission import (
    DRONE_EAST_OFFSETS_M,
    DRONE_IDS,
    OBSTACLE_HEIGHT_M,
    OBSTACLE_WIDTH_M,
    unique_obstacle_centers_enu,
)


def _box(name: str, x: float, y: float, z: float, sx: float, sy: float, sz: float) -> str:
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material>
            <ambient>0.65 0.65 0.65 1</ambient>
            <diffuse>0.65 0.65 0.65 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def _drone(name: str, x: float, y: float, z: float) -> str:
    # Initial NED yaw is zero, so the FLU nose points north: Gazebo yaw = pi/2.
    yaw = math.pi / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 {yaw:.8f}</pose>
      <link name="base_link">
        <visual name="body">
          <geometry><box><size>0.45 0.45 0.12</size></box></geometry>
          <material>
            <ambient>0.1 0.3 0.7 1</ambient>
            <diffuse>0.1 0.3 0.7 1</diffuse>
          </material>
        </visual>
        <sensor name="lidar" type="gpu_lidar">
          <pose>0 0 0.4 0 0 0</pose>
          <topic>{name}/lidar</topic>
          <update_rate>10</update_rate>
          <lidar>
            <scan>
              <horizontal>
                <samples>716</samples>
                <min_angle>{-math.radians(179):.8f}</min_angle>
                <max_angle>{math.radians(179):.8f}</max_angle>
              </horizontal>
              <vertical>
                <samples>15</samples>
                <min_angle>{-math.radians(15):.8f}</min_angle>
                <max_angle>{math.radians(15):.8f}</max_angle>
              </vertical>
            </scan>
            <range>
              <min>0.2</min>
              <max>7.0</max>
              <resolution>0.02</resolution>
            </range>
          </lidar>
          <always_on>1</always_on>
          <visualize>true</visualize>
        </sensor>
      </link>
    </model>"""


def live_world() -> str:
    """Buildings plus three drones at the single-UAV start pose."""
    models = [_ground()]
    centers = unique_obstacle_centers_enu()
    for drone_index, offset in enumerate(DRONE_EAST_OFFSETS_M):
        for center in centers:
            models.append(_box(
                f"building_{drone_index + 1}_{center[0]:.0f}_{center[1]:.0f}",
                center[0] + offset, center[1], OBSTACLE_HEIGHT_M / 2.0,
                OBSTACLE_WIDTH_M, OBSTACLE_WIDTH_M, OBSTACLE_HEIGHT_M,
            ))
        models.append(_drone(DRONE_IDS[drone_index], offset, 0.0, 7.0))
    return _wrap("ares_three_uav", "\n".join(models))


def playback_world(samples: list[tuple[str, list[tuple[float, float, float, float, float, float, float]]]]) -> str:
    """Actor playback of a completed guidance-plant trajectory.

    ``samples`` is ``(name, [(time, x, y, z, roll, pitch, yaw), ...])``.
    """
    actors = []
    for name, points in samples:
        waypoints = "\n".join(
            f"          <waypoint><time>{t:.3f}</time><pose>{x:.4f} {y:.4f} {z:.4f} {r:.5f} {p:.5f} {yaw:.5f}</pose></waypoint>"
            for t, x, y, z, r, p, yaw in points
        )
        actors.append(f"""
    <actor name="{name}_playback">
      <pose>0 0 0 0 0 0</pose>
      <link name="body">
        <visual name="visual">
          <geometry><box><size>0.45 0.45 0.12</size></box></geometry>
          <material><ambient>0.8 0.3 0.1 1</ambient><diffuse>0.8 0.3 0.1 1</diffuse></material>
        </visual>
      </link>
      <script>
        <loop>false</loop>
        <delay_start>0</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="flight">
{waypoints}
        </trajectory>
      </script>
    </actor>""")
    centers = unique_obstacle_centers_enu()
    buildings = [_ground()]
    for drone_index, offset in enumerate(DRONE_EAST_OFFSETS_M):
        for center in centers:
            buildings.append(_box(
                f"building_{drone_index + 1}_{center[0]:.0f}_{center[1]:.0f}",
                center[0] + offset, center[1], OBSTACLE_HEIGHT_M / 2.0,
                OBSTACLE_WIDTH_M, OBSTACLE_WIDTH_M, OBSTACLE_HEIGHT_M,
            ))
    return _wrap("ares_three_uav_playback", "\n".join(buildings + actors))


def _ground() -> str:
    return """
    <model name="ground">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry>
          <material><ambient>0.35 0.4 0.3 1</ambient><diffuse>0.35 0.4 0.3 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def _wrap(name: str, body: str) -> str:
    return f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="{name}">
    <physics name="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>-0.3 0.2 -0.9</direction>
    </light>
{body}
  </world>
</sdf>
"""
