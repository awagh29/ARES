"""Generate the Gazebo world: ground, the ARES buildings and one quadcopter.

Buildings come from ``scripts/initializeSingleUav.m`` via :mod:`ares_gz.scene`,
so Gazebo and MATLAB fly the same map. The quadcopter is a mesh-free copy of
Gazebo's X3 example vehicle (1.5 kg, four rotors driven by the
MulticopterMotorModel system, MulticopterVelocityControl as its flight
controller), carrying a lidar configured like the MATLAB one:
15 x 716 beams, +-15 deg vertical, +-179 deg horizontal, 7 m range, 10 Hz,
mounted 0.4 m above the body centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .frames import ned_euler_to_gazebo_rpy, ned_to_enu
from .scene import SceneMap

WORLD_NAME = "ares_test"
MODEL_NAME = "ares_quad"


@dataclass(frozen=True)
class QuadParameters:
    """X3 example vehicle (gz-sim ``multicopter_velocity_control`` world)."""

    mass_kg: float = 1.5
    inertia: tuple = (0.0347563, 0.07, 0.0977)
    arm_positions: tuple = ((0.13, -0.22, 0.023), (-0.13, 0.20, 0.023), (0.13, 0.22, 0.023), (-0.13, -0.20, 0.023))
    rotor_directions: tuple = ("ccw", "ccw", "cw", "cw")
    motor_constant: float = 8.54858e-06
    moment_constant: float = 0.016
    max_rotor_velocity: float = 800.0
    time_constant_up: float = 0.0125
    time_constant_down: float = 0.025
    rotor_drag_coefficient: float = 8.06428e-05
    rolling_moment_coefficient: float = 1e-06
    # Velocity controller gains. The commanded acceleration is gain/mass * velocity error.
    velocity_gain: tuple = (2.7, 2.7, 2.7)
    attitude_gain: tuple = (2.0, 3.0, 0.15)
    angular_rate_gain: tuple = (0.4, 0.52, 0.18)
    maximum_linear_acceleration: tuple = (2.0, 2.0, 2.0)


def _vector(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _building(index: int, center_enu: np.ndarray, width: float, height: float) -> str:
    size = f"{width:.6g} {width:.6g} {height:.6g}"
    return f"""
    <model name="building_{index}">
      <static>true</static>
      <pose>{center_enu[0]:.6f} {center_enu[1]:.6f} {height / 2:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>{size}</size></box></geometry>
          <material><ambient>0.65 0.65 0.65 1</ambient><diffuse>0.65 0.65 0.65 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def _rotor(index: int, position, quad: QuadParameters) -> str:
    return f"""
      <link name="rotor_{index}">
        <pose>{_vector(position)} 0 0 0</pose>
        <inertial>
          <mass>0.005</mass>
          <inertia><ixx>9.75e-07</ixx><iyy>4.17e-05</iyy><izz>4.26e-05</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <geometry><cylinder><radius>0.1</radius><length>0.005</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material>
        </visual>
      </link>
      <joint name="rotor_{index}_joint" type="revolute">
        <child>rotor_{index}</child>
        <parent>base_link</parent>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>
      <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
        <robotNamespace>{MODEL_NAME}</robotNamespace>
        <jointName>rotor_{index}_joint</jointName>
        <linkName>rotor_{index}</linkName>
        <turningDirection>{quad.rotor_directions[index]}</turningDirection>
        <timeConstantUp>{quad.time_constant_up}</timeConstantUp>
        <timeConstantDown>{quad.time_constant_down}</timeConstantDown>
        <maxRotVelocity>{quad.max_rotor_velocity}</maxRotVelocity>
        <motorConstant>{quad.motor_constant}</motorConstant>
        <momentConstant>{quad.moment_constant}</momentConstant>
        <commandSubTopic>command/motor_speed</commandSubTopic>
        <motorNumber>{index}</motorNumber>
        <rotorDragCoefficient>{quad.rotor_drag_coefficient}</rotorDragCoefficient>
        <rollingMomentCoefficient>{quad.rolling_moment_coefficient}</rollingMomentCoefficient>
        <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
        <motorType>velocity</motorType>
      </plugin>"""


def quad_model(start_ned, euler_zyx, quad: QuadParameters = QuadParameters()) -> str:
    x, y, z = ned_to_enu(start_ned)
    roll, pitch, yaw = ned_euler_to_gazebo_rpy(euler_zyx)
    ixx, iyy, izz = quad.inertia
    rotors = "".join(_rotor(index, position, quad) for index, position in enumerate(quad.arm_positions))
    rotor_config = "".join(
        f"""
          <rotor>
            <jointName>rotor_{index}_joint</jointName>
            <forceConstant>{quad.motor_constant}</forceConstant>
            <momentConstant>{quad.moment_constant}</momentConstant>
            <direction>{1 if quad.rotor_directions[index] == "ccw" else -1}</direction>
          </rotor>""" for index in range(4))
    vertical = 15
    horizontal = 716
    return f"""
    <model name="{MODEL_NAME}">
      <pose>{x:.6f} {y:.6f} {z:.6f} {roll:.9f} {pitch:.9f} {yaw:.9f}</pose>
      <link name="base_link">
        <inertial>
          <mass>{quad.mass_kg}</mass>
          <inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="collision"><geometry><box><size>0.47 0.47 0.11</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>0.47 0.47 0.11</size></box></geometry>
          <material><ambient>0.1 0.3 0.8 1</ambient><diffuse>0.1 0.3 0.8 1</diffuse></material>
        </visual>
        <sensor name="lidar" type="gpu_lidar">
          <pose>0 0 0.4 0 0 0</pose>
          <topic>/{MODEL_NAME}/lidar</topic>
          <update_rate>10</update_rate>
          <always_on>1</always_on>
          <visualize>false</visualize>
          <lidar>
            <scan>
              <horizontal><samples>{horizontal}</samples><resolution>1</resolution>
                <min_angle>{-math.radians(179):.10f}</min_angle><max_angle>{math.radians(179):.10f}</max_angle></horizontal>
              <vertical><samples>{vertical}</samples><resolution>1</resolution>
                <min_angle>{-math.radians(15):.10f}</min_angle><max_angle>{math.radians(15):.10f}</max_angle></vertical>
            </scan>
            <range><min>0.05</min><max>7.0</max><resolution>0.001</resolution></range>
          </lidar>
        </sensor>
      </link>{rotors}
      <plugin filename="gz-sim-multicopter-control-system" name="gz::sim::systems::MulticopterVelocityControl">
        <robotNamespace>{MODEL_NAME}</robotNamespace>
        <commandSubTopic>cmd_vel</commandSubTopic>
        <enableSubTopic>enable</enableSubTopic>
        <comLinkName>base_link</comLinkName>
        <velocityGain>{_vector(quad.velocity_gain)}</velocityGain>
        <attitudeGain>{_vector(quad.attitude_gain)}</attitudeGain>
        <angularRateGain>{_vector(quad.angular_rate_gain)}</angularRateGain>
        <maximumLinearAcceleration>{_vector(quad.maximum_linear_acceleration)}</maximumLinearAcceleration>
        <rotorConfiguration>{rotor_config}
        </rotorConfiguration>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <odom_frame>world</odom_frame>
        <robot_base_frame>{MODEL_NAME}</robot_base_frame>
        <odom_publish_frequency>1000</odom_publish_frequency>
        <dimensions>3</dimensions>
        <odom_topic>/{MODEL_NAME}/odometry</odom_topic>
      </plugin>
    </model>"""


def world_sdf(scene: SceneMap, start_ned, euler_zyx=(0.0, 0.0, 0.0), physics_step_s: float = 0.001,
              quad: QuadParameters = QuadParameters()) -> str:
    buildings = "".join(
        _building(index, center, scene.width_m, scene.height_m)
        for index, center in enumerate(scene.centers_enu))
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{WORLD_NAME}">
    <physics name="1ms" type="dart">
      <max_step_size>{physics_step_s}</max_step_size>
      <real_time_factor>0</real_time_factor>
    </physics>
    <gravity>0 0 -9.81</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 30 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>-0.3 0.2 -0.9</direction>
    </light>
    <model name="ground">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.3 0.35 0.3 1</ambient><diffuse>0.3 0.35 0.3 1</diffuse></material>
        </visual>
      </link>
    </model>{buildings}
{quad_model(start_ned, euler_zyx, quad)}
  </world>
</sdf>
"""
