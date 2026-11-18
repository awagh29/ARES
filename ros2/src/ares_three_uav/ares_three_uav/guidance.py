"""Multirotor guidance plant and the Simulink acceleration chart.

The 13-state order matches ``ares_safe_training.dynamics``: NED position,
NED velocity, roll/pitch/yaw, body rates p/q/r, and actual thrust. RK4
integrates the first twelve states at 1 ms. Thrust uses the same analytic
first-order solution, because a 3900/s lag is not stable under RK4 at 1 ms.
Roll and pitch are clamped to +/-25 degrees inside the original chart.
"""

from __future__ import annotations

import math

import numpy as np

from .mission import (
    ATTITUDE_LIMIT_RAD,
    GRAVITY_MPS2,
    MASS_KG,
    PITCH_KD,
    PITCH_KP,
    ROLL_KD,
    ROLL_KP,
    THRUST_GAIN,
    YAW_RATE_GAIN,
)


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def acceleration_chart(acceleration, yaw: float) -> tuple[float, float, float]:
    """``accControl`` in Controller/Attitude Control, including the 25 degree clamps."""
    north, east, down = (float(acceleration[0]), float(acceleration[1]), float(acceleration[2]))
    vertical = -down + GRAVITY_MPS2
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    pitch = math.atan2((-north * cosine - east * sine), vertical)
    pitch = min(ATTITUDE_LIMIT_RAD, max(-ATTITUDE_LIMIT_RAD, pitch))
    roll = math.atan2(((-north * sine + east * cosine) * math.cos(pitch)), vertical)
    roll = min(ATTITUDE_LIMIT_RAD, max(-ATTITUDE_LIMIT_RAD, roll))
    thrust = MASS_KG * vertical / (math.cos(pitch) * math.cos(roll))
    return roll, pitch, thrust


def _derivative12(state: np.ndarray, thrust: float, command: np.ndarray) -> np.ndarray:
    velocity = state[3:6]
    roll, pitch, yaw = state[6], state[7], state[8]
    sin_roll, sin_pitch, sin_yaw = math.sin(roll), math.sin(pitch), math.sin(yaw)
    cos_roll, cos_pitch, cos_yaw = math.cos(roll), math.cos(pitch), math.cos(yaw)
    p, q, r = state[9], state[10], state[11]
    thrust_mass = thrust / MASS_KG
    acceleration = np.array((
        -thrust_mass * (cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll),
        -thrust_mass * (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll),
        GRAVITY_MPS2 - thrust_mass * cos_pitch * cos_roll,
    ))
    common = sin_roll * q + cos_roll * r
    euler_rate = np.array((
        p + (sin_pitch / cos_pitch) * common,
        cos_roll * q - sin_roll * r,
        common / cos_pitch,
    ))
    roll_acceleration = ROLL_KP * (command[0] - roll) - ROLL_KD * euler_rate[0]
    pitch_acceleration = PITCH_KP * (command[1] - pitch) - PITCH_KD * euler_rate[1]
    yaw_acceleration = YAW_RATE_GAIN * (command[2] - euler_rate[2])
    body_rate = np.array((
        roll_acceleration - sin_pitch * yaw_acceleration,
        cos_roll * pitch_acceleration + sin_roll * cos_pitch * yaw_acceleration,
        -sin_roll * pitch_acceleration + cos_roll * cos_pitch * yaw_acceleration,
    ))
    return np.concatenate((velocity, acceleration, euler_rate, body_rate))


def rk4_step(state: np.ndarray, command: np.ndarray, dt: float) -> np.ndarray:
    half_decay = math.exp(-THRUST_GAIN * dt * 0.5)
    full_decay = half_decay * half_decay
    thrust = float(state[12])
    commanded = float(command[3])
    delta = thrust - commanded
    half_thrust = commanded + delta * half_decay
    next_thrust = commanded + delta * full_decay

    def shifted(derivative, scale):
        nxt = np.array(state, dtype=float, copy=True)
        nxt[:12] = state[:12] + scale * derivative
        return nxt

    first = _derivative12(state, thrust, command)
    second = _derivative12(shifted(first, 0.5 * dt), half_thrust, command)
    third = _derivative12(shifted(second, 0.5 * dt), half_thrust, command)
    fourth = _derivative12(shifted(third, dt), next_thrust, command)
    nxt = shifted(first + 2.0 * second + 2.0 * third + fourth, dt / 6.0)
    nxt[12] = next_thrust
    return nxt


def initial_state(position_ned) -> np.ndarray:
    state = np.zeros(13)
    state[:3] = position_ned
    return state
