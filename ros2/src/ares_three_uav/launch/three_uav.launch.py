"""Start the Gazebo scene and the three-vehicle guidance node."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("ares_three_uav")
    # Playback is the flight already produced by the guidance node. The live
    # scene with lidar models is ares_three_uav.sdf in the same folder.
    world = os.path.join(share, "worlds", "ares_three_uav_playback.sdf")
    return LaunchDescription([
        ExecuteProcess(
            cmd=["gz", "sim", "-r", world],
            output="screen",
        ),
        Node(
            package="ares_three_uav",
            executable="ares_three_uav_node",
            name="ares_three_uav",
            output="screen",
        ),
    ])
