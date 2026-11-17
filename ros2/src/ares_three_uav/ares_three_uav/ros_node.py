"""ROS 2 node that runs the same three pilots and publishes their poses.

Gazebo stays the scene. The 1 ms guidance plant in this node is the dynamics
used by the Simulink model, so the trajectory is not re-integrated by a
different quadrotor plugin. Pose topics are ``/ares/<drone>/odometry``.
"""

from __future__ import annotations


def main() -> None:
    import rclpy
    from geometry_msgs.msg import Quaternion
    from nav_msgs.msg import Odometry
    from rclpy.node import Node

    from .frames import enu_orientation, ned_to_enu
    from .mission import PLANT_DT_S, STOP_TIME_S
    from .pilot import Pilot
    from .schedule import intermittent_schedule

    class ThreeUavFlight(Node):
        def __init__(self):
            super().__init__("ares_three_uav")
            times, enabled, intervals = intermittent_schedule(STOP_TIME_S)
            self.get_logger().info(f"LiDAR guidance outages (s): {intervals.tolist()}")
            self.pilots = [Pilot(index, times, enabled) for index in range(3)]
            self.publishers_ = [
                self.create_publisher(Odometry, f"/ares/{pilot.name}/odometry", 10)
                for pilot in self.pilots
            ]
            self.step_index = 0
            self.steps = int(round(STOP_TIME_S / PLANT_DT_S))
            self.substeps = int(round(0.01 / PLANT_DT_S))
            self.timer = self.create_timer(0.01, self._advance)

        def _advance(self):
            if self.step_index >= self.steps:
                return
            for _ in range(self.substeps):
                if self.step_index >= self.steps:
                    break
                for pilot in self.pilots:
                    pilot.step(self.step_index)
                self.step_index += 1
            stamp = self.get_clock().now().to_msg()
            for pilot, publisher in zip(self.pilots, self.publishers_):
                message = Odometry()
                message.header.stamp = stamp
                message.header.frame_id = "map"
                message.child_frame_id = f"{pilot.name}/base_link"
                north, east, down = pilot.position_ned
                x, y, z = ned_to_enu(float(north), float(east), float(down))
                _roll, _pitch, _yaw, quat = enu_orientation(
                    float(pilot.state[6]), float(pilot.state[7]), float(pilot.state[8])
                )
                message.pose.pose.position.x = x
                message.pose.pose.position.y = y
                message.pose.pose.position.z = z
                message.pose.pose.orientation = Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])
                publisher.publish(message)
            if all(pilot.finished for pilot in self.pilots):
                self.get_logger().info("All three vehicles reached the waypoint-follower stop condition.")
                self.timer.cancel()

    rclpy.init()
    node = ThreeUavFlight()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
