#!/usr/bin/env python3
"""Bridge /cmd_vel -> vehicle steer/throttle with hardware-timed rate limits."""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float32
from shared_objects.ROS_utils import Topics

MAX_STEER_DEG = 10.0
WHEELBASE = 1.6
MS_TO_KMH = 3.6
MIN_V_FOR_STEER = 0.5 # In m/s
STEER_PERIOD_S = 0.1
SPEED_PERIOD_S = 1.0
CMD_VEL_STALE_S = 0.5


class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__("obstacle_avoidance_node")

        topics = Topics()
        tn = topics.topic_names

        self.steer_pub = self.create_publisher(Float64, tn["steering"], 1)
        self.speed_pub = self.create_publisher(Float32, tn["requested_speed"], 1)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 1)

        self._steer_target_deg = 0.0
        self._speed_kmh = 0.0
        self._last_cmd_time = None

        self.create_timer(STEER_PERIOD_S, self._steer_timer)
        self.create_timer(SPEED_PERIOD_S, self._speed_timer)

        self.get_logger().info(
            f"Bridge: /cmd_vel -> {tn['steering']} @ {1/STEER_PERIOD_S:.1f}Hz, "
            f"{tn['requested_speed']} @ {1/SPEED_PERIOD_S:.1f}Hz"
        )

    def _cmd_vel_cb(self, msg: Twist):
        v = msg.linear.x
        omega = msg.angular.z
        self.get_logger().warn(f'Received linear x velocity {v*MS_TO_KMH:.2f}km/h, Received angular z velocity {omega*57.3:.2f}°/s')

        if abs(v) < MIN_V_FOR_STEER:
            steer_deg = 0.0
        else:
            steer_deg = math.degrees(math.atan(WHEELBASE * omega / v))
        steer_deg = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steer_deg))

        # CARLA assumes left is - and right is + 
        # but ROS2 standard controller assumes the opposite(also opposite to our base_path_planning
        # which is fine with CARLA but not on ROS conventions)
        # Thats why I'm putting a minus when we receive the steering angle from Nav2 controller
        self._steer_target_deg = -steer_deg
        self._speed_kmh = abs(v) * MS_TO_KMH
        self._last_cmd_time = self.get_clock().now().nanoseconds * 1e-9

    def _cmd_fresh(self) -> bool:
        if self._last_cmd_time is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self._last_cmd_time) <= CMD_VEL_STALE_S

    def _steer_timer(self):
        if not self._cmd_fresh():
            return
        m = Float64()
        m.data = float(self._steer_target_deg)
        self.steer_pub.publish(m)
        self.get_logger().info(f"Sent Command: Steering degree raw: {self._steer_target_deg} ")

    def _speed_timer(self):
        if not self._cmd_fresh():
            return
        m = Float32()
        m.data = float(self._speed_kmh)
        self.speed_pub.publish(m)
        self.get_logger().info(f"Sent Command: Speed{self._speed_kmh} km/h")

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down obstacle_avoidance_node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
