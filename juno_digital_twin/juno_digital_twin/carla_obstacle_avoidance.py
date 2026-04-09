#!/usr/bin/env python3
"""
Bridge between Nav2 MPPI controller for obstacles and the vehicle's steering/throttle.

Nav2 outputs:  geometry_msgs/Twist on /cmd_vel  (m/s, rad/s)
Vehicle needs: Float64 on commands/KalmanAngle  (degrees, ±10 max)
               Float32 on commands/speed         (km/h threshold)
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float32
from shared_objects.ROS_utils import Topics
import numpy as np
from nav_msgs.msg import OccupancyGrid
MAX_STEER_DEG = 10.0
WHEELBASE = 1.6
MS_TO_KMH = 3.6


class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')

        topics = Topics()
        tn = topics.topic_names

        # Publishers
        self.steer_pub = self.create_publisher(Float64, tn["steering"], 1)
        self.speed_pub = self.create_publisher(Float32, tn["requested_speed"], 1)

        # Nav2 controller_server publishes cmd_vel here by default
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 1)

        self.active_pp = False #Pure pursuit active boolean
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_check, 1)
        self.get_logger().info(f"Obstacle avoidance bridge started: /cmd_vel -> {tn['steering']} + {tn['requested_speed']}")

    def costmap_check(self, msg):
        self.active_pp = bool(np.any(np.array(msg.data, dtype=np.int8) > 0))

    def cmd_vel_callback(self, msg: Twist):
        if not self.active_pp:
            return
        v = msg.linear.x       # m/s
        omega = msg.angular.z   # rad/s

        # Ackermann: steering_angle = atan(wheelbase * omega / v)
        if abs(v) < 0.01:
            steering_deg = 0.0
        else:
            steering_rad = math.atan(WHEELBASE * omega / v)
            steering_deg = math.degrees(steering_rad)

        # comply to physical limits
        steering_deg = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steering_deg))

        steer_msg = Float64()
        steer_msg.data = steering_deg
        self.steer_pub.publish(steer_msg)

        # Convert velocity to km/h speed threshold for throttle node
        speed_kmh = abs(v) * MS_TO_KMH
        speed_msg = Float32()
        speed_msg.data = float(speed_kmh)
        self.speed_pub.publish(speed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down obstacle_avoidance_node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
