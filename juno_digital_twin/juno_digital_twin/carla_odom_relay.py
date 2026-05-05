#!/usr/bin/env python3
"""
CARLA Odometry Relay Node

Problem: CARLA ros-bridge publishes odometry in the 'map' frame using
absolute world coordinates. The 'map' frame sits at CARLA world origin (0,0,0),
so the vehicle appears hundreds of meters from the map frame in RViz.

Solution: This node captures the vehicle's initial pose as the origin,
then publishes relative odometry in a new 'odom' frame. This gives us the
REP-105 compliant TF chain:  map -> odom -> hero

The node:
  1. Subscribes to CARLA's /carla/hero/odometry (absolute, frame_id='map')
  2. Records the first pose as the origin
  3. Publishes relative odometry on /odom with frame_id='odom', child_frame_id='hero'
  4. Broadcasts the odom -> hero TF transform
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation


class CarlaOdomRelay(Node):
    def __init__(self):
        super().__init__('carla_odom_relay')

        # Parameters
        self.declare_parameter('input_odom_topic', '/carla/hero/odometry')
        self.declare_parameter('output_odom_topic', '/odom')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'hero')

        input_topic = self.get_parameter('input_odom_topic').value
        output_topic = self.get_parameter('output_odom_topic').value
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.child_frame = self.get_parameter('child_frame_id').value

        # State
        self.initial_pose = None  # Will be set on first message
        self.initial_rotation_inv = None

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publisher
        self.odom_pub = self.create_publisher(Odometry, output_topic, 30)

        # Subscriber — use BEST_EFFORT to match CARLA's QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            Odometry, input_topic, self.odom_callback, qos
        )

        self.get_logger().info(
            f'Relaying {input_topic} -> {output_topic} '
            f'(frame: {self.odom_frame} -> {self.child_frame})'
        )

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # Current absolute pose
        abs_position = np.array([pos.x, pos.y, pos.z])
        abs_rotation = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w])

        # Capture initial pose on first message
        if self.initial_pose is None:
            self.initial_pose = abs_position.copy()
            initial_yaw = abs_rotation.as_euler('xyz')[2]
            self.initial_rotation_inv = Rotation.from_euler('z', initial_yaw).inv()
            self.get_logger().info(
                f'Initial pose captured: position=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})'
                f'yaw={np.degrees(initial_yaw):.2f}°'
            )

        # Compute relative pose: subtract initial position, rotate into initial frame
        rel_position = self.initial_rotation_inv.apply(abs_position - self.initial_pose)
        rel_rotation = self.initial_rotation_inv * abs_rotation
        rel_quat = rel_rotation.as_quat()  # [x, y, z, w]

        # Also transform linear velocity into the relative frame
        abs_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])
        rel_vel = self.initial_rotation_inv.apply(abs_vel)

        # Build output odometry message
        odom_out = Odometry()
        odom_out.header.stamp = msg.header.stamp
        odom_out.header.frame_id = self.odom_frame
        odom_out.child_frame_id = self.child_frame

        odom_out.pose.pose.position.x = rel_position[0]
        odom_out.pose.pose.position.y = rel_position[1]
        odom_out.pose.pose.position.z = rel_position[2]
        odom_out.pose.pose.orientation.x = rel_quat[0]
        odom_out.pose.pose.orientation.y = rel_quat[1]
        odom_out.pose.pose.orientation.z = rel_quat[2]
        odom_out.pose.pose.orientation.w = rel_quat[3]

        odom_out.twist.twist.linear.x = rel_vel[0]
        odom_out.twist.twist.linear.y = rel_vel[1]
        odom_out.twist.twist.linear.z = rel_vel[2]
        odom_out.twist.twist.angular = msg.twist.twist.angular

        # Copy covariances
        odom_out.pose.covariance = msg.pose.covariance
        odom_out.twist.covariance = msg.twist.covariance

        self.odom_pub.publish(odom_out)

        # Broadcast odom -> hero TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.child_frame
        tf_msg.transform.translation.x = rel_position[0]
        tf_msg.transform.translation.y = rel_position[1]
        tf_msg.transform.translation.z = rel_position[2]
        tf_msg.transform.rotation.x = rel_quat[0]
        tf_msg.transform.rotation.y = rel_quat[1]
        tf_msg.transform.rotation.z = rel_quat[2]
        tf_msg.transform.rotation.w = rel_quat[3]

        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaOdomRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()