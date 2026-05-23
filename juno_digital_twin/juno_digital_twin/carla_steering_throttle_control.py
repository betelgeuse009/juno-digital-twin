#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64, Bool
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
import math
import numpy as np
MS_TO_KMH = 3.6
class CarlaSimBridge(Node):
    def __init__(self):
        super().__init__('carla_sim_bridge')
        self.declare_parameter('max_steer_deg', 40.0) # Physical limit of steering
        self.max_steer = self.get_parameter('max_steer_deg').get_parameter_value().double_value

    
        self.create_subscription(Float64, 'commands/KalmanAngle', self.steering_callback, 10)
        self.create_subscription(Float32, '/ECU/throttle', self.throttle_callback, 10)
        self.create_subscription(Bool, 'commands/stop', self.stop_callback, 10)

        # 3. Odometry (Speed feedback from CARLA)
        self.create_subscription(
            Odometry, 
            '/carla/hero/odometry', 
            self.odometry_callback, 
            10
        )

        # 1. Command to CARLA
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl, 
            '/carla/hero/vehicle_control_cmd', 
            10
        )
        
        # 2. Speed feedback (m/s) for your ThrottleNode
        self.speed_pub = self.create_publisher(Float32, '/ECU/speed', 10)

        # Internal State
        self.current_steer = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.stop = False

        # Control Loop (20Hz)
        self.create_timer(0.05, self.publish_control)

        self.get_logger().info(f"CARLA Bridge Initialized for role: our Hero (JUNO)")

    def steering_callback(self, msg):
        # In CARLA: -1.0 is Left, +1.0 is Right (Standard). 
        # so need to verify if path planner outputs positive for Left or Right.
        val = msg.data / self.max_steer 
        self.get_logger().warn(f"Received steering angle={val}={msg.data}/{self.max_steer}")
        self.current_steer = np.clip(val, -1.0, 1.0)

    def stop_callback(self, msg):
        self.stop = msg.data
        if self.stop:
            # Mirror Stepper.brake() from real vehicle: full brake, zero throttle
            self.current_throttle = 0.0
            self.current_brake = 1.0
        else:
            self.current_brake = 0.0

    def throttle_callback(self, msg):
        # Skip throttle/brake update while stop is active
        if self.stop:
            return

        # Map 0-50 to [0.0, 1.0]
        val = msg.data / 50.0
        self.current_throttle = np.clip(val, 0.0, 1.0)
        self.current_brake = 0.0

    def odometry_callback(self, msg):
        # Calculate speed in m/s
        vel = msg.twist.twist.linear
        speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_kmh = speed_ms * MS_TO_KMH
        # Publish to the topic your Throttle Node expects
        speed_msg = Float32()
        speed_msg.data = speed_kmh
        self.speed_pub.publish(speed_msg)

    def publish_control(self, _=None):
        msg = CarlaEgoVehicleControl()
        msg.throttle = float(self.current_throttle)
        msg.steer = float(self.current_steer)
        msg.brake = float(self.current_brake)
        msg.hand_brake = False
        msg.manual_gear_shift = False
        msg.gear = 1
        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CarlaSimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()