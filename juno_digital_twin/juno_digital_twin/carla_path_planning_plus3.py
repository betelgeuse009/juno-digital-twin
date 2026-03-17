
#!/usr/bin/env python3
import pathlib as Path
import os
import math
import numpy as np
import cv2
from datetime import datetime
from rclpy.node import Node
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from shared_objects.utils_path import computing_lateral_distance, processing_mask
from shared_objects.ROS_utils import Topics, SHOW

class PathPlanningPlus2Node(Node):
    def __init__(self):
        super().__init__('carla_path_planning_plus3')
        # Parameters
        self.debug = self.declare_parameter('debug', True).value
        self.simulation = self.declare_parameter('simulation', False).value
        self.bridge = CvBridge()
        self.topics = Topics()
        self.topic_names = self.topics.topic_names

        self.wheelbase = 1.6
        self.gain = self.declare_parameter('gain', 0.0).value
        self.counter = 0

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, self.topic_names["goal"], 10)
        self.bev_pub = self.create_publisher(Image, "/birds_eye_view", 10)

        # Subscribers
        self.create_subscription(Image, self.topic_names["segmented_image"], self.image_callback, 10)
        self.original_image_sub = self.create_subscription(
            Image, '/carla/hero/rgb_front/image', self.original_image_callback, 10)
        
        # Debug folders
        self.declare_parameter(
            'debug_root',
            '/home/ubuntu/Workspace/ros-bridge/src/DEBUG'
        )
        self.debug_root = Path(
            self.get_parameter('debug_root').get_parameter_value().string_value
        )

        if self.DEBUG:
            self.logs_folder, self.output_folder, self.frames_folder = self.set_debug_folders()

        # Initial speed
        req_speed_msg = Float64()
        req_speed_msg.data = self.speed
        self.req_speed_pub.publish(req_speed_msg)


        self.get_logger().info("Path Planning Plus2 Node Initialized")

    def set_debug_folders(self):
        try:
            self.debug_root.mkdir(parents=True, exist_ok=True)

            ts_folder = self.debug_root / datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            ts_folder.mkdir()

            logs = ts_folder / 'logs'
            out = ts_folder / 'output'
            frames = ts_folder / 'frames'
            for p in (logs, out, frames):
                p.mkdir()

            self.get_logger().info(
                f'DEBUG output ->\n  logs:   {logs}\n  output: {out}\n  frames: {frames}'
            )
            return str(logs), str(out), str(frames)

        except Exception as e:
            self.get_logger().error(f'Failed to create debug folders: {e}')
            raise
    def original_image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

    def image_callback(self, data):
        mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        if self.cv_image is None:
            self.get_logger().warning("Image not received yet from original_image_callback.")
            return
        
        line_edges = processing_mask(mask, self.cv_image, show=False)

        birds_eye_view_msg = self.bridge.cv2_to_imgmsg(line_edges, encoding="mono8")
        self.bev_pub.publish(birds_eye_view_msg)

        lateral_distance, longitudinal_distance, curvature, midpoints = computing_lateral_distance(line_edges, show=False)
        
        if lateral_distance == -np.inf:
            degree_steering_angle = -10.0
        elif lateral_distance == np.inf:
            degree_steering_angle = 10.0
        else:
            distance_to_waypoint = (longitudinal_distance + self.gain) ** 2 + lateral_distance ** 2
            degree_steering_angle = math.degrees(math.atan2(2 * self.wheelbase * lateral_distance, distance_to_waypoint))

        if self.debug:
            resized_image = cv2.resize(self.cv_image, (540, 360))
            resized_mask = cv2.resize(mask, (540, 360))
            resized_line_edges = cv2.resize(line_edges, (540, 360))
            concatenated_image = np.hstack((
                cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(resized_line_edges, cv2.COLOR_GRAY2BGR)
            ))

            frame_name = f"frame_{self.counter}.png"
            frame_path = os.path.join(self.frames_folder, frame_name)
            cv2.imwrite(frame_path, self.cv_image)

            output_name = f"output_{self.counter}.png"
            output_path = os.path.join(self.output_folder, output_name)
            cv2.imwrite(output_path, concatenated_image)

            log_file = os.path.join(self.logs_folder, f"log_{self.counter}.txt")
            with open(log_file, "w") as log:
                log.write(f"{self.counter}: Curvature: {curvature} - Longitudinal Distance: {longitudinal_distance} - Degree Steering Angle: {degree_steering_angle}\n")
            self.counter += 1

        # Publish goal message
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "camera_link" # normally the name is zed_camera_link irl
        goal_msg.pose.position.x = longitudinal_distance
        goal_msg.pose.position.y = -lateral_distance
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)

def main(args=None):
    
    rclpy.init(args=args)
    node = PathPlanningPlus2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Path Planning Plus2 Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


