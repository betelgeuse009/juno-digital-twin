#!/usr/bin/env python3

import os
import math
import numpy as np
import cv2
from datetime import datetime

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from shared_objects.utils_path import computing_lateral_distance, processing_mask
from shared_objects.ROS_utils import Topics, SHOW
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64
import tf2_geometry_msgs
from pathlib import Path 
from rclpy.qos import qos_profile_sensor_data

class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_plus3')

        self.debug = self.declare_parameter('debug', True).value
        self.wheelbase = self.declare_parameter('wheelbase', 1.6).value
        self.gain = self.declare_parameter('gain', 0.0).value

        self.costmap_has_obstacles = False
        self.goal_published = False
        self.declare_parameter(
            'debug_root',
            '/home/ubuntu/Workspace/ros-bridge/src/DEBUG'   # default
        )
        self.debug_root = Path(
        self.get_parameter('debug_root').get_parameter_value().string_value
        )
      
        self.bridge = CvBridge()
        self.cv_image = None
        self.counter = 0

        topics = Topics()
        self.topic_names = topics.topic_names


        # TF for transforming goal from camera frame to map frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(Image, self.topic_names["segmented_image"],self.image_callback, 1)
        self.create_subscription(Image, '/carla/hero/rgb_front/image',self.original_image_callback, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, 1)
        
        
        # Publisher
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose',1)
        self.steer_pub = self.create_publisher(Float64, self.topic_names["steering"], 1)
        self.bev_pub = self.create_publisher(Image, "/birds_eye_view", 10)

        
        self._latest_goal = None
        self._goal_timer = self.create_timer(2.0, self._goal_timer_callback)
        

        if self.debug:
            self.logs_folder, self.output_folder, self.frames_folder = self.set_debug_folders()

        self.get_logger().info("PathPlanningPlus3 initialized (Nav2 action client)")

    def _goal_timer_callback(self):
        if self._latest_goal is None:
            return
        self.goal_pub.publish(self._latest_goal)
        self._latest_goal = None


    def costmap_callback(self, msg: OccupancyGrid):
        data = np.array(msg.data, dtype=np.int8)
        self.costmap_has_obstacles = bool(np.any(data > 0))
        

    def set_debug_folders(self):
        try:
            # absolute, user-controlled root
            self.debug_root.mkdir(parents=True, exist_ok=True)

            # time-stamped run folder
            ts_folder = self.debug_root / (datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_plus3')
            ts_folder.mkdir()

            # sub-folders
            logs  = ts_folder / 'logs'
            out   = ts_folder / 'output'
            frames= ts_folder / 'frames'
            for p in (logs, out, frames):
                p.mkdir()

            # show where we’re writing
            self.get_logger().info(
                f'DEBUG output =>\n  logs:   {logs}\n  output: {out}\n  frames: {frames}'
            )
            return str(logs), str(out), str(frames)

        except Exception as e:
            self.get_logger().error(f'Failed to create debug folders: {e}')
            raise    
        
    def original_image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

    def image_callback(self, data):
        if self.cv_image is None:
            return

        mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        line_edges = processing_mask(mask, self.cv_image, show=False)

        birds_eye_msg = self.bridge.cv2_to_imgmsg(line_edges, encoding="mono8")
        
        self.bev_pub.publish(birds_eye_msg)
        
        lateral_distance, longitudinal_distance, midpoints = computing_lateral_distance(
            line_edges, show=False)

        if SHOW and midpoints is not None:
            posm = midpoints[-1]
            for p in midpoints[:-1]:
                cv2.circle(line_edges, tuple(p[::-1]), 2, (200, 200, 200), 3)
            cv2.circle(line_edges, tuple(posm[::-1]), 2, (255, 255, 255), 5)

        if lateral_distance == -np.inf:
            self.get_logger().warn("Lane lost, emergency steering angle -10.0") 
            degree_steering_angle = -10.0

        elif lateral_distance == np.inf:
            self.get_logger().warn("Lane lost, emergency steering angle 10.0") 
            degree_steering_angle = 10.0
        else:
            degree_steering_angle = None


        if self.costmap_has_obstacles:
            if lateral_distance in (-np.inf, np.inf):
                self.get_logger().warn("Obstacles are present but lane lost")
                return
            # Build goal in camera_link frame, then transform to map
            goal_camera = PoseStamped()
            goal_camera.header.stamp =data.header.stamp  
            goal_camera.header.frame_id = "hero/rgb_front"
            goal_camera.pose.position.x = float(longitudinal_distance)
            goal_camera.pose.position.y = float(-lateral_distance)
            goal_camera.pose.orientation.w = 1.0

            try:
                goal_map = self.tf_buffer.transform(goal_camera, "map",
                                                    timeout=rclpy.duration.Duration(seconds=0.1))
            except Exception as e:
                self.get_logger().warn(f"TF camera->map failed: {e}")
                return

            self._latest_goal = goal_map
            self.goal_published = True
            
        else:
            self.goal_published = False
            self._latest_goal = None
            if degree_steering_angle is None:
                # Pure pursuit angle
                dist = (longitudinal_distance + self.gain) ** 2 + lateral_distance ** 2
                degree_steering_angle = math.degrees(math.atan2(2 * self.wheelbase * lateral_distance, dist))
            steer_msg = Float64()
            steer_msg.data = degree_steering_angle
            self.steer_pub.publish(steer_msg)
        if self.debug:
            self._save_debug(mask, line_edges, lateral_distance,
                             longitudinal_distance, midpoints, degree_steering_angle)
            
    def _save_debug(self, mask, line_edges, lateral_distance,
                    longitudinal_distance, midpoints, degree_steering_angle):
        if midpoints is not None:
            posm = midpoints[-1]
            midpoints = midpoints[:-1]
            cv2.circle(line_edges, tuple(posm[::-1]), 2, (255, 255, 255), 5)
            for p in midpoints:
                cv2.circle(line_edges, tuple(p[::-1]), 2, (200, 200, 200), 3)

        resized_image = cv2.resize(self.cv_image, (540, 360))
        resized_mask = cv2.resize(mask, (540, 360))
        resized_line_edges = cv2.resize(line_edges, (540, 360))
        concatenated_image = np.hstack((
            cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(resized_line_edges, cv2.COLOR_GRAY2BGR)
        ))

        cv2.imwrite(
            os.path.join(self.frames_folder, f"frame_{self.counter}.png"),
            self.cv_image)
        cv2.imwrite(
            os.path.join(self.output_folder, f"output_{self.counter}.png"),
            concatenated_image)

        log_file = os.path.join(self.logs_folder, f"log_{self.counter}.txt")
        with open(log_file, "w") as log:
            if degree_steering_angle is not None:
                log.write(f"{self.counter}: lateral={lateral_distance:.3f} "
                      f"longitudinal={longitudinal_distance:.3f} steering_degree_command={degree_steering_angle:.3f} time={datetime.now().isoformat()}\n")
            else:
                log.write(f"{self.counter}: lateral={lateral_distance:.3f} "
                      f"longitudinal={longitudinal_distance:.3f} steering_degree_command=None(nav2 command-obstacle_avoidance) time={datetime.now().isoformat()}\n")
        self.counter +=1

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down path_planning_plus3")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


