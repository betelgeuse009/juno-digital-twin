#!/usr/bin/env python3.10
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float32
from cv_bridge import CvBridge
import math
from datetime import datetime
#modify the directory of shared_objects
import os
import sys
from pathlib import Path
from nav_msgs.msg import OccupancyGrid
from shared_objects.ROS_utils import Topics, SHOW

from shared_objects.new_utils_midpoint_avoidance_path import computing_lateral_distance, processing_mask
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Initialize topics and node-related variables
topics = Topics()
topic_names = topics.topic_names

# Node class
class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')
        self.wheelbase = 1.6
        self.speed = 10.0
        self.gain = 1.1
        self.DEBUG = True

        self._in_obs_mode = False
        self.LETHAL_THRESHOLD = 90      # cells with cost >= this count as obstacle
        self.MIN_LETHAL_CELLS = 3       # need >= N lethal cells
        self._raw_obstacle_history = [False, False, False]

        qos_profile_floats = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        qos_profile_images = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.steer_pub = self.create_publisher(Float64, topic_names['steering'], qos_profile_floats)
        self.req_speed_pub = self.create_publisher(Float32, topic_names['requested_speed'], qos_profile_floats)
        self.bev_pub = self.create_publisher(Image, "/birds_eye_view", qos_profile_images)
        self.image_sub = self.create_subscription(Image, topic_names['segmented_image'], self.image_callback, qos_profile_images)
        self.original_image_sub = self.create_subscription(Image, '/carla/hero/rgb_front/image', self.original_image_callback, qos_profile_images)
        self.costmap_sub = self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, qos_profile_floats)

        self.bridge = CvBridge()
        self.counter = 0
        self.cv_image = None 

        self.declare_parameter(
            'debug_root',
            '/home/ubuntu/Workspace/ros-bridge/src/DEBUG'   # default
        )
        self.debug_root = Path(
        self.get_parameter('debug_root').get_parameter_value().string_value
        )
        
        if self.DEBUG:
            self.logs_folder, self.output_folder, self.frames_folder = self.set_debug_folders()

        # Initial speed
        req_speed_msg = Float32()
        req_speed_msg.data = self.speed
        self.req_speed_pub.publish(req_speed_msg)
    
    def costmap_callback(self, msg: OccupancyGrid):
        data = np.frombuffer(bytes(msg.data), dtype=np.int8)  # faster than np.array on list
        raw = int(np.sum(data >= self.LETHAL_THRESHOLD)) >= self.MIN_LETHAL_CELLS
        self._raw_obstacle_history.append(raw)
        self._raw_obstacle_history.pop(0)
        
        # Only flip when all recent frames agree
        if all(self._raw_obstacle_history) and not self._in_obs_mode:
            self.get_logger().warn("OBSTACLE_AVOID - Filtering active")
            self._in_obs_mode = True
            req_speed_msg = Float32()
            req_speed_msg.data = 4.0
            self.req_speed_pub.publish(req_speed_msg)
        elif not any(self._raw_obstacle_history) and self._in_obs_mode:
            self.get_logger().info("LANE_FOLLOW - Filtering inactive")
            self._in_obs_mode = False
            req_speed_msg = Float32()
            req_speed_msg.data = 12.0
            self.req_speed_pub.publish(req_speed_msg)

    def set_debug_folders(self):
        try:
            # absolute, user-controlled root
            self.debug_root.mkdir(parents=True, exist_ok=True)

            # time-stamped run folder
            ts_folder = self.debug_root / datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            ts_folder.mkdir()

            # sub-folders
            logs  = ts_folder / 'logs'
            out   = ts_folder / 'output'
            frames= ts_folder / 'frames'
            for p in (logs, out, frames):
                p.mkdir()

            # show where we’re writing
            self.get_logger().info(
                f'DEBUG output →\n  logs:   {logs}\n  output: {out}\n  frames: {frames}'
            )
            return str(logs), str(out), str(frames)

        except Exception as e:
            self.get_logger().error(f'Failed to create debug folders: {e}')
            raise

    def original_image_callback(self, data):
        # Store the RGB image
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

    def image_callback(self, data):
        # Convert ROS Image message to OpenCV image
        mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        if self.cv_image is None:
            self.get_logger().warn("Image not recieved from original_image_callback")
            return
        
        line_edges = processing_mask(mask, self.cv_image)

        birds_eye_msg = self.bridge.cv2_to_imgmsg(line_edges, encoding="mono8")
        
        self.bev_pub.publish(birds_eye_msg)
        
        # Calculate distances and midpoint
        lateral_distance, longitudinal_distance, midpoints = computing_lateral_distance(
            line_edges, 
            obstacle_seen=self._in_obs_mode, 
            show=SHOW
        )

        if lateral_distance == -np.inf:
            degree_steering_angle = -10.0
        elif lateral_distance == np.inf:
            degree_steering_angle = 10.0
        else:
            distance_to_waypoint = (longitudinal_distance + self.gain) ** 2 + lateral_distance ** 2
            degree_steering_angle = math.degrees(math.atan2(2 * self.wheelbase * lateral_distance, distance_to_waypoint))

        # Debug mode image display and logging
        if self.DEBUG:
            if midpoints is not None:
                posm = midpoints[-1]
                midpoints = midpoints[:-1]
                cv2.circle(line_edges, tuple(posm[::-1]), 2, (255, 255, 255), 5)
                for p in midpoints:
                    cv2.circle(line_edges, tuple(p[::-1]), 2, (200, 200, 200), 3)

            resized_image = cv2.resize(self.cv_image, (540, 360))
            resized_mask = cv2.resize(mask, (540, 360))
            resized_line_edges = cv2.resize(line_edges, (540, 360))
            concatenated_image = np.hstack((cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB),
                                            cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB),
                                            cv2.cvtColor(resized_line_edges, cv2.COLOR_GRAY2BGR)))
            steer_label = f"Steer: {degree_steering_angle:.2f} deg, Lateral Distance: {lateral_distance:.2f} m"
            (text_w, text_h), baseline = cv2.getTextSize(
                steer_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            cv2.rectangle(
                concatenated_image,
                (8, 8),
                (14 + text_w, 18 + text_h + baseline),
                (0, 0, 0),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                concatenated_image,
                steer_label,
                (10, 12 + text_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Save frame and output images
            frame_name = f"frame_{self.counter}.png"
            frame_path = os.path.join(self.frames_folder, frame_name)
            cv2.imwrite(frame_path,self.cv_image)

            output_name = f"output_{self.counter}.png"
            output_path = os.path.join(self.output_folder, output_name)
            cv2.imwrite(output_path, concatenated_image)

            # Log data to file
            log_file = os.path.join(self.logs_folder, f"log_{self.counter}.txt")
            with open(log_file, "w") as log:
                log.write(f"{self.counter}: - lateral_distance : {lateral_distance} - longitudinal_distance: {longitudinal_distance} - degree_steering_angle: {degree_steering_angle}\n")
            self.counter += 1

        # Display concatenated image if SHOW is enabled
        if SHOW:
            resized_image = cv2.resize(self.cv_image, (540, 360))
            resized_mask = cv2.resize(mask, (540, 360))
            resized_line_edges = cv2.resize(line_edges, (540, 360))
            concatenated_image = np.hstack((cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB),
                                            cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB),
                                            cv2.cvtColor(resized_line_edges, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("Denedik :(", concatenated_image)
            cv2.waitKey(1)

        # Publish the steering angle
        steer_msg = Float64()
        steer_msg.data = degree_steering_angle
        self.steer_pub.publish(steer_msg)

        # Log information to the console
        self.get_logger().info(f"Longitudinal Distance: {longitudinal_distance}")
        self.get_logger().info(f"Steering Angle: {degree_steering_angle}")

    def new_method(self):
        self.cv_image
        return 

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Path Planning Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
