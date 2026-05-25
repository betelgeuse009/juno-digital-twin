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
from shared_objects.new_utils import computing_lateral_distance, processing_mask
from shared_objects.ROS_utils import Topics, SHOW
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32, Float64, Bool
import tf2_geometry_msgs
from pathlib import Path 
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_plus3')

        self.debug = self.declare_parameter('debug', True).value
        self.wheelbase = self.declare_parameter('wheelbase', 1.6).value
        self.gain = self.declare_parameter('gain', 1.1).value

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
        self.steer_pub = self.create_publisher(Float64, self.topic_names["steering"], 1)
        self.bev_pub = self.create_publisher(Image, "/birds_eye_view", 10)
        
        # Nav2 action client
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        latched_qos = QoSProfile(depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.nav_mode_pub = self.create_publisher(Bool, '/nav_mode', latched_qos)
        

        # state machine bookkeeping
        self._in_obs_mode = False
        self._current_goal_handle = None
        self._last_sent_goal = None 
        # costmap filtering values 
        self.LETHAL_THRESHOLD = 90      # cells with cost >= this count as obstacle
        self.MIN_LETHAL_CELLS = 3       # need >= N lethal cells
        self.CONSECUTIVE_FRAMES = 3     # require N agreeing frames
        self.GOAL_UPDATE_DIST_M = 5.0   # only update active goal if target moved > this
        # added this because we saw from track test that it was too easy to switch from noise
        self._raw_obstacle_history = [False, False, False]

        if self.debug:
            self.logs_folder, self.output_folder, self.frames_folder = self.set_debug_folders()
        self.get_logger().info("PathPlanningPlus3 initialized (Nav2 action client)")
    
    def costmap_callback(self, msg: OccupancyGrid):
        data = np.frombuffer(bytes(msg.data), dtype=np.int8)  # faster than np.array on list
        raw = int(np.sum(data >= self.LETHAL_THRESHOLD)) >= self.MIN_LETHAL_CELLS
        self._raw_obstacle_history.append(raw)
        self._raw_obstacle_history.pop(0)
        # Only flip when all recent frames agree
        if all(self._raw_obstacle_history) and not self._in_obs_mode:
            self._enter_obs_mode()
        elif not any(self._raw_obstacle_history) and self._in_obs_mode:
            self._exit_obs_mode()        

    def _enter_obs_mode(self):
        self.get_logger().warn("OBSTACLE_AVOID")
        self._in_obs_mode = True
        m = Bool(); m.data = True
        self.nav_mode_pub.publish(m)
        # actual goal is sent on the next valid image_callback

    def _exit_obs_mode(self):
        self.get_logger().info("LANE_FOLLOW")
        # cancel first, then flip the bridge off
        if self._current_goal_handle is not None:
            cancel_future = self._current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._after_cancel)
        else:
            self._after_cancel(None)

    def _after_cancel(self, _future):
        self._current_goal_handle = None
        self._last_sent_goal = None
        self._in_obs_mode = False
        m = Bool(); m.data = False
        self.nav_mode_pub.publish(m)

    def _send_goal(self, pose_map: PoseStamped):
        if not self.nav_action_client.server_is_ready():
            # Cheap non-blocking check; wait_for_server would block the executor
            self.get_logger().warn("Nav2 action server not ready, skipping goal send")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_map
        goal_msg.behavior_tree = ''   # use bt_navigator default (your custom XML)

        send_future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_cb,
        )
        send_future.add_done_callback(self._goal_response_cb)
        self._last_sent_goal = pose_map
        self.get_logger().info(
            f"Sent Nav2 goal: x={pose_map.pose.position.x:.2f}, "
            f"y={pose_map.pose.position.y:.2f}"
        )

        def _goal_response_cb(self, future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Nav2 goal rejected")
                self._current_goal_handle = None
                return
            self._current_goal_handle = goal_handle
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._goal_result_cb)

        def _goal_result_cb(self, future):
            status = future.result().status
            # 4 = SUCCEEDED, 5 = CANCELED, 6 = ABORTED
            self.get_logger().info(f"Nav2 goal finished with status={status}")
            self._current_goal_handle = None
            # Don't auto-exit obs mode here — let the costmap callback decide.
            # The goal will be re-sent on the next image_callback if obstacles persist.

        def _nav_feedback_cb(self, feedback_msg):
            # Optional: log distance_remaining, navigation_time
            pass

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
            self.get_logger().warn("Lane lost, emergency steering angle +10.0") 
            degree_steering_angle = 10.0

        elif lateral_distance == np.inf:
            self.get_logger().warn("Lane lost, emergency steering angle -10.0") 
            degree_steering_angle = -10.0
        else:
            degree_steering_angle = None


        if self._in_obs_mode:
            self.get_logger().info("Obstacles found switching to obstacle avoidance mode")
            if lateral_distance in (-np.inf, np.inf):
                self.get_logger().warn("Obstacles are present but lane lost")
                return
            # Build goal in camera_link frame, then transform to map
            goal_camera = PoseStamped()
            goal_camera.header.stamp = self.get_clock().now().to_msg()
            goal_camera.header.frame_id = "hero"
            goal_camera.pose.position.x = 15.0
            goal_camera.pose.position.y = 0.0
            goal_camera.pose.position.z = 0.0
            goal_camera.pose.orientation.x = 0.0
            goal_camera.pose.orientation.y = 0.0
            goal_camera.pose.orientation.z = 0.0
            goal_camera.pose.orientation.w = 1.0

            try:
                goal_map = self.tf_buffer.transform(goal_camera, "map",
                                                    timeout=rclpy.duration.Duration(seconds=0.2))
            except Exception as e:
                self.get_logger().warn(f"TF camera->map failed: {e}")
                return
            self._latest_goal = goal_map
            if self._should_update_goal(goal_map):
                self._send_goal(goal_map)
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
            self.get_logger().info(f"Published steering angle: {degree_steering_angle}, Lookahead distance: {longitudinal_distance}")
        if self.debug:
            self._save_debug(mask, line_edges, lateral_distance,
                             longitudinal_distance, midpoints, degree_steering_angle)
            
    def _should_update_goal(self, new_goal: PoseStamped) -> bool:
        if self._last_sent_goal is None:
            return True
        dx = new_goal.pose.position.x - self._last_sent_goal.pose.position.x
        dy = new_goal.pose.position.y - self._last_sent_goal.pose.position.y
        return (dx*dx + dy*dy) > (self.GOAL_UPDATE_DIST_M ** 2)

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


