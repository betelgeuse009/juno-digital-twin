#!/usr/bin/env python3

import os
import math
import csv
import numpy as np
import cv2
from datetime import datetime
from enum import Enum, auto

from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from pyproj import Transformer
from shared_objects.new_utils import computing_lateral_distance, processing_mask
from shared_objects.ROS_utils import Topics, SHOW

ARRIVAL_THRESHOLD_M = 1.5
GPS_MSG_TIMEOUT_S = 5.0
MIN_RTK_STATUS = 2
STDDEV_MAX_M = 0.1
DEGRADE_DWELL_S = 3.0
RECOVER_DWELL_S = 3.0


class Mode(Enum):
    GPS_NAV = auto()
    VISION = auto()


class PathPlanningNode(Node):
    def __init__(self):
        super().__init__("path_planning_plus5")

        self.debug = self.declare_parameter("debug", True).value
        self.wheelbase = self.declare_parameter("wheelbase", 1.6).value
        self.gain = self.declare_parameter("gain", 0.0).value

        self.bridge = CvBridge()
        self.cv_image = None
        self.counter = 0

        self.declare_parameter(
            'debug_root',
            '/home/ubuntu/Workspace/ros-bridge/src/DEBUG'   # default
        )
        self.debug_root = Path(
        self.get_parameter('debug_root').get_parameter_value().string_value)
        

        topics = Topics()
        self.topic_names = topics.topic_names

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # GPS coordinate transform (WGS84 -> UTM zone 32N for Italy)
        # But from CARLA GNSS values it seems we are in EPSG:32631 (idk why but Zone = [(Longitude+ 180)/6] +1  )
        self.transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:32631", always_xy=True
        )
        self.datum_east = None
        self.datum_north = None
        self.datum_set = False

        # Waypoints
        self.waypoints = self._load_waypoints("/home/ubuntu/Workspace/ros-bridge/track_waypoints.csv")
        self.wp_index = 0

        # Mode
        self.mode = Mode.VISION
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._last_gps_msg_time = None 
        self._degraded_since = None
        self._good_since = None
        self.initial_snap_done = False
        # The single goal that Nav2 is currently pursuing
        self._current_goal = None

        self.create_subscription(NavSatFix, "/carla/hero/gnss", self._gps_cb, 10)
        self.create_subscription(Image, self.topic_names["segmented_image"], self._seg_image_cb, 1)
        self.create_subscription( Image, "/carla/hero/rgb_front/image", self._rgb_image_cb, 1)

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 1)
        self.bev_pub = self.create_publisher(Image, "/birds_eye_view", 1)
        self.mode_pub = self.create_publisher(String, "/nav_mode", 1)

        # Republish current goal to Nav2 at 5 Hz
        self.create_timer(0.2, self._goal_timer_cb)

        # Arrival check at 5 Hz (GPS mode only)
        self.create_timer(0.2, self._arrival_check_cb)

        # GPS health watchdog at 1 Hz
        self.create_timer(1.0, self._gps_watchdog_cb)

        if self.debug:
            self.logs_folder, self.output_folder, self.frames_folder = (self.set_debug_folders())

        self.get_logger().info("plus5 ready: Nav2 always active, goal source switches")

    def _load_waypoints(self, csv_path: str) -> np.ndarray:
        rows = []
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)

        lats = [float(r["latitude"]) for r in rows]
        lons = [float(r["longitude"]) for r in rows]
        eastings, northings = self.transformer.transform(lons, lats)

        headings = []
        for i in range(len(rows)):
            if i < len(rows) - 1:
                headings.append(
                    math.atan2(
                        northings[i + 1] - northings[i],
                        eastings[i + 1] - eastings[i],
                    )
                )
            else:
                headings.append(headings[-1] if headings else 0.0)

        wp = np.column_stack([lats, lons, headings])
        self.get_logger().info(f"Loaded {len(wp)} waypoints from {csv_path}")
        return wp

    def _gps_cb(self, msg: NavSatFix):
        # commented since there is no RTK in Carla
        #if msg.status.status < 0:
        #    return

        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s <= 0:
            return

        self._last_gps_msg_time = now_s
        # covariance_type 0 = UNKNOWN → treat as degraded
        if msg.position_covariance_type > 0:
            stddev_e = math.sqrt(max(0.0, msg.position_covariance[0]))
            stddev_n = math.sqrt(max(0.0, msg.position_covariance[4]))
            stddev_bad = max(stddev_e, stddev_n) > STDDEV_MAX_M
        else:
            stddev_bad = False

        degraded = (msg.status.status < 0) or stddev_bad

        if not self.datum_set and not degraded:
            e, n = self.transformer.transform(msg.longitude, msg.latitude)
            self.datum_east = e
            self.datum_north = n
            self.datum_set = True
            self.get_logger().info(f"Datum: E={e:.2f} N={n:.2f}")

        if degraded:
            if self._degraded_since is None:
                self._degraded_since = now_s
            self._good_since = None
            if (self.mode == Mode.GPS_NAV
                    and (now_s - self._degraded_since) >= DEGRADE_DWELL_S):
                self._switch_mode(Mode.VISION)
        else:
            if self._good_since is None:
                self._good_since = now_s
            self._degraded_since = None
            if (self.mode == Mode.VISION
                    and (now_s - self._good_since) >= RECOVER_DWELL_S):
                self._switch_mode(Mode.GPS_NAV)

    def _gps_watchdog_cb(self):
        if self.mode != Mode.GPS_NAV:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9

        if self._last_gps_msg_time is not None and (now_s - self._last_gps_msg_time) > GPS_MSG_TIMEOUT_S:
            self.get_logger().warn(
                f"No GPS msg for >{GPS_MSG_TIMEOUT_S}s, falling back to VISION"
            )
            self._switch_mode(Mode.VISION)

    def _switch_mode(self, new_mode: Mode):
        if new_mode == self.mode:
            return
        prev = self.mode
        self.mode = new_mode
        self._current_goal = None
        self._degraded_since = None
        self._good_since = None

        if new_mode == Mode.GPS_NAV and prev == Mode.VISION:
            self._snap_wp_to_nearest_ahead()

        self.get_logger().info(f"Mode {prev.name} -> {new_mode.name}")
        mode_msg = String()
        mode_msg.data = new_mode.name
        self.mode_pub.publish(mode_msg)

    def _snap_wp_to_nearest_ahead(self):
        if not self.datum_set or self.wp_index >= len(self.waypoints):
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "hero", rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception as e:
            self.get_logger().warn(
                f"wp snap: TF map->base_link failed ({e}), keeping wp_index={self.wp_index}"
            )
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        fx, fy = math.cos(yaw), math.sin(yaw)

        best_idx = None
        best_dist = float("inf")
        for i in range(self.wp_index, len(self.waypoints)):
            lat, lon, _ = self.waypoints[i]
            e, n = self.transformer.transform(lon, lat)
            dx = (e - self.datum_east) - rx
            dy = (n - self.datum_north) - ry
            if dx * fx + dy * fy <= 0.0:
                continue
            d = math.hypot(dx, dy)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_idx != self.wp_index:
            self.get_logger().info(
                f"wp snap: {self.wp_index} -> {best_idx} (dist {best_dist:.1f}m)"
            )
            self.wp_index = best_idx

    def _make_gps_goal(self) -> PoseStamped:
        lat, lon, yaw = self.waypoints[self.wp_index]
        e, n = self.transformer.transform(lon, lat)
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = e - self.datum_east
        pose.pose.position.y = n - self.datum_north
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def _arrival_check_cb(self):
        if self.mode != Mode.GPS_NAV:
            return
        if not self.datum_set or self.wp_index >= len(self.waypoints):
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "hero", rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return

        goal = self._make_gps_goal()
        dx = goal.pose.position.x - tf.transform.translation.x
        dy = goal.pose.position.y - tf.transform.translation.y

        if math.hypot(dx, dy) < ARRIVAL_THRESHOLD_M:
            self.wp_index += 1
            remaining = len(self.waypoints) - self.wp_index
            if remaining > 0:
                self.get_logger().info(
                    f"Waypoint {self.wp_index}/{len(self.waypoints)} reached"
                )
            else:
                self.get_logger().info("All waypoints reached")
    
    def _rgb_image_cb(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

    def _seg_image_cb(self, data):
        if self.cv_image is None:
            return

        mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        line_edges = processing_mask(mask, self.cv_image, show=False)
        self.bev_pub.publish(self.bridge.cv2_to_imgmsg(line_edges, encoding="mono8"))

        lateral_distance, longitudinal_distance, midpoints = (
            computing_lateral_distance(line_edges, show=False)
        )

        if SHOW and midpoints is not None:
            for p in midpoints[:-1]:
                cv2.circle(line_edges, tuple(p[::-1]), 2, (200, 200, 200), 3)
            cv2.circle(
                line_edges, tuple(midpoints[-1][::-1]), 2, (255, 255, 255), 5
            )

        if self.debug:
            self._save_debug(
                mask, line_edges, lateral_distance, longitudinal_distance, midpoints)

        # Only update vision goals when in VISION mode
        if self.mode != Mode.VISION:
            return

        if lateral_distance in (-np.inf, np.inf):
            if lateral_distance == -np.inf:
                degree_steering_angle = -10.0
            elif lateral_distance == np.inf:
                degree_steering_angle = 10.0
            return

        # Build goal in camera frame, then transform to odom
        goal_cam = PoseStamped()
        goal_cam.header.stamp = data.header.stamp
        goal_cam.header.frame_id = "hero/rgb/front"
        goal_cam.pose.position.x = float(longitudinal_distance)
        goal_cam.pose.position.y = float(-lateral_distance)
        goal_cam.pose.orientation.w = 1.0

        try:
            goal_odom = self.tf_buffer.transform(
                goal_cam, "odom", timeout=Duration(seconds=0.1)
            )
            self._current_goal = goal_odom
        except Exception as e:
            self.get_logger().warn(f"TF camera->odom failed: {e}")

    def _goal_timer_cb(self):
        if self.mode == Mode.GPS_NAV:
            if not self.datum_set or self.wp_index >= len(self.waypoints):
                return
            self._current_goal = self._make_gps_goal()

        if not self.tf_buffer.can_transform("map", "hero", rclpy.time.Time()):
            return
        if not self.initial_snap_done:
                self._snap_wp_to_nearest_ahead()
                self.initial_snap_done = True
        # In VISION mode, _current_goal is set by _seg_image_cb

        if self._current_goal is not None:
            self.goal_pub.publish(self._current_goal)


    def set_debug_folders(self):
        try:
            # absolute, user-controlled root
            self.debug_root.mkdir(parents=True, exist_ok=True)

            # time-stamped run folder
            ts_folder = self.debug_root / (datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_plus5')
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

    def _save_debug(self, mask, line_edges, lateral_distance, longitudinal_distance, midpoints):
        if midpoints is not None:
            posm = midpoints[-1]
            midpoints = midpoints[:-1]
            cv2.circle(line_edges, tuple(posm[::-1]), 2, (255, 255, 255), 5)
            for p in midpoints:
                cv2.circle(line_edges, tuple(p[::-1]), 2, (200, 200, 200), 3)

        resized = cv2.resize(self.cv_image, (540, 360))
        concat = np.hstack((
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.resize(mask, (540, 360)), cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(cv2.resize(line_edges, (540, 360)), cv2.COLOR_GRAY2BGR),
        ))
        cv2.imwrite(
            os.path.join(self.frames_folder, f"frame_{self.counter}.png"),
            self.cv_image,
        )
        cv2.imwrite(
            os.path.join(self.output_folder, f"output_{self.counter}.png"), concat
        )
        with open(
            os.path.join(self.logs_folder, f"log_{self.counter}.txt"), "w"
        ) as f:
            f.write(
                f"{self.counter}: lat={lateral_distance:.3f} "
                f"lon={longitudinal_distance:.3f}\n"
            )
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down plus5")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
