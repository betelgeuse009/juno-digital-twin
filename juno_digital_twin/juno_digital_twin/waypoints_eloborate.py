import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


class WaypointRecorder(Node):
    def __init__(self):
        super().__init__("waypoint_recorder")

        # Subscribe to GPS fixes and process each message in gps_callback.
        self.subscription = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gps_callback, 10
        )
        # Recording and quality thresholds.
        self.min_dist = self._param_to_float("min_dist_m", 1.0)
        self.min_heading_change_deg = self._param_to_float(
            "min_heading_change_deg", 12.0
        )
        self.min_turn_dist_m = self._param_to_float("min_turn_dist_m", 0.4)
        self.max_horizontal_accuracy_m = self._param_to_float(
            "max_horizontal_accuracy_m", 0.10
        )
        self.max_vertical_accuracy_m = self._param_to_float(
            "max_vertical_accuracy_m", 0.10
        )
        self.max_speed_mps = self._param_to_float("max_speed_mps", 65.0)
        self.max_jump_m = self._param_to_float("max_jump_m", 8.0)
        
        _repo_root = Path(__file__).resolve().parents[3]

        self.output_dir = Path(
            str(self.declare_parameter("output_dir", str(_repo_root / "tracks")).value)
        )

        self.waypoints = []
        self.records = []
        self.get_logger().info("Waypoint Recorder Started. Waiting for RTK Fix...")

    def _param_to_float(self, name: str, default: float) -> float:
        """Helper to make sure we never pass a nullable as a param"""
        val = self.declare_parameter(name, default).value
        if val is None:
            self._logger.error(
                f"Parameter {name} can't be None. Passing to the default"
            )
            return default
        return float(val)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Great-circle distance on Earth in meters between two lat/lon coordinates.
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def bearing_deg(self, lat1, lon1, lat2, lon2):
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lon2 - lon1)
        x = math.sin(delta_lambda) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(
            phi2
        ) * math.cos(delta_lambda)
        return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

    def angular_diff_deg(self, a, b):
        diff = abs(a - b) % 360.0
        return min(diff, 360.0 - diff)

    def extract_accuracy(self, msg):
        var_h = max(msg.position_covariance[0], msg.position_covariance[4])
        var_v = msg.position_covariance[8]
        h_acc = math.sqrt(var_h) if var_h > 0.0 else float("inf")
        v_acc = math.sqrt(var_v) if var_v > 0.0 else float("inf")
        return h_acc, v_acc

    def gps_callback(self, msg):
        # Record points only when RTK-quality fix is available (status == 2).
        if msg.status.status < 0:
            self.get_logger().warn("Waiting for RTK fix...", throttle_duration_sec=5.0)
            return

        lat, lon, alt = msg.latitude, msg.longitude, msg.altitude
        h_acc, v_acc = self.extract_accuracy(msg)
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # Item 1: accuracy gate in addition to RTK status.
        if (
            h_acc > self.max_horizontal_accuracy_m
            or v_acc > self.max_vertical_accuracy_m
        ):
            self.get_logger().warn(
                f"Skipping low-quality RTK sample: h_acc={h_acc:.3f}m v_acc={v_acc:.3f}m",
                throttle_duration_sec=2.0,
            )
            return

        # First accepted point becomes the route start.
        if not self.waypoints:
            self.waypoints.append([lat, lon, alt])
            self.records.append(
                {
                    "index": 0,
                    "stamp_sec": stamp_sec,
                    "status": int(msg.status.status),
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "h_acc": h_acc,
                    "v_acc": v_acc,
                    "dist_from_prev_m": 0.0,
                    "heading_change_deg": float("nan"),
                    "reason": "start",
                }
            )
            self.get_logger().info(
                f"Start Point: {lat:.6f}, {lon:.6f} | h_acc={h_acc:.3f}m"
            )
            return

        # Compare current position with last stored waypoint.
        last_lat, last_lon, _ = self.waypoints[-1]
        dist = self.haversine_distance(last_lat, last_lon, lat, lon)
        last_stamp_sec = self.records[-1]["stamp_sec"]
        dt = stamp_sec - last_stamp_sec

        # Item 2: reject outlier jumps and speed spikes.
        if dist > self.max_jump_m:
            self.get_logger().warn(
                f"Rejected outlier jump: {dist:.2f}m", throttle_duration_sec=1.0
            )
            return
        if dt > 0.0 and dist / dt > self.max_speed_mps:
            self.get_logger().warn(
                f"Rejected speed outlier: {dist / dt:.2f}m/s",
                throttle_duration_sec=1.0,
            )
            return

        # Item 3: save by distance or by heading change (corner capture).
        heading_change = None
        by_heading = False
        if len(self.waypoints) >= 2:
            prev_lat, prev_lon, _ = self.waypoints[-2]
            prev_heading = self.bearing_deg(prev_lat, prev_lon, last_lat, last_lon)
            curr_heading = self.bearing_deg(last_lat, last_lon, lat, lon)
            heading_change = self.angular_diff_deg(prev_heading, curr_heading)
            by_heading = (
                dist >= self.min_turn_dist_m
                and heading_change >= self.min_heading_change_deg
            )

        by_dist = dist >= self.min_dist
        if by_dist or by_heading:
            reason = "distance" if by_dist else "turn"
            self.waypoints.append([lat, lon, alt])
            self.records.append(
                {
                    "index": len(self.waypoints) - 1,
                    "stamp_sec": stamp_sec,
                    "status": int(msg.status.status),
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "h_acc": h_acc,
                    "v_acc": v_acc,
                    "dist_from_prev_m": dist,
                    "heading_change_deg": (
                        heading_change if heading_change is not None else float("nan")
                    ),
                    "reason": reason,
                }
            )
            self.get_logger().info(
                f"Recorded: {lat:.6f}, {lon:.6f} | dist={dist:.2f}m reason={reason}"
            )

    def save_waypoints(self):
        # Items 5 & 6: save richer output in deterministic folder with run id.
        if self.waypoints:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            arr = np.array(self.waypoints, dtype=np.float64)
            npy_path = self.output_dir / f"track_waypoints_{run_id}.npy"
            csv_path = self.output_dir / f"track_waypoints_{run_id}.csv"

            np.save(npy_path, arr)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "index",
                        "stamp_sec",
                        "status",
                        "lat",
                        "lon",
                        "alt",
                        "h_acc",
                        "v_acc",
                        "dist_from_prev_m",
                        "heading_change_deg",
                        "reason",
                    ],
                )
                writer.writeheader()
                writer.writerows(self.records)

            # Stable names for tooling that expects fixed filenames.
            np.save(self.output_dir / "track_waypoints.npy", arr)
            with open(
                self.output_dir / "track_waypoints.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "index",
                        "stamp_sec",
                        "status",
                        "lat",
                        "lon",
                        "alt",
                        "h_acc",
                        "v_acc",
                        "dist_from_prev_m",
                        "heading_change_deg",
                        "reason",
                    ],
                )
                writer.writeheader()
                writer.writerows(self.records)

            self.get_logger().info(
                f"Saved {len(arr)} waypoints to {npy_path.name} and {csv_path.name} in {self.output_dir}."
            )
        else:
            self.get_logger().warn("No RTK waypoints were recorded.")


def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Always save what has been collected before shutting down.
        node.save_waypoints()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()